"""
EMG Streaming Client con Análisis en Tiempo Real
Alejandro Solar Iglesias

Cliente de adquisición de EMG en tiempo real con soporte para:
- Servidor HTTP Noraxon (127.0.0.1:9220/samples)
- Procesamiento de señal: filtro bandpass (20-450 Hz), envolvente, features
- Extracción de features: RMS, MAV, IEMG, WL, ZC, Peak, DAMV (time-domain)
                          MNF, MDF, Peak_Freq, Total_Power, PowerBelow100Hz (frequency-domain)
- Detección de fatiga basada en MDF (Median Frequency)
- Guardado opcional de RMS y features en CSV
- Modo simulación para pruebas sin servidor

Mejoras técnicas:
- Reconexión automática con backoff exponencial al servidor HTTP
- Parseo flexible del JSON de Noraxon (canal con index, sampleindex, samples)
- Filtros IIR con estados persistentes por canal (continuidad en streaming)
- Ventanas deslizantes configurables para extracción de features
- Buffer circular por canal para procesamiento eficiente
- Detección de fatiga: baseline MDF en primeras ~50 ventanas, alerta si MDF < baseline × threshold

EJEMPLOS DE USO:
    # Con MVC de 500 µV (default)
    python EMG_RT.py --simulate --verbose --show-mvc

    # Con MVC personalizado (ej: 300 µV)
    python EMG_RT.py --simulate --verbose --show-mvc --mvc 300

    # Tiempo real con MVC personalizado
    python EMG_RT.py --ip localhost --port 9220 --endpoint /samples --verbose --show-mvc --mvc 300

PARÁMETROS PRINCIPALES:
  --ip, --port              Servidor HTTP (default: localhost:9220)
  --endpoint                Ruta HTTP (default: /features, usar /samples para Noraxon)
  --fs                      Frecuencia de muestreo en Hz (default: 2000)
  --simulate                Usar señal simulada en lugar de servidor
  --save-csv PATH           Guardar RMS por chunk en CSV
  --save-features PATH      Guardar features por ventana en CSV
  --mvc VALUE               Valor MVC para normalización (default: 500.0 µV)
  --window-ms MS            Tamaño de ventana para features (default: 200 ms)
  --hop-ms MS               Salto entre ventanas (default: 50 ms)
  --env-cutoff HZ           Cutoff del filtro de envolvente (default: 6.0 Hz)
  --fatigue-threshold VAL   Fracción MDF para detectar fatiga (default: 0.9)
  --baseline-frames N       Ventanas para estimar baseline MDF (default: 50)
  --log-freq HZ             Frecuencia de logs (default: 10 Hz, 0 = cada bloque)
  --show-mvc                Mostrar MVC en pantalla durante ejecución
  --verbose                 Logging detallado (DEBUG level)
"""

import argparse
import csv
import logging
import math
import os
import time
from collections import deque
from statistics import mean
from typing import List, Optional

import numpy as np
import requests
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.signal import welch


# --- UTILIDADES Y CONFIG DEFAULT ---
DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 9220
DEFAULT_FS = 2000
DEFAULT_BLOCK_SAMPLES = 50  # tamaño esperado aproximado por paquete
DEFAULT_RMS_MS = 100  # ventana RMS en ms
SAVE_BUFFER_SECONDS = 10  # cuánto guardar en memoria por canal


def estimate_mvc(samples: List[float], method: str = "peak") -> float:
    """
    Estimate MVC from accumulated samples using selected method.
    
    Methods:
        - peak: Maximum value (simple but can be affected by noise)
        - 95perc: 95th percentile (robust to outliers)
        - mean_top10: Mean of top 10% samples (robust to noise)
    
    Args:
        samples: List of envelope sample values
        method: Estimation method ("peak", "95perc", "mean_top10")
    
    Returns:
        MVC value in microvolts
    """
    if not samples:
        return 0.0
    arr = sorted(samples)
    if method == "peak":
        return float(max(arr))
    if method == "95perc":
        idx = min(len(arr) - 1, math.ceil(0.95 * len(arr)) - 1)
        return float(arr[idx])
    if method == "mean_top10":
        n = max(1, int(0.1 * len(arr)))
        return float(mean(arr[-n:]))
    # fallback
    return float(max(arr))


def create_filters(fs: int):
    # Diseño de banda 20-450 Hz (SENIAM)
    sos = butter(4, [20, 450], btype="bandpass", fs=fs, output="sos")
    return sos

# CONFIGURAR mvc OBTENIDO EN CALIBRATE YA QUE POR DEFECTO ES 500.0
class EMGStreamer:
    def __init__(self, ip: str, port: int, fs: int, rms_ms: int, simulate: bool, save_csv: Optional[str], mvc: float = 500.0, log_freq: float = 10.0, endpoint: str = "/features", http_timeout: float = 0.2, window_ms: int = 200, hop_ms: int = 50, envelope_cutoff: float = 6.0, save_features: Optional[str] = None, fatigue_threshold: float = 0.9, baseline_frames_needed: int = 50, mvc_method: str = "peak"):
        self.ip = ip
        self.port = port
        self.fs = fs
        self.simulate = simulate
        self.save_csv = save_csv

        # Normalización MVC (µV) y control de logs
        self.mvc = mvc
        self.log_freq = log_freq
        self._last_log_time = 0.0
        self._log_interval = (1.0 / log_freq) if log_freq and log_freq > 0 else 0.0

        # HTTP endpoint y timeout configurables
        self.endpoint = endpoint
        self.http_timeout = http_timeout

        self.sos = create_filters(fs)
        self.session = requests.Session()

        # Estado por canal (se crearán al recibir el primer chunk)
        self.zi_per_channel: List[np.ndarray] = []
        self.buffers: List[deque] = []

        # RMS window in samples
        self.rms_window = max(1, int((rms_ms / 1000.0) * fs))
        self.buffer_size = max(self.rms_window, int(SAVE_BUFFER_SECONDS * fs))

        # CSV file setup
        if self.save_csv:
            os.makedirs(os.path.dirname(self.save_csv) or ".", exist_ok=True)
            self.csv_file = open(self.save_csv, "w", newline="")
            self.csv_writer = None
        else:
            self.csv_file = None
            self.csv_writer = None

        # --- Real-time feature extraction parameters (configurable) ---
        self.window_ms = int(window_ms)
        self.hop_ms = int(hop_ms)
        self.window_samples = max(1, int((self.window_ms / 1000.0) * fs))
        self.hop_samples = max(1, int((self.hop_ms / 1000.0) * fs))

        # envelope low-pass filter config (Hz)
        self.envelope_cutoff = float(envelope_cutoff)
        self.env_sos = butter(4, self.envelope_cutoff, btype="low", fs=fs, output="sos")

        # per-channel envelope filter states (created on first chunk)
        self.zi_env_per_channel: List[np.ndarray] = []

        # feature saving (CSV)
        self.save_features = save_features
        self.features_writer = None
        self.features_file = None
        self.features_count = 0

        # fatigue detection baseline
        self.mdf_baseline = None
        self.baseline_frames_needed = int(baseline_frames_needed)
        self.fatigue_threshold = float(fatigue_threshold)
        
        # MVC estimation method
        self.mvc_method = mvc_method
        
        # Show MVC on screen
        self.show_mvc = False

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        if getattr(self, "features_file", None):
            try:
                self.features_file.close()
            except Exception:
                pass

    def _init_channels(self, n_channels: int):
        # Crear buffer y estado de filtro por canal
        self.zi_per_channel = [sosfilt_zi(self.sos).copy() for _ in range(n_channels)]
        self.buffers = [deque(maxlen=self.buffer_size) for _ in range(n_channels)]
        self.zi_env_per_channel = [sosfilt_zi(self.env_sos).copy() for _ in range(n_channels)]
        # samples since last feature extraction per channel
        self.samples_since_last_feature = [0 for _ in range(n_channels)]

    def _parse_json_samples(self, data) -> Optional[List[List[float]]]:
        # Intentos flexibles para extraer muestras por canal del JSON
        # Retorna samples as List[channel][samples]
        if isinstance(data, dict):
            # Estructura tipo Noraxon: 'channels': [{ 'index': 0, 'samples': [...] }, ...]
            if "channels" in data and isinstance(data["channels"], list):
                chans = []
                for ch in data["channels"]:
                    # Soporta 'data' o 'samples' o 'values'
                    for key in ("samples", "data", "values"):
                        if key in ch and isinstance(ch[key], list):
                            chans.append(list(ch[key]))
                            break
                if chans:
                    return chans

            # Estructura simple: 'samples' : [..] (single channel)
            for key in ("samples", "data", "values"):
                if key in data and isinstance(data[key], list):
                    return [list(data[key])]

        # No reconocido
        return None

    def fetch_chunk(self, timeout: Optional[float] = None) -> Optional[List[List[float]]]:
        if self.simulate:
            # Simulación: un canal con ruido blanco
            samples = np.random.randn(DEFAULT_BLOCK_SAMPLES)
            return [samples.tolist()]

        if timeout is None:
            timeout = self.http_timeout

        url = f"http://{self.ip}:{self.port}{self.endpoint}"
        print(  f"Fetching data from {url}...")
        try:
            r = self.session.get(url, timeout=timeout)
            # Log status if not OK
            if r.status_code != 200:
                # Debug suppressed: HTTP status returned from server
                # logging.debug("HTTP %d from %s: %s", r.status_code, url, r.text[:200])
                return None

            try:
                data = r.json()
            except ValueError as ve:
                # Debug suppressed: JSON decode error from server
                # logging.debug("JSON decode error from %s: %s", url, ve)
                return None

            parsed = self._parse_json_samples(data)
            if parsed is not None:
                return parsed
            # Fallback: si la respuesta es una lista plana
            if isinstance(data, list):
                return [list(data)]

        except requests.exceptions.RequestException as e:
            # Debug suppressed: low-level HTTP/request error
            # logging.debug("Fetch error: %s", e)
            return None

        return None

    def process_and_compute_rms(self, chunk: List[List[float]]) -> List[float]:
        # chunk: list of channels, each channel list of samples
        n_channels = len(chunk)
        if not self.zi_per_channel or len(self.zi_per_channel) != n_channels:
            self._init_channels(n_channels)

        rms_values = []
        for ch_idx, samples in enumerate(chunk):
            arr = np.asarray(samples, dtype=float)
            # Filtrado con estado por canal
            zi = self.zi_per_channel[ch_idx]
            filtered, zf = sosfilt(self.sos, arr, zi=zi)
            self.zi_per_channel[ch_idx] = zf

            # Envelope (rectify + low-pass) with per-channel state
            env_zi = self.zi_env_per_channel[ch_idx]
            rectified = np.abs(filtered)
            envelope, zf_env = sosfilt(self.env_sos, rectified, zi=env_zi)
            self.zi_env_per_channel[ch_idx] = zf_env

            # Añadir al buffer (filtered signal)
            buf = self.buffers[ch_idx]
            buf.extend(filtered.tolist())

            # update samples counter for feature extraction
            self.samples_since_last_feature[ch_idx] += len(filtered)

            # Calcular RMS sobre la ventana más reciente
            if len(buf) >= 1:
                tail = list(buf)[-self.rms_window:] if len(buf) >= self.rms_window else list(buf)
                tail_arr = np.asarray(tail, dtype=float)
                rms = math.sqrt(float(np.mean(tail_arr * tail_arr)))
            else:
                rms = 0.0

            rms_values.append(rms)

            # Sliding-window feature extraction (hop/window in samples)
            if self.samples_since_last_feature[ch_idx] >= self.hop_samples and len(buf) >= self.window_samples:
                window_arr = np.asarray(list(buf)[-self.window_samples:], dtype=float)
                tdom = self.compute_time_domain_features(window_arr, self.fs)
                fdom = self.compute_frequency_domain_features(window_arr, self.fs)

                # write features CSV if requested
                if self.save_features:
                    if self.features_writer is None:
                        os.makedirs(os.path.dirname(self.save_features) or ".", exist_ok=True)
                        self.features_file = open(self.save_features, "w", newline="")
                        self.features_writer = csv.writer(self.features_file)
                        # include ForcePercent estimated from RMS/MVC
                        header = ["timestamp", "channel"] + list(tdom.keys()) + ["ForcePercent"] + list(fdom.keys())
                        self.features_writer.writerow(header)
                    force_pct = (tdom.get("RMS", 0.0) / self.mvc) * 100.0
                    row = [time.time(), ch_idx] + list(tdom.values()) + [force_pct] + list(fdom.values())
                    self.features_writer.writerow(row)
                    self.features_file.flush()
                    self.features_count += 1

                # fatigue baseline estimation and detection
                mdf_val = fdom.get("MDF", None)
                if mdf_val is not None:
                    if self.mdf_baseline is None:
                        if not hasattr(self, "_mdf_acc"):
                            self._mdf_acc = []
                        self._mdf_acc.append(mdf_val)
                        if len(self._mdf_acc) >= self.baseline_frames_needed:
                            self.mdf_baseline = float(np.mean(self._mdf_acc))
                            logging.info("MDF baseline set: %.3f Hz", self.mdf_baseline)
                    else:
                        if mdf_val < (self.mdf_baseline * self.fatigue_threshold):
                            logging.warning("FATIGUE DETECTED on channel %d: MDF=%.2f < %.2f (baseline)", ch_idx, mdf_val, self.mdf_baseline * self.fatigue_threshold)

                # decrement counter by hop
                self.samples_since_last_feature[ch_idx] -= self.hop_samples

        return rms_values

    # --- Feature helper methods ---
    def compute_time_domain_features(self, signal: np.ndarray, fs: int):
        rms = float(np.sqrt(np.mean(signal * signal)))
        mav = float(np.mean(np.abs(signal)))
        iemg = float(np.sum(np.abs(signal)))
        wl = float(np.sum(np.abs(np.diff(signal))))
        zc = int(np.sum(((signal[:-1] * signal[1:]) < 0) & (np.abs(np.diff(signal)) > 0.01)))
        peak = float(np.max(np.abs(signal)))
        damv = float(np.mean(np.abs(np.diff(signal))))
        return {"RMS": rms, "MAV": mav, "IEMG": iemg, "WL": wl, "ZC": zc, "Peak": peak, "DAMV": damv}

    def compute_frequency_domain_features(self, signal: np.ndarray, fs: int, nperseg: int = 1024):
        nperseg = min(nperseg, len(signal))
        if nperseg < 4:
            return {"MNF": 0.0, "MDF": 0.0, "Peak_Freq": 0.0, "Total_Power": 0.0, "PowerBelow100Hz": 0.0}
        f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
        total_power = float(np.trapz(Pxx, f))
        if total_power <= 0:
            return {"MNF": 0.0, "MDF": 0.0, "Peak_Freq": 0.0, "Total_Power": 0.0, "PowerBelow100Hz": 0.0}
        mnf = float(np.sum(f * Pxx) / np.sum(Pxx))
        cumsum = np.cumsum(Pxx)
        half = cumsum[-1] / 2.0
        idx = np.searchsorted(cumsum, half)
        mdf = float(f[idx]) if idx < len(f) else float(f[-1])
        peak_idx = np.argmax(Pxx)
        peak_freq = float(f[peak_idx])
        power_below_100 = float(np.trapz(Pxx[f <= 100], f[f <= 100]))
        return {"MNF": mnf, "MDF": mdf, "Peak_Freq": peak_freq, "Total_Power": total_power, "PowerBelow100Hz": power_below_100}

    def run(self, target_fps: Optional[float] = 30.0):
        logging.info("Conectando a %s:%d (simulate=%s)", self.ip, self.port, self.simulate)
        backoff = 0.1
        try:
            while True:
                t0 = time.time()

                chunk = self.fetch_chunk(timeout=0.2)
                if chunk is None:
                    # No hay datos: aumentar backoff hasta cierto límite
                    # Debug suppressed: no chunk received from server
                    # logging.debug("No chunk received; sleeping backoff %.3f", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 1.0)
                    continue

                # Procesamos el bloque
                rms_list = self.process_and_compute_rms(chunk)

                # Normalización / estimación sencilla (usar MVC provisto)
                forces = [(r / self.mvc) * 100.0 for r in rms_list]

                timestamp = time.time()
                # Imprimir / logear controlando frecuencia
                now = time.time()
                do_log = True
                if self._log_interval > 0:
                    do_log = (now - self._last_log_time) >= self._log_interval

                if do_log:
                    for i, f in enumerate(forces):
                        msg = f"Chan {i}: RMS={rms_list[i]:.3f} µV | Fuerza={f:.2f}%"
                        if self.show_mvc:
                            msg += f" | MVC={self.mvc:.1f} µV"
                        logging.info(msg)
                    self._last_log_time = now

                # Guardar CSV opcional
                if self.save_csv:
                    if self.csv_writer is None:
                        header = ["timestamp"] + [f"chan_{i}_rms" for i in range(len(rms_list))] + [f"chan_{i}_force" for i in range(len(rms_list))]
                        self.csv_writer = csv.writer(self.csv_file)
                        self.csv_writer.writerow(header)
                    row = [timestamp] + rms_list + forces
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()

                # Control de bucle para target_fps
                elapsed = time.time() - t0
                if target_fps and elapsed < (1.0 / target_fps):
                    time.sleep((1.0 / target_fps) - elapsed)

                backoff = 0.1

        except KeyboardInterrupt:
            logging.info("Parando adquisición por KeyboardInterrupt")
        finally:
            self.close()


def build_arg_parser():
    p = argparse.ArgumentParser(description="EMG streaming client (Noraxon-like)")
    p.add_argument("--ip", default=DEFAULT_IP, help="IP del servidor")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Puerto HTTP")
    p.add_argument("--fs", type=int, default=DEFAULT_FS, help="Frecuencia de muestreo")
    p.add_argument("--rms-ms", type=int, default=DEFAULT_RMS_MS, help="Ventana RMS (ms)")
    p.add_argument("--simulate", action="store_true", help="Usar señal simulada en lugar de servidor")
    p.add_argument("--save-csv", default=None, help="Guardar RMS en CSV (ruta)")
    p.add_argument("--mvc", type=float, default=500.0, help="MVC para normalización (µV)")
    p.add_argument("--log-freq", type=float, default=10.0, help="Frecuencia de logs en Hz (0 = cada bloque)")
    p.add_argument("--endpoint", type=str, default="/features", help="Ruta HTTP del endpoint de streaming (por ejemplo /features)")
    p.add_argument("--http-timeout", type=float, default=0.2, help="Timeout HTTP en segundos para cada petición")
    p.add_argument("--verbose", action="store_true", help="Logging detallado")
    p.add_argument("--window-ms", type=int, default=200, help="Window length for features (ms)")
    p.add_argument("--hop-ms", type=int, default=50, help="Hop between feature windows (ms)")
    p.add_argument("--env-cutoff", type=float, default=6.0, help="Envelope lowpass cutoff (Hz)")
    p.add_argument("--save-features", default=None, help="Guardar features por ventana en CSV (ruta)")
    p.add_argument("--fatigue-threshold", type=float, default=0.9, help="Threshold fraction for MDF fatigue detection")
    p.add_argument("--baseline-frames", type=int, default=50, help="Frames to compute MDF baseline")
    p.add_argument("--mvc-method", type=str, default="peak", choices=["peak", "95perc", "mean_top10"], help="MVC estimation method: peak (máximo), 95perc (percentil 95), mean_top10 (media del 10% superior)")
    p.add_argument("--show-mvc", action="store_true", help="Mostrar MVC en pantalla durante ejecución")
    return p


def main():
    args = build_arg_parser().parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    # If not verbose, suppress noisy DEBUG logs from urllib3/requests
    if not args.verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    streamer = EMGStreamer(
        args.ip,
        args.port,
        args.fs,
        args.rms_ms,
        args.simulate,
        args.save_csv,
        mvc=args.mvc,
        log_freq=args.log_freq,
        endpoint=args.endpoint,
        http_timeout=args.http_timeout,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        envelope_cutoff=args.env_cutoff,
        save_features=args.save_features,
        fatigue_threshold=args.fatigue_threshold,
        baseline_frames_needed=args.baseline_frames,
        mvc_method=args.mvc_method,
    )
    streamer.show_mvc = args.show_mvc
    streamer.run()


if __name__ == "__main__":
    main()