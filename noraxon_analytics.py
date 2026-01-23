"""
Librería de Procesamiento EMG para Noraxon Ultium.
Autor: Alejandro Solar Iglesias
Fecha: Diciembre 2025
Versión: 1.0
"""


# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy.signal import butter, filtfilt, welch, correlate, sosfilt, sosfilt_zi, find_peaks
from statistics import mean
import math
import os
from pathlib import Path
from scipy.stats import linregress
import pywt

# --- CONFIGURACIÓN GLOBAL ---
# --- 1. CONFIGURACIÓN BASADA EN HARDWARE Y ESTÁNDARES ---

# Frecuencia de muestreo: Seleccionable en hardware Ultium (2000 o 4000 Hz).
# [REF: P-8808, Appendix A, pág. 42 "Selectable Sample Rate: 2000Hz or 4000Hz"]
DEFAULT_FS = 2000  

# Filtros Pasa-Banda: 
# Noraxon recomienda 10-500 Hz. Tu rango (20-450) es un subconjunto válido y más limpio.
# [REF: ABC_EMG, pág. 29 "amplifier bandpass (10-500 Hz)"]
# [REF: P-8808, Sec 7.3, pág. 27 "High Pass... 10Hz", "Low Pass... 500Hz"]
CUTOFF_LOW = 20    
CUTOFF_HIGH = 450  

# Filtro de Envolvente (Linear Envelope):
# Estándar para "Magnitud y Sincronización" en biomecánica (Paper Hombro).
# [REF: ABC_EMG, pág. 29 "... at 6 Hz (e.g. Butterworth, 2nd order or higher ..."]
# Tambien se cita en: 
# Fuentes del Toro S, Aranda-Ruiz J. (2025). "The Impact of Normalization 
#   Procedures on Surface Electromyography (sEMG) Data Integrity". 
#   Sensors, 25(9), 2668. https://doi.org/10.3390/s25092668
ENVELOPE_CUTOFF = 6  

# Recomienda filtro de orden 4:
# Zhao K, Jin Y, Feng Y, Li J, Zhou Y. (2024). "Standardizing EMG Pipelines 
#   for Muscle Synergy Analysis". Signals, 6(4), 68.
#   https://doi.org/10.3390/signals6040068
ENVELOPE_ORDER = 4   # Butterworth de cuarto orden

# Umbral de Fatiga:
# Caída de la frecuencia mediana.
# [REF: ABC_EMG, pág. 51 "Muscle Fatigue Index... left shift of the Total Power Spectrum"]
FATIGUE_THRESHOLD = 0.90

# --- FUNCIONES PRINCIPALES ---

# 1. CÁLCULO DE CALIDAD DE SEÑAL (SNR)
def calculate_signal_quality_snr(signal: np.ndarray, noise_floor_percentile: float = 5) -> dict:
    """
    Evalúa la calidad de la señal estimando el SNR y el nivel de ruido basal.
    
    Criterios estrictos según ABC of EMG:
    -------------------------------------
    1. Límite Máximo: "Small amplitude spikes... should not exceed 10 - 15 microvolts."
       [REF: ABC of EMG, pág. 23]
       
    2. Excelencia: "The average noise level... should be located at 1 (=excellent) to 3.5 microvolts."
       [REF: ABC of EMG, pág. 23]
      
    Args:
        signal: Señal EMG procesada (uV).
        noise_floor_percentile: Percentil para estimar el ruido basal (Default: 5%).
        
    Returns:
        dict: {SNR_dB, Noise_uV, Quality_Status}
    """
    # 1. Potencia de la Señal Activa (Estimación del 95%)
    signal_amplitude = np.percentile(np.abs(signal), 95)
    signal_power = signal_amplitude ** 2
    
    # 2. Estimación del Ruido (Average Noise Level)
    # El texto pide "mean amplitude... for 5 seconds".
    # Usamos el percentil 5 como la aproximación estadística más robusta 
    # para encontrar ese nivel basal en una señal que contiene movimiento.
    noise_amplitude = np.percentile(np.abs(signal), noise_floor_percentile)
    noise_estimate = noise_amplitude ** 2
    
    # 3. Cálculo SNR (dB)
    if noise_estimate > 0:
        snr_linear = signal_power / noise_estimate
        snr_db = 10 * np.log10(snr_linear)
    else:
        snr_db = float('inf')
        
    # 4. Evaluación Clínica ESTRICTA (Según pág. 23 ABC of EMG)
    
    if noise_amplitude <= 3.5:
        # Rango ideal descrito: 1 a 3.5 uV
        status = f"EXCELENTE ({noise_amplitude:.2f} uV <= 3.5 uV)"
        
    elif noise_amplitude <= 15.0:
        # Zona de tolerancia: Picos visibles pero aceptables hasta 10-15 uV
        status = f"ACEPTABLE / CON RUIDO ({noise_amplitude:.2f} uV <= 15 uV)"
        
    else:
        # Fuera de norma (> 15 uV)
        status = f"MALA CALIDAD - REVISAR ({noise_amplitude:.2f} uV > 15 uV)"
        
    return {
        "SNR_dB": float(snr_db),
        "Noise_Floor_uV": float(noise_amplitude),
        "Signal_Amp_uV": float(signal_amplitude),
        "Status": status
    }

# 2. REMOCIÓN DE DC OFFSET
def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """
    Centra la señal en cero restando la media (DC Offset Removal).
    
    Justificación / References:
    1. "Most amplifiers work with an auto offset correction. However, it is possible 
       that the EMG baseline is shifted away from the true zero line." 
       [REF: ABC of EMG, p. 23]
       
    2. "If not identified and corrected, all amplitude based calculations are 
       invalid for that record." 
       [REF: ABC of EMG, p. 23]

    Args:
        data: Señal EMG cruda (numpy array).
        
    Returns:
        Señal centrada (data - mean).
    """
    return data - np.mean(data)

# 3. FILTRO PASA-BANDA (BUTTERWORTH)
def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """
    Aplica filtro Butterworth Pasa-Banda de fase cero.
    
    Incluye validación del Teorema de Nyquist para evitar Aliasing.
    [REF: ABC of EMG, pág. 29 "Digital Filtering"]
    [REF: ABC of EMG, pág. 14 "Sampling Theorem of Nyquist: aliasing effects"]
    """
    # --- VALIDACIÓN DE NYQUIST ---
    # "Sampling a signal at a frequency which is too low results in aliasing effects"
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    
    if highcut >= nyq:
        raise ValueError(
            f"Error de Diseño: La frecuencia de corte alta ({highcut} Hz) debe ser "
            f"menor que la frecuencia de Nyquist ({nyq} Hz). "
            f"[REF: ABC of EMG, pág. 14]"
        )
    if lowcut <= 0:
        raise ValueError("Error de Diseño: La frecuencia de corte baja debe ser > 0 Hz.")
    if lowcut >= highcut:
        raise ValueError("Error de Diseño: La frecuencia baja debe ser menor que la alta.")
    
    # --- DISEÑO Y APLICACIÓN ---
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    return filtfilt(b, a, data)

# 4. CÁLCULO DE ENVOLVENTE LINEAL
def compute_linear_envelope(signal: np.ndarray, fs: int, cutoff: float = ENVELOPE_CUTOFF, order: int = ENVELOPE_ORDER) -> np.ndarray:
    """
    Calcula la Envolvente Lineal (Linear Envelope).
    1. Rectificación de Onda Completa.
    2. Filtro Pasa-Bajos (6 Hz).
    
    [REF: Paper 'Magnitud y sincronización...' - Estándar biomecánico para amplitud]
    [REF: ABC of EMG, pág. 27 "Rectification", pág. 29 "Smoothing - Low Pass 6Hz"]
    """
    # --- VALIDACIÓN DE NYQUIST ---
    # Aunque 6 Hz es muy bajo, validamos para robustez si cambia la Fs.
    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(
            f"Error de Envolvente: El cutoff ({cutoff} Hz) viola el Teorema de Nyquist ({nyq} Hz). "
            f"[REF: ABC of EMG, pág. 14]"
        )

    # 1. Rectificación (Valor Absoluto)
    rectified_signal = np.abs(signal)
    
    # 2. Filtro Pasa-Bajos
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    
    # Usamos filtfilt para evitar retardo de fase (Phase Shift)
    envelope = filtfilt(b, a, rectified_signal)
    
    return envelope

# 5. CÁLCULO DE VALOR MVC ROBUSTO
def estimate_mvc(samples: list, method: str = "peak") -> float:
    """
    Función auxiliar interna para calcular el estadístico MVC.
    
    Referencias Científicas:
    ------------------------
    1. Zhao K, et al. (2024). "Standardizing EMG Pipelines...". Signals 6(4).
       Recomienda métodos robustos (95%) para reproducibilidad.
    2. Fuentes del Toro S, et al. (2025). "The Impact of Normalization...". Sensors 25(9).
       Valida el MVC como gold standard.
    3. Konrad P. (2005). "The ABC of EMG". Noraxon Inc.
       Recomienda promedios de ventanas estables.
    """
    if not samples:
        return 0.0
        
    arr = sorted(samples)
    n = len(arr)
    
    if method == "peak":
        # "Among the most widely employed methods... scales the signal to the peak amplitude." 
        #
        return float(max(arr))
        
    if method == "95perc":
        # "...normalization methods... interactively influenced the outcomes."
        # Se prefiere percentil 95 para evitar outliers de ruido.
        #
        idx = min(n - 1, math.ceil(0.95 * n) - 1)
        return float(arr[idx])
        
    if method == "mean_top10":
        # "A more stable reference value is the mean amplitude of the highest signal portion."
        #
        n_top = max(1, int(0.1 * n))
        return float(np.mean(arr[-n_top:]))
        
    return float(max(arr))

def calculate_mvc_value(signal: np.ndarray, fs: int = DEFAULT_FS, window_ms: int = 500, method: str = "95perc") -> float:
    """
    Calcula el valor MVC robusto usando una ventana móvil promedio (no un pico único).
    
    Metodología:
    " A more stable reference value is the mean amplitude of the highest signal portion with e.g. 500 ms duration"
    [REF: ABC of EMG, pág. 31]
    
    Args:
        signal: Señal EMG procesada (Filtrada + Rectificada o Envolvente).
        fs: Frecuencia de muestreo.
        window_ms: Duración de la ventana de suavizado (Estándar: 500 ms).
        method: Método para estimar el MVC ("peak", "95perc", "mean_top10").
        
    Returns:
        float: El valor máximo de referencia (uV).
    """
    # 1. Convertir ventana de ms a muestras
    window_size = int(fs * (window_ms / 1000))
    
    if len(signal) < window_size:
        return estimate_mvc(signal.tolist(), method=method)

    # 2. Usar pandas para calcular media móvil eficientemente
    # Esto crea una señal suavizada donde cada punto es el promedio de los 500ms anteriores
    series = pd.Series(np.abs(signal))
    rolling_mean = series.rolling(window=window_size, center=True).mean()
    
    # 3. Eliminar valores NaN generados por los bordes de la ventana
    valid_values = rolling_mean.dropna().tolist()
    
    # 4. Calcular el MVC usando el método seleccionado (recomendado: 95perc)
    mvc_value = estimate_mvc(valid_values, method=method)
    
    return float(mvc_value)

# 6. NORMALIZACIÓN A %MVC
def normalize_to_mvc(envelope: np.ndarray, mvc_value: float) -> np.ndarray:
    """
    Normaliza la señal a porcentaje de la Contracción Voluntaria Máxima (%MVC).
    [REF: ABC of EMG, pág. 35 "Amplitude Normalization"]
    """
    if mvc_value <= 0:
        return envelope 
    
    return (envelope / mvc_value) * 100

# 7. CÁLCULO DE CARACTERÍSTICAS EN DOMINIO DEL TIEMPO (OLEINIKOV et al. 2018)
def compute_time_domain_features_oleinikov(signal: np.ndarray, fs: int = DEFAULT_FS, prev_mav: float = 0.0) -> dict:
    """
    Calcula características en el dominio del tiempo siguiendo estrictamente
    el orden y definiciones de Oleinikov et al. (2018).
    
    Paper: "Feature extraction and real-time recognition..." (IEEE, 2018).
    
    Argumentos:
        signal (np.ndarray): Ventana de señal EMG procesada (filtrada).
        fs (int): Frecuencia de muestreo (Default: 2000 Hz).
        prev_mav (float): MAV de la ventana anterior (necesario para Eq. 3). 
                          Si es la primera ventana, usar 0.0.
    
    Retorna:
        dict: Diccionario con las características calculadas.
    """
    features = {}
    N = len(signal)
    
    # Definición del umbral de ruido (Noise Threshold)
    # El paper sugiere 0.004 mV (4 uV). Usamos un umbral dinámico para robustez.
    # Noraxon advierte: "Noise may not be interpreted as increased tonus"[cite: 3699].
    threshold = max(0.004, 0.05 * np.max(np.abs(signal)))

    # ---------------------------------------------------------
    # 1. Mean Absolute Value (MAV) - Ecuación (2)
    # ---------------------------------------------------------
    # Fórmula: (1/N) * sum(|x_k|)
    # Noraxon Reference: "Mean value ... best describes the gross innervation input"[cite: 3424].
    mav = np.mean(np.abs(signal))
    features['MAV'] = float(mav)

    # ---------------------------------------------------------
    # 2. Difference Mean Absolute Value (DMAV) - Ecuación (3)
    # ---------------------------------------------------------
    # Fórmula: MAV_i - MAV_{i-1}
    # Nota: Requiere el valor de la ventana anterior.
    features['DMAV'] = float(mav - prev_mav)

    # ---------------------------------------------------------
    # 3. Zero Crossing (ZC) - Ecuación (4)
    # ---------------------------------------------------------
    # Criterio: Cambio de signo AND |x_k - x_{k+1}| >= Umbral
    # Noraxon Reference: "Zero Crossing rate is highly correlated to the FFT ... 
    # becomes less important today".
    # (Aunque Noraxon lo considera antiguo, es vital para la red neuronal de Oleinikov).
    zc_count = 0
    # Vectorizamos la detección de cambio de signo para eficiencia
    sign_changes = np.diff(np.signbit(signal))
    # Verificamos el umbral de amplitud en esos puntos
    for i in np.where(sign_changes)[0]:
        if np.abs(signal[i] - signal[i+1]) >= threshold:
            zc_count += 1
    features['ZC'] = float(zc_count)

    # ---------------------------------------------------------
    # 4. Slope Sign Change (SSC) - Ecuación (5)
    # ---------------------------------------------------------
    # Criterio: Cambio de pendiente (pico o valle) dentro del umbral.
    # Detecta cuando (x_k > x_{k-1} y x_k > x_{k+1}) o viceversa.
    ssc_count = 0
    for k in range(1, N - 1):
        diff_prev = signal[k] - signal[k-1]
        diff_next = signal[k] - signal[k+1]
        # Si el producto es positivo, hubo cambio de dirección (pico local)
        if (diff_prev * diff_next) > 0:
            # Aplicar umbral para descartar ruido de fondo
            if np.abs(diff_prev) >= threshold or np.abs(diff_next) >= threshold:
                ssc_count += 1
    features['SSC'] = float(ssc_count)

    # ---------------------------------------------------------
    # 5. Waveform Length (WL) - Ecuación (6)
    # ---------------------------------------------------------
    # Fórmula: Sum(|x_k - x_{k-1}|)
    # Representa la complejidad de la señal (frecuencia + amplitud).
    features['WL'] = float(np.sum(np.abs(np.diff(signal))))

    # ---------------------------------------------------------
    # 6. Root Mean Square (RMS) - Ecuación (7)
    # ---------------------------------------------------------
    # Fórmula: sqrt( (1/N) * Sum(x_k^2) )
    # Noraxon Reference: "Preferred recommendation for smoothing ... 
    # reflects the mean power of the signal".
    features['RMS'] = float(np.sqrt(np.mean(signal**2)))

    # ---------------------------------------------------------
    # 7. Variance (VAR) - Ecuación (8)
    # ---------------------------------------------------------
    # Fórmula: (1 / (N-1)) * Sum(x_k^2)
    # Usamos ddof=1 para usar (N-1) como indica el paper estándar.
    features['VAR'] = float(np.var(signal, ddof=1))

    # ---------------------------------------------------------
    # EXTRA: Integrated EMG (IEMG) - Métrica Noraxon
    # ---------------------------------------------------------
    # [REF: ABC_EMG, pág. 40 "IEMG means integrated EMG (=Area under the curve) and in earlier days this term was often disused for analog smoothed EMG curves..."]
    features['IEMG'] = float(np.sum(np.abs(signal)))

    return features

# 8. CÁLCULO DE CARACTERÍSTICAS EN DOMINIO DE LA FRECUENCIA (OLEINIKOV et al. 2018)
def compute_frequency_domain_features_oleinikov(signal: np.ndarray, fs: int = DEFAULT_FS) -> dict:
    """
    Calcula características en el dominio de la frecuencia y tiempo-frecuencia.
    
    Metodología basada en: 
    A. Oleinikov et al., "Feature extraction and real-time recognition...", IEEE, 2018.
    
    Incluye:
    1. FFT (Ecuación X_k): Frecuencia Media (MNF) y Pico (PKF).
    2. Wavelet (Ecuación 10): Daubechies (db4) Nivel 4 para detección de MUAP.
    3. Autocorrelación (Ecuación 11): Ritmo de disparo de unidades motoras.

    Args:
        signal (np.ndarray): Ventana de señal EMG procesada.
        fs (int): Frecuencia de muestreo.

    Returns:
        dict: Diccionario con las características calculadas.
    """
    features = {}
    N = len(signal)

    # ---------------------------------------------------------
    # 1. Transformada Rápida de Fourier (FFT) - Ecuación X_k
    # ---------------------------------------------------------
    # Fórmula: X_k = sum(x_n * e^(-i2pi*k*n/N))
    # [REF: Oleinikov et al. (2018), Sección "Frequency Domain Features"]
    
    fft_values = scipy.fft.fft(signal)
    fft_freqs = scipy.fft.fftfreq(N, 1/fs)
    
    # Filtrar solo la mitad positiva del espectro (Teorema de Nyquist)
    positive_idxs = fft_freqs > 0
    fft_vals_real = np.abs(fft_values[positive_idxs])
    freqs_real = fft_freqs[positive_idxs]
    
    # Espectro de Potencia (P)
    power_spectrum = fft_vals_real ** 2
    
    # A) Frecuencia Media (MNF - Mean Frequency)
    # Indicador estándar de fatiga muscular (Spectral Compression).
    # [REF: ABC of EMG, pág. 41 "Mean Frequency... decline indicates fatigue"]
    if np.sum(power_spectrum) > 0:
        mnf = np.sum(freqs_real * power_spectrum) / np.sum(power_spectrum)
    else:
        mnf = 0.0
    features['FREQ_MNF'] = float(mnf)

    # B) Frecuencia de Amplitud Máxima (Peak Frequency) y su Amplitud
    # [REF: Oleinikov et al. (2018), "maximum amplitude frequency and amplitude value"]
    if len(power_spectrum) > 0:
        peak_idx = np.argmax(power_spectrum)
        peak_freq = freqs_real[peak_idx]
        peak_amp = fft_vals_real[peak_idx]
    else:
        peak_freq = 0.0
        peak_amp = 0.0
        
    features['FREQ_Peak_Hz'] = float(peak_freq)
    features['FREQ_Peak_Amp'] = float(peak_amp)

    # ---------------------------------------------------------
    # 2. Transformada Wavelet (DWT) - Ecuación (10)
    # ---------------------------------------------------------
    # Fórmula: psi(a,b) ... usando Daubechies (db4) nivel 4.
    # Contexto: "La función wavelet db4 en nivel 4 es la más adecuada para la detección 
    # del Potencial de Acción de la Unidad Motora (MUAP)".
    # [REF: Oleinikov et al. (2018), Eq. 10 y referencia [3]]

    try:
        import pywt
        WAVELET_AVAILABLE = True
    except ImportError:
        WAVELET_AVAILABLE = False

    if WAVELET_AVAILABLE:
        try:
            # Descomposición multinivel usando db4 hasta nivel 4
            # wavedec devuelve: [cA4, cD4, cD3, cD2, cD1]
            coeffs = pywt.wavedec(signal, 'db4', level=4)
            
            # El Coeficiente de Detalle 4 (cD4) contiene la info en la banda de frecuencia
            # típica de los disparos de unidades motoras.
            cD4 = coeffs[1] 
            
            # Característica: Energía de la Wavelet en Nivel 4
            features['WAVE_db4_L4_Energy'] = float(np.sum(cD4 ** 2))
            
            # Característica: Desviación Estándar (Variabilidad de la forma de onda)
            features['WAVE_db4_L4_Std'] = float(np.std(cD4))
            
        except Exception as e:
            print(f"Error Wavelet: {e}")
            features['WAVE_db4_L4_Energy'] = 0.0
            features['WAVE_db4_L4_Std'] = 0.0
    else:
        features['WAVE_db4_L4_Energy'] = 0.0
        features['WAVE_db4_L4_Std'] = 0.0

    # ---------------------------------------------------------
    # 3. Autocorrelación (Ryy) - Ecuación (11)
    # ---------------------------------------------------------
    # Fórmula: R_yy(l) = Sum(y_n * y_{n-l})
    # [REF: Oleinikov et al. (2018), Eq. 11]
    
    # Usamos 'full' y tomamos la mitad derecha (lags positivos)
    corr_full = np.correlate(signal, signal, mode='full')
    corr_positive = corr_full[len(corr_full)//2:]
    
    # Normalizamos (Lag 0 = 1.0) para obtener el coeficiente R
    if corr_positive[0] > 0:
        corr_norm = corr_positive / corr_positive[0]
    else:
        corr_norm = corr_positive
        
    # Buscamos el primer pico significativo después del lag 0
    # Esto indica la periodicidad de la señal (frecuencia de disparo)
    peaks, _ = scipy.signal.find_peaks(corr_norm, height=0.1, distance=fs*0.01) # min 10ms
    
    if len(peaks) > 0:
        first_peak_idx = peaks[0]
        # Convertimos muestras a tiempo (ms)
        time_lag_ms = (first_peak_idx / fs) * 1000
        features['AUTOCORR_FirstPeak_ms'] = float(time_lag_ms)
        features['AUTOCORR_MaxVal'] = float(corr_norm[first_peak_idx])
    else:
        features['AUTOCORR_FirstPeak_ms'] = 0.0
        features['AUTOCORR_MaxVal'] = 0.0

    return features

# 9. EXTRACCIÓN DE CARACTERÍSTICAS EN VENTANAS DESLIZANTES (OLEINIKOV)
def extract_features_continuous(signal: np.ndarray, fs: int, window_ms: int = 200, step_ms: int = 50):
    """
    Aplica el protocolo de Oleinikov usando Ventanas Deslizantes (Sliding Windows).
    Esto permite ver la evolución de la señal y hace funcionar correctamente el cálculo de DMAV.
    
    Args:
        signal: Señal EMG completa (filtrada).
        fs: Frecuencia de muestreo (2000 Hz).
        window_ms: Tamaño de la ventana en ms (Oleinikov usa 200ms).
        step_ms: Cada cuánto avanza la ventana (solapamiento). 
                 50ms de paso significa 150ms de solapamiento (Overlap).
    
    Returns:
        pd.DataFrame: Tabla con la evolución temporal de todas las características.
    """
    # Convertir ms a número de muestras
    window_size = int(fs * (window_ms / 1000))
    step_size = int(fs * (step_ms / 1000))
    
    features_list = []
    prev_mav = 0.0  # Inicializamos MAV previo para la primera ventana
    
    # Bucle de ventana deslizante
    num_windows = (len(signal) - window_size) // step_size
    
    print(f" Generando {num_windows} ventanas de análisis...")
    
    for i in range(0, len(signal) - window_size, step_size):
        # 1. Extraer el segmento actual
        segment = signal[i : i + window_size]
        
        # 2. Calcular características de Tiempo (Oleinikov)
        time_feats = compute_time_domain_features_oleinikov(
            segment, 
            fs, 
            prev_mav=prev_mav  # Pasamos el MAV de la ventana anterior
        )
        
        # 3. Calcular características de Frecuencia (Oleinikov)
        freq_feats = compute_frequency_domain_features_oleinikov(segment, fs)
        
        # 4. Guardar estado para la siguiente iteración (Clave para DMAV)
        prev_mav = time_feats['MAV']
        
        # 5. Añadir estampa de tiempo (centro de la ventana)
        time_point = (i + window_size/2) / fs
        
        # Unir todo en un diccionario
        row = {'Time_s': time_point, **time_feats} #, **freq_feats}
        features_list.append(row)
        
    return pd.DataFrame(features_list)

# 10. ANÁLISIS DE FATIGA MUSCULAR (FRECUENCIA MEDIANA)
def analyze_muscle_fatigue(signal: np.ndarray, fs: int, window_sec: float = 1.0, step_sec: float = 0.5) -> dict:
    """
    Analiza la fatiga muscular mediante la caída de la Frecuencia Mediana (MDF).
    
    Metodología: "Spectral Left Shift" (Desplazamiento espectral a la izquierda).
    [REF: ABC of EMG, pág. 51 "Muscle Fatigue Analysis"]
    [REF: Delsys / Oleinikov "Frequency domain features"]

    Args:
        signal (np.ndarray): Señal EMG filtrada (20-450 Hz). ¡NO USAR LA ENVOLVENTE!
        fs (int): Frecuencia de muestreo (2000 Hz).
        window_sec (float): Tamaño de la ventana de análisis en segundos (Estándar: 1s).
        step_sec (float): Paso de avance (solapamiento).

    Returns:
        dict: Pendiente de fatiga, valores MDF por tiempo y estado.
    """
    # Parámetros de ventana
    nperseg = int(window_sec * fs)
    nstep = int(step_sec * fs)
    
    mdf_trajectory = []
    time_points = []
    
    # Bucle de ventanas deslizantes para análisis de tendencia
    # Si la señal es más corta que la ventana, se analiza como un solo bloque
    if len(signal) < nperseg:
        # Fallback para señales muy cortas
        freqs, power = scipy.signal.welch(signal, fs, nperseg=len(signal))
        area_total = np.sum(power)
        cumulative_area = np.cumsum(power)
        idx_mdf = np.where(cumulative_area >= area_total / 2)[0][0]
        return {
            "MDF_Values": np.array([freqs[idx_mdf]]),
            "Time_Points": np.array([0]),
            "Fatigue_Slope": 0.0,
            "Is_Fatigued": False,
            "Initial_MDF": freqs[idx_mdf],
            "Final_MDF": freqs[idx_mdf]
        }

    for i in range(0, len(signal) - nperseg, nstep):
        segment = signal[i : i + nperseg]
        
        # 1. Calcular Densidad Espectral de Potencia (Welch)
        freqs, power = scipy.signal.welch(segment, fs, nperseg=nperseg)
        
        # 2. Calcular Frecuencia Mediana (MDF)
        # MDF es el punto donde el área bajo la curva se divide en dos mitades iguales
        area_total = np.sum(power)
        cumulative_area = np.cumsum(power)
        
        # Encontrar el índice donde se supera el 50% de la energía
        if len(cumulative_area) > 0 and area_total > 0:
            idx_mdf = np.where(cumulative_area >= area_total / 2)[0][0]
            mdf = freqs[idx_mdf]
        else:
            mdf = 0
        
        mdf_trajectory.append(mdf)
        time_points.append(i / fs)

    # 3. Análisis de Tendencia (Regresión Lineal)
    if len(mdf_trajectory) > 1:
        slope, intercept, r_value, _, _ = linregress(time_points, mdf_trajectory)
    else:
        slope = 0.0
    
    # Determinación de Fatiga (Criterio: Pendiente negativa significativa)
    # Umbral empírico: Si la pendiente es menor a -0.05 Hz/s, consideramos fatiga probable.
    is_fatigued = slope < -0.05 
    
    return {
        "MDF_Values": np.array(mdf_trajectory),
        "Time_Points": np.array(time_points),
        "Fatigue_Slope": slope,  # Hz por segundo (negativo = fatiga)
        "R_Squared": r_value**2 if len(mdf_trajectory) > 1 else 0,
        "Is_Fatigued": is_fatigued,
        "Initial_MDF": mdf_trajectory[0] if len(mdf_trajectory) > 0 else 0,
        "Final_MDF": mdf_trajectory[-1] if len(mdf_trajectory) > 0 else 0
    }




################################################################################################################################
# --- FUNCIONES AUXILIARES ---

# --- FUNCIÓN DE CARGA ROBUSTA---
def load_noraxon_csv_multi(file_path: str):
    """
    Carga CSV de Noraxon y extrae:
    - Señales EMG (uV)
    - Aceleraciones (mG)
    - Giroscopios (deg/s)
    - Magnetómetros (mGauss)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo no existe: {file_path}")

    # 1. Extraer fs real y saltar metadatos usando encoding con 'sig' para eliminar el BOM
    fs = 2000.0
    skip_rows = 2  # Por defecto, headers en fila 3
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        # Frecuencia en fila 2, columna 3 (índice 1)
        if len(lines) > 1:
            header_parts = lines[1].replace('"', '').split(';')
            if len(header_parts) > 2:
                try:
                    fs = float(header_parts[2].strip().replace(',', '.'))
                except (ValueError, IndexError):
                    fs = 2000.0  # Fallback a valor por defecto

        # Localizar fila de nombres de columna (row 3 por defecto en Noraxon)
        for i, line in enumerate(lines):
            if 'time' in line.lower() and '(uV)' in line:
                skip_rows = i
                break

    # 2. Cargar DataFrame
    df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=skip_rows, on_bad_lines='skip')
    
    # Limpiar nombres de columnas: minúsculas y sin comillas
    df.columns = [str(c).replace('"', '').strip() for c in df.columns]

    # 3. Extraer Tiempo (Fallback: si falla el nombre, usar primera columna)
    time_col = [c for c in df.columns if 'time' in c.lower()]
    if time_col:
        time_vec = df[time_col[0]].values
    else:
        time_vec = df.iloc[:, 0].values
        print("[ADVERTENCIA] 'time' detectado por posicion, no por nombre.")

    # 4. Organizar señales EMG
    emg_cols = [col for col in df.columns if '(uV)' in col]
    
    if not emg_cols:
        raise ValueError(f"No se encontraron columnas EMG con (uV) en el archivo")
    
    print(f"[INFO] Detectadas {len(emg_cols)} columnas EMG")
    
    # Limpieza de nombres para el diccionario final
    signals_dict = {}
    for col in emg_cols:
        # Extraer nombre limpio (quitar " (uV)" y "RT" si existe)
        muscle_name = col.split(' (uV)')[0].replace(' RT', '').strip()
        signals_dict[muscle_name] = df[col].values

    # 5. Organizar Aceleraciones (mG)
    accel_cols = [col for col in df.columns if 'Accel' in col and 'mG' in col]
    accel_dict = {}
    for col in accel_cols:
        accel_dict[col] = df[col].values
    
    # 6. Organizar Giroscopios (deg/s)
    gyro_cols = [col for col in df.columns if 'Gyro' in col and 'deg/s' in col]
    gyro_dict = {}
    for col in gyro_cols:
        gyro_dict[col] = df[col].values
    
    # 7. Organizar Magnetómetros (mGauss)
    mag_cols = [col for col in df.columns if 'Mag' in col and 'mGauss' in col]
    mag_dict = {}
    for col in mag_cols:
        mag_dict[col] = df[col].values

    return signals_dict, time_vec, fs, accel_dict, gyro_dict, mag_dict


def load_csv(file_path: str, fs: int = DEFAULT_FS):
    """
    Carga un archivo CSV de Noraxon automáticamente.
    
    Busca la columna que contenga 'uV' (microvoltios).
    Asume formato: sep=';', decimal=',', header=2
    
    Returns:
        signal: Array numpy con valores de EMG
        time: Array numpy con tiempos en segundos
        column_name: Nombre de la columna EMG encontrada
    """
    # Leer CSV con parámetros específicos de Noraxon
    df = pd.read_csv(file_path, sep=';', decimal=',', header=2)
    
    # Encontrar columna EMG (contiene "uV")
    emg_cols = [col for col in df.columns if '(uV)' in col]
    if not emg_cols:
        raise ValueError(f"No se encontró columna EMG (uV) en {file_path}")
    
    emg_col = emg_cols[0]
    
    # Extraer columnas
    signal = df[emg_col].values
    time = df['time'].values
    
    return signal, time, emg_col

def load_all_emg_data(file_path: str):
    """
    Carga múltiples canales EMG de un archivo Noraxon.
    
    Args:
        file_path (str): Ruta al archivo CSV.
        
    Returns:
        signals_dict (dict): Diccionario {Nombre_Sensor: Array_Datos}.
        time (np.array): Vector de tiempo.
        emg_cols (list): Lista con los nombres de todos los sensores detectados.
    """
    # Intentar leer con header=2 (formato estándar)
    try:
        df = pd.read_csv(file_path, sep=';', decimal=',', header=2)
        if not any('(uV)' in str(c) for c in df.columns):
            raise ValueError
    except:
        # Fallback header=3
        df = pd.read_csv(file_path, sep=';', decimal=',', header=3)
    
    # Limpieza de nombres
    df.columns = [str(c).replace('"', '').strip() for c in df.columns]
    
    # Identificar columnas
    emg_cols = [col for col in df.columns if '(uV)' in col]
    
    if not emg_cols:
        raise ValueError(f"No se encontraron columnas '(uV)' en {file_path}")
    
    # Extraer datos
    time = df['time'].values
    signals_dict = {col: df[col].values for col in emg_cols}
    
    # Retornar los 3 valores que pediste
    return signals_dict, time, emg_cols

def plot_signals(time, signals, labels=None, title="EMG Signals", ylabel="Amplitud (uV)", figsize=(12, 6), subplots=False):
    """
    Función genérica para graficar señales EMG.
    Soporta gráficos superpuestos (subplots=False) o separados (subplots=True).
    
    Args:
        time (np.array): Vector de tiempo (eje X).
        signals (list or np.array): Señal única o lista de señales [s1, s2, ...].
        labels (list of str, optional): Nombres para la leyenda.
        title (str): Título general.
        ylabel (str): Etiqueta del eje Y.
        figsize (tuple): Tamaño base de la figura.
        subplots (bool): Si es True, separa cada señal en un gráfico distinto.
    """
    # 1. Estandarización de entradas
    if isinstance(signals, np.ndarray):
        signals = [signals]
    
    # Generar etiquetas por defecto si no existen
    if labels is None:
        labels = [f"Señal {i+1}" for i in range(len(signals))]
    elif isinstance(labels, str):
        labels = [labels]
        
    # Paleta de colores
    colors = plt.cm.tab10.colors

    # ---------------------------------------------------------
    # MODO SUBPLOTS (Gráficos separados verticalmente)
    # ---------------------------------------------------------
    if subplots and len(signals) > 1:
        n_signals = len(signals)
        # Ajuste dinámico de altura: 2.5 pulgadas por subplot mínimo
        dynamic_height = max(figsize[1], 2.5 * n_signals)
        dynamic_figsize = (figsize[0], dynamic_height)
        
        # sharex=True hace que al hacer zoom en uno, se muevan todos
        fig, axes = plt.subplots(nrows=n_signals, ncols=1, sharex=True, figsize=dynamic_figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99) # Título arriba del todo
        
        for i, ax in enumerate(axes):
            color = colors[i % len(colors)]
            # Plot
            ax.plot(time, signals[i], color=color, linewidth=1.2, label=labels[i])
            
            # Estilos por subplot
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=9, frameon=True)
            
            # Solo poner etiqueta X en el último gráfico de abajo
            if i == n_signals - 1:
                ax.set_xlabel("Tiempo (s)", fontsize=12)
        
        plt.tight_layout()
        # Ajuste fino para que el título no se solape con el primer gráfico
        plt.subplots_adjust(top=0.95) 
        plt.show()

    # ---------------------------------------------------------
    # MODO SUPERPUESTO (Todo en uno) - Comportamiento original
    # ---------------------------------------------------------
    else:
        plt.figure(figsize=figsize)
        
        for i, signal in enumerate(signals):
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f"Señal {i+1}"
            plt.plot(time, signal, color=color, label=label, alpha=0.8, linewidth=1.2)
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Tiempo (s)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Mostrar leyenda si hay etiquetas o múltiples señales
        if len(signals) > 1 or labels:
            plt.legend(fontsize=10, loc='upper right', frameon=True)
            
        plt.tight_layout()
        plt.show()

def visualize_mvc_sliding_window(signal: np.ndarray, time: np.ndarray, fs: int = DEFAULT_FS, window_ms: int = 500, title_muscle: str = ""):
    """
    Grafica el proceso de la ventana deslizante para auditar el cálculo del MVC.
    
    Genera 2 Subplots:
    1. Panorama General: Señal completa + Rastro de la ventana móvil (Rojo).
    2. Zoom de Precisión: Acercamiento a la ventana seleccionada (Verde).
    
    Args:
        signal (np.ndarray): Señal envolvente (ya procesada/rectificada).
        time (np.ndarray): Vector de tiempo.
        fs (int): Frecuencia de muestreo.
        window_ms (int): Tamaño de la ventana en ms (Def: 500).
        title_muscle (str): Nombre del músculo para el título (opcional).
    """
    # 1. Cálculos internos necesarios para graficar
    window_samples = int(fs * (window_ms / 1000))
    
    # Serie de Pandas para calcular la Media Móvil (el "rastro" rojo)
    series = pd.Series(signal)
    rolling_trace = series.rolling(window=window_samples, center=True).mean()
    
    # Encontrar la ventana ganadora (MVC)
    idx_max = np.nanargmax(rolling_trace)
    mvc_peak = rolling_trace[idx_max]
    
    # Definir tiempos de inicio y fin de la zona verde
    t_inicio = time[max(0, idx_max - window_samples // 2)]
    t_fin    = time[min(len(time)-1, idx_max + window_samples // 2)]
    
    # ---------------------------------------------------------
    # CONFIGURACIÓN DE GRÁFICOS (SUBPLOTS)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    
    # Título dinámico
    if title_muscle:
        title_str = f"Auditoria de Ventana Deslizante - {title_muscle} (MVC: {mvc_peak:.2f} uV)"
    else:
        title_str = f"Auditoria de Ventana Deslizante (MVC: {mvc_peak:.2f} uV)"
    
    fig.suptitle(title_str, fontsize=16, fontweight='bold')

    # --- SUBPLOT 1: PANORAMA GENERAL (Todos los pasos) ---
    ax1.set_title("1. Rastro de la Ventana Móvil (Línea Roja = Promedio en cada instante)", fontsize=12)
    # Señal de fondo
    ax1.plot(time, signal, color='lightgray', label='Señal Envolvente (Cruda)', alpha=0.6)
    # El "camino" de la ventana (Todos los pasos)
    ax1.plot(time, rolling_trace, color='red', linewidth=2, label=f'Media Móvil ({window_ms}ms)')
    # La zona ganadora
    ax1.axvspan(t_inicio, t_fin, color='green', alpha=0.3, label='Ventana Seleccionada')
    
    ax1.set_ylabel("Amplitud (uV)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- SUBPLOT 2: ZOOM A LA VENTANA VERDE (Detalle) ---
    ax2.set_title(f"2. Detalle de la Ventana Ganadora ({t_inicio:.2f}s - {t_fin:.2f}s)", fontsize=12)
    
    # Recortamos los datos para el zoom (añadimos 1 segundo de margen a cada lado)
    margin_s = 1.0 
    idx_zoom_start = max(0, int(idx_max - (window_samples/2) - (margin_s * fs)))
    idx_zoom_end   = min(len(time), int(idx_max + (window_samples/2) + (margin_s * fs)))
    
    # Graficar solo el segmento de zoom
    ax2.plot(time[idx_zoom_start:idx_zoom_end], signal[idx_zoom_start:idx_zoom_end], 
             color='gray', alpha=0.5, label='Señal')
    ax2.plot(time[idx_zoom_start:idx_zoom_end], rolling_trace[idx_zoom_start:idx_zoom_end], 
             color='red', linewidth=2, label='Promedio Móvil')
    
    # Resaltar la caja verde
    ax2.axvspan(t_inicio, t_fin, color='green', alpha=0.2)
    # Línea del nivel MVC
    ax2.axhline(mvc_peak, color='green', linestyle='--', linewidth=2, label=f'Nivel MVC: {mvc_peak:.2f} uV')
    
    # Anotación de texto
    ax2.text(time[idx_max], mvc_peak + (mvc_peak*0.05), f"{mvc_peak:.1f} uV", 
             color='green', fontweight='bold', ha='center')

    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Amplitud (uV)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.show()


# --- FUNCIONES DE VISUALIZACIÓN ---
def plot_emg_processed_signals(signals_dict: dict, time_vec: np.ndarray, fs: float, mvc_results: dict):
    """Grafica cada señal EMG procesada en una figura separada"""
    if len(signals_dict) == 0:
        print("[ERROR] No hay senales para graficar")
        return
    
    # Convertir tiempo a segundos
    time_sec = np.arange(len(time_vec)) / fs
    
    for muscle, signal in signals_dict.items():
        # Procesar señal
        centered = remove_dc_offset(signal)
        filtered = butter_bandpass_filter(centered, 20, 450, fs)
        envelope = compute_linear_envelope(filtered, fs, cutoff=6)
        
        # Obtener MVC
        mvc_val = mvc_results.get(muscle, 0)
        
        # Graficar
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time_sec, filtered, 'b-', linewidth=0.6, alpha=0.6, label='Senal Filtrada (20-450 Hz)')
        ax.plot(time_sec, envelope, 'r-', linewidth=2, alpha=0.9, label='Envolvente (6 Hz)')
        ax.axhline(mvc_val, color='g', linestyle='--', linewidth=2.5, label=f'MVC: {mvc_val:.2f} uV')
        ax.set_title(f'Senal EMG Procesada - {muscle}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Amplitud (uV)', fontsize=12)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()


def plot_accelerations(accel_dict: dict, time_vec: np.ndarray, fs: float):
    """Grafica cada aceleracion en una figura separada"""
    if not accel_dict:
        print("[ADVERTENCIA] No hay datos de aceleracion")
        return
    
    # Agrupar por sensor (1, 2, 3, ... 12)
    sensors = {}
    for col_name, data in accel_dict.items():
        # Extraer numero de sensor
        parts = col_name.split()
        sensor_num = None
        for i, part in enumerate(parts):
            if part == 'Accel' and i+1 < len(parts):
                try:
                    sensor_num = int(parts[i+1])
                    axis = parts[i+2] if i+2 < len(parts) else 'X'
                    break
                except:
                    pass
        
        if sensor_num and sensor_num not in sensors:
            sensors[sensor_num] = {'Ax': None, 'Ay': None, 'Az': None}
        if sensor_num:
            sensors[sensor_num][axis] = data
    
    time_sec = np.arange(len(time_vec)) / fs
    
    for sensor_num in sorted(sensors.keys()):
        axes_data = sensors[sensor_num]
        fig, ax = plt.subplots(figsize=(14, 5))
        
        if axes_data['Ax'] is not None:
            ax.plot(time_sec, axes_data['Ax'], 'r-', linewidth=1, label='Ax', alpha=0.8)
        if axes_data['Ay'] is not None:
            ax.plot(time_sec, axes_data['Ay'], 'g-', linewidth=1, label='Ay', alpha=0.8)
        if axes_data['Az'] is not None:
            ax.plot(time_sec, axes_data['Az'], 'b-', linewidth=1, label='Az', alpha=0.8)
        
        ax.set_title(f'Aceleracion - Sensor {sensor_num}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Aceleracion (mG)', fontsize=12)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()


def plot_gyroscopes(gyro_dict: dict, time_vec: np.ndarray, fs: float):
    """Grafica cada giroscopo en una figura separada"""
    if not gyro_dict:
        print("[ADVERTENCIA] No hay datos de giroscopo")
        return
    
    # Agrupar por sensor (1, 2, 3, ... 12)
    sensors = {}
    for col_name, data in gyro_dict.items():
        # Extraer numero de sensor
        parts = col_name.split()
        sensor_num = None
        for i, part in enumerate(parts):
            if part == 'Gyro' and i+1 < len(parts):
                try:
                    sensor_num = int(parts[i+1])
                    axis = parts[i+2] if i+2 < len(parts) else 'X'
                    break
                except:
                    pass
        
        if sensor_num and sensor_num not in sensors:
            sensors[sensor_num] = {'Gx': None, 'Gy': None, 'Gz': None}
        if sensor_num:
            sensors[sensor_num][axis] = data
    
    time_sec = np.arange(len(time_vec)) / fs
    
    for sensor_num in sorted(sensors.keys()):
        axes_data = sensors[sensor_num]
        fig, ax = plt.subplots(figsize=(14, 5))
        
        if axes_data['Gx'] is not None:
            ax.plot(time_sec, axes_data['Gx'], 'r-', linewidth=1, label='Gx', alpha=0.8)
        if axes_data['Gy'] is not None:
            ax.plot(time_sec, axes_data['Gy'], 'g-', linewidth=1, label='Gy', alpha=0.8)
        if axes_data['Gz'] is not None:
            ax.plot(time_sec, axes_data['Gz'], 'b-', linewidth=1, label='Gz', alpha=0.8)
        
        ax.set_title(f'Giroscopo - Sensor {sensor_num}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Velocidad Angular (deg/s)', fontsize=12)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()


def plot_magnetometers(mag_dict: dict, time_vec: np.ndarray, fs: float):
    """Grafica cada magnetometro en una figura separada"""
    if not mag_dict:
        print("[ADVERTENCIA] No hay datos de magnetometro")
        return
    
    # Agrupar por sensor (1, 2, 3, ... 12)
    sensors = {}
    for col_name, data in mag_dict.items():
        # Extraer numero de sensor
        parts = col_name.split()
        sensor_num = None
        for i, part in enumerate(parts):
            if part == 'Mag' and i+1 < len(parts):
                try:
                    sensor_num = int(parts[i+1])
                    axis = parts[i+2] if i+2 < len(parts) else 'X'
                    break
                except:
                    pass
        
        if sensor_num and sensor_num not in sensors:
            sensors[sensor_num] = {'Mx': None, 'My': None, 'Mz': None}
        if sensor_num:
            sensors[sensor_num][axis] = data
    
    time_sec = np.arange(len(time_vec)) / fs
    
    for sensor_num in sorted(sensors.keys()):
        axes_data = sensors[sensor_num]
        fig, ax = plt.subplots(figsize=(14, 5))
        
        if axes_data['Mx'] is not None:
            ax.plot(time_sec, axes_data['Mx'], 'r-', linewidth=1, label='Mx', alpha=0.8)
        if axes_data['My'] is not None:
            ax.plot(time_sec, axes_data['My'], 'g-', linewidth=1, label='My', alpha=0.8)
        if axes_data['Mz'] is not None:
            ax.plot(time_sec, axes_data['Mz'], 'b-', linewidth=1, label='Mz', alpha=0.8)
        
        ax.set_title(f'Magnetometro - Sensor {sensor_num}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Campo Magnetico (mGauss)', fontsize=12)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
