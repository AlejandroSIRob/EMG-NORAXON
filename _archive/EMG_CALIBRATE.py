#!/usr/bin/env python3
"""
Simple calibrator to estimate MVC (Maximum Voluntary Contraction)
Alejandro Solar Iglesias
Uses `EMG.EMGStreamer` to stream and process EMG signals (simulate or real),
accumulates ENVELOPE samples (not RMS), and computes a per-channel MVC estimate.
This ensures consistency with EMG_CSV.py analysis.

Usage examples:
    python EMG_CALIBRATE.py --simulate --duration 5
    python EMG_CALIBRATE.py --ip localhost --port 9220 --endpoint /samples --duration 10 --method 95perc --save-json mvc.json

Methods:
    - peak: max envelope value observed (default)
    - 95perc: 95th percentile of envelope
    - mean_top10: mean of top 10% envelope samples

Behavior note:
    If communication with the streamer is lost (no data received for
    the configured `max_missed` interval), the script will terminate the
    recording and print the current per-channel MVC estimates it has
    collected so far. This makes it convenient to run a calibration and
    still obtain estimates if the device disconnects unexpectedly.

    NOTE: This script accumulates ENVELOPE samples (rectified + low-pass filtered)
    to match the offline analysis (EMG_CSV.py) for consistency.

Output: prints per-channel MVC (same units as envelope, µV) and optionally saves JSON.
"""

import argparse
import json
import math
import time
from statistics import mean
from typing import List, Dict

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from _archive.EMG_RT import EMGStreamer, DEFAULT_IP, DEFAULT_PORT, DEFAULT_FS, DEFAULT_RMS_MS


def estimate_mvc(samples: List[float], method: str = "peak") -> float:
    """Estimate MVC from accumulated samples using selected method."""
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


def run_calibration(ip: str, port: int, endpoint: str, simulate: bool, duration: float, method: str, mvc_out: str, fs: int, rms_ms: int, timeout: float, max_missed: float = 2.0):
    """
    Calibrate MVC by accumulating ENVELOPE samples (for consistency with EMG_CSV.py).
    
    Process:
    1. Fetch raw samples from streamer
    2. Apply bandpass filter (20-450 Hz) with state per channel
    3. Rectify and apply low-pass envelope filter (6 Hz) with state per channel
    4. Accumulate all envelope samples per channel
    5. Compute MVC statistic (peak, 95perc, mean_top10) from envelope samples
    """
    streamer = EMGStreamer(ip, port, fs, rms_ms, simulate, None, mvc=500.0, log_freq=0.0, endpoint=endpoint, http_timeout=timeout)

    # Design envelope filter (same as EMGStreamer: 6 Hz lowpass)
    env_sos = butter(4, 6.0, btype="low", fs=fs, output="sos")
    zi_env_per_channel: Dict[int, np.ndarray] = {}

    per_channel_samples: Dict[int, List[float]] = {}
    t_start = time.time()
    t_end = t_start + duration
    print(f"Starting calibration (simulate={simulate}) for {duration:.1f}s, method={method}")
    print("(Accumulating envelope samples for consistency with EMG_CSV.py)")
    print(f"{'Time':>8s} {'Elapsed':>10s} {'Samples':>12s} {'Rate':>10s}")
    print("-" * 50)
    last_data_time = time.time()
    last_print_time = t_start

    try:
        while time.time() < t_end:
            chunk = streamer.fetch_chunk(timeout=timeout)
            if chunk is None:
                # no data available right now
                # if we've missed data for longer than max_missed, assume communication lost
                if (time.time() - last_data_time) > max_missed:
                    print(f"\nNo se reciben datos desde hace {max_missed:.1f}s — finalizando calibración por pérdida de comunicación.")
                    break
                time.sleep(0.01)
                continue
            
            # Process each channel: bandpass -> rectify -> envelope
            for ch_idx, samples in enumerate(chunk):
                arr = np.asarray(samples, dtype=float)
                
                # Bandpass filter using streamer's filter state (20-450 Hz, SENIAM)
                if not streamer.zi_per_channel or ch_idx >= len(streamer.zi_per_channel):
                    streamer._init_channels(ch_idx + 1)
                
                zi = streamer.zi_per_channel[ch_idx]
                filtered, zf = sosfilt(streamer.sos, arr, zi=zi)
                streamer.zi_per_channel[ch_idx] = zf
                
                # Envelope: rectify + low-pass filter (6 Hz)
                if ch_idx not in zi_env_per_channel:
                    zi_env_per_channel[ch_idx] = sosfilt_zi(env_sos).copy()
                
                rectified = np.abs(filtered)
                envelope, zf_env = sosfilt(env_sos, rectified, zi=zi_env_per_channel[ch_idx])
                zi_env_per_channel[ch_idx] = zf_env
                
                # Accumulate envelope samples (each sample of the envelope)
                per_channel_samples.setdefault(ch_idx, []).extend(envelope.tolist())
            
            last_data_time = time.time()
            
            # Print progress every 1 second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                elapsed = current_time - t_start
                total_samples = len(per_channel_samples.get(0, []))
                rate = total_samples / elapsed if elapsed > 0 else 0
                print(f"{elapsed:>7.1f}s {total_samples:>12d} {rate:>10.0f} samples/s")
                last_print_time = current_time
            
            # small sleep to avoid busy loop
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Calibration interrupted by user")
    finally:
        streamer.close()

    # compute MVC per channel
    total_elapsed = time.time() - t_start
    results = {}
    results_detailed = {}
    
    for ch_idx, samples in per_channel_samples.items():
        mvc_val = estimate_mvc(samples, method=method)
        
        # Formato simplificado para resultados (int keys)
        results[ch_idx] = {
            "MVC": mvc_val,
            "n_samples": len(samples),
            "method": method,
        }
        
        # Formato detallado para JSON (string keys más descriptivos)
        channel_name = f"channel_{ch_idx}"
        results_detailed[channel_name] = {
            "channel_index": ch_idx,
            "mvc_uV": round(mvc_val, 2),
            "n_samples": len(samples),
            "method": method,
            "duration_s": round(total_elapsed, 2),
            "sampling_rate_hz": round(len(samples) / total_elapsed, 1)
        }

    # Print results
    print("-" * 50)
    print(f"Total duration: {total_elapsed:.2f}s")
    print(f"\nCalibration results:")
    for ch_idx in sorted(results.keys()):
        r = results[ch_idx]
        print(f"  Channel {ch_idx}: MVC={r['MVC']:.2f} µV  (n={r['n_samples']}, method={r['method']})")
        print(f"                Expected samples (~2000 Hz): {int(total_elapsed * 2000)}")
        print(f"                Actual samples collected: {r['n_samples']}")

    # Save JSON if requested
    if mvc_out:
        payload = {
            "timestamp": time.time(),
            "calibration_info": {
                "duration_s": round(total_elapsed, 2),
                "method": method,
                "sampling_rate_hz": fs,
                "num_channels": len(results_detailed)
            },
            "channels": results_detailed
        }
        with open(mvc_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n✓ MVC guardado en: {mvc_out}")
        print(f"  Canales guardados: {len(results_detailed)}")
        print(f"  Formato JSON mejorado para fácil lectura")

    return results


def build_parser():
    p = argparse.ArgumentParser(description="Calibrate MVC using EMGStreamer (simulate or real)")
    p.add_argument("--ip", default=DEFAULT_IP, help="Server IP (default: localhost)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    p.add_argument("--endpoint", type=str, default="/samples", help="Endpoint path")
    p.add_argument("--simulate", action="store_true", help="Use simulated data")
    p.add_argument("--duration", type=float, default=5.0, help="Seconds to record for calibration")
    p.add_argument("--method", type=str, default="peak", choices=["peak", "95perc", "mean_top10"], help="MVC estimation method")
    p.add_argument("--save-json", type=str, default=None, help="Save MVC results to JSON file")
    p.add_argument("--fs", type=int, default=DEFAULT_FS, help="Sampling frequency (Hz)")
    p.add_argument("--rms-ms", type=int, default=DEFAULT_RMS_MS, help="RMS window used by streamer (ms)")
    p.add_argument("--timeout", type=float, default=0.2, help="HTTP timeout per fetch (s)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_calibration(args.ip, args.port, args.endpoint, args.simulate, args.duration, args.method, args.save_json, args.fs, args.rms_ms, args.timeout)
