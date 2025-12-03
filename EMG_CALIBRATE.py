#!/usr/bin/env python3
"""
Simple calibrator to estimate MVC (Maximum Voluntary Contraction)
Alejandro Solar Iglesias
Uses `EMG.EMGStreamer` to stream RMS values (simulate or real) and
computes a per-channel MVC estimate.

Usage examples:
    python EMG_CALIBRATE.py --simulate --duration 5
    python EMG_CALIBRATE.py --ip localhost --port 9220 --endpoint /samples --duration 10 --method 95perc --save-json mvc.json

Methods:
    - peak: max RMS observed (default)
    - 95perc: 95th percentile RMS
    - mean_top10: mean of top 10% RMS samples

Behavior note:
    If communication with the streamer is lost (no data received for
    the configured `max_missed` interval), the script will terminate the
    recording and print the current per-channel MVC estimates it has
    collected so far. This makes it convenient to run a calibration and
    still obtain estimates if the device disconnects unexpectedly.

Output: prints per-channel MVC (same units as RMS, µV) and optionally saves JSON.
"""

import argparse
import json
import time
from statistics import mean
from typing import List, Dict

from EMG import EMGStreamer, DEFAULT_IP, DEFAULT_PORT, DEFAULT_FS, DEFAULT_RMS_MS


def estimate_mvc(samples: List[float], method: str = "peak") -> float:
    if not samples:
        return 0.0
    arr = sorted(samples)
    if method == "peak":
        return float(max(arr))
    if method == "95perc":
        import math
        idx = min(len(arr) - 1, math.ceil(0.95 * len(arr)) - 1)
        return float(arr[idx])
    if method == "mean_top10":
        n = max(1, int(0.1 * len(arr)))
        return float(mean(arr[-n:]))
    # fallback
    return float(max(arr))


def run_calibration(ip: str, port: int, endpoint: str, simulate: bool, duration: float, method: str, mvc_out: str, fs: int, rms_ms: int, timeout: float, max_missed: float = 2.0):
    streamer = EMGStreamer(ip, port, fs, rms_ms, simulate, None, mvc=500.0, log_freq=0.0, endpoint=endpoint, http_timeout=timeout)

    per_channel_samples: Dict[int, List[float]] = {}
    t_end = time.time() + duration
    print(f"Starting calibration (simulate={simulate}) for {duration:.1f}s, method={method}")
    last_data_time = time.time()

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
            rms_list = streamer.process_and_compute_rms(chunk)
            # add per-channel
            for ch_idx, rms in enumerate(rms_list):
                per_channel_samples.setdefault(ch_idx, []).append(rms)
            last_data_time = time.time()
            # small sleep to avoid busy loop
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Calibration interrupted by user")
    finally:
        streamer.close()

    # compute MVC per channel
    results = {}
    for ch_idx, samples in per_channel_samples.items():
        mvc_val = estimate_mvc(samples, method=method)
        results[ch_idx] = {
            "MVC": mvc_val,
            "n_samples": len(samples),
            "method": method,
        }

    # Print results
    print("\nCalibration results:")
    for ch_idx in sorted(results.keys()):
        r = results[ch_idx]
        print(f" Channel {ch_idx}: MVC={r['MVC']:.3f} µV  (n={r['n_samples']}, method={r['method']})")

    # Save JSON if requested
    if mvc_out:
        payload = {"timestamp": time.time(), "results": results}
        with open(mvc_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved MVC to {mvc_out}")

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
