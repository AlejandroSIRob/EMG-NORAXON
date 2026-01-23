"""
Script de Calibración MVC Multicanal para Noraxon Ultium.
Autor: Alejandro Solar Iglesias
Fecha: Diciembre 2025
Versión: 1.5 
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from noraxon_analytics import *


if __name__ == "__main__":
    # Ruta absoluta corregida según tu estructura actual
    filename = r'C:\Users\alexs\Desktop\EMG_NORAVOX\Test_Medicion_Sin_Procesar\Solar_Iglesias,_Alejandro\2025-12-17-11-24_Test_brazo_completo_MVC.csv'

    print(f"[INICIO] Iniciando protocolo para: {os.path.basename(filename)}")

    try:
        # 1. Carga de datos con fs real (ej: 2000.4 Hz)
        signals, time_vec, fs_real, accel_data, gyro_data, mag_data = load_noraxon_csv_multi(filename)
        mvc_results = {}

        print(f" Archivo cargado. Frecuencia detectada: {fs_real} Hz")
        print(f"\n{'-'*70}")
        print(f"{'MUSCULO':<25} | {'ESTADO SNR':<18} | {'MVC (uV)':<10}")
        print(f"{'-'*70}")

        # 2. Pipeline de procesamiento por canal
        for muscle, raw_data in signals.items():
            # Pre-procesamiento SENIAM
            centered = remove_dc_offset(raw_data)
            qa = calculate_signal_quality_snr(centered)
            
            # Filtro Pasa-Banda 20-450 Hz
            filtered = butter_bandpass_filter(centered, 20, 450, fs_real)
            
            # Envolvente Lineal (Smoothing 6 Hz)
            envelope = compute_linear_envelope(filtered, fs_real, cutoff=6)
            
            # Cálculo MVC: Ventana móvil 100ms + Percentil 95 (Zhao et al. 2024)
            mvc_val = calculate_mvc_value(envelope, fs=fs_real, window_ms=100, method="95perc")
            
            mvc_results[muscle] = round(float(mvc_val), 2)
            
            # Log de resultados clínico
            status = qa['Status'].split(' (')[0]
            print(f"{muscle:<25} | {status:<18} | {mvc_val:>8.2f} uV")

        # 3. Guardar resultados para normalización de futuros registros
        output_json = filename.replace('.csv', '_results.json')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(mvc_results, f, indent=4, ensure_ascii=False)
        
        print(f"{'-'*70}")
        print(f"[GUARDADO] Resultados guardados en: {os.path.basename(output_json)}")

        # 4. Auditoría visual (para TODOS los canales)
        print(f"\n[AUDITORIA] Generando auditorias de ventana deslizante para todos los canales...")
        for muscle, raw_data in signals.items():
            centered = remove_dc_offset(raw_data)
            filtered = butter_bandpass_filter(centered, 20, 450, fs_real)
            envelope = compute_linear_envelope(filtered, fs_real, cutoff=6)
            visualize_mvc_sliding_window(envelope, time_vec, fs=fs_real, window_ms=500, title_muscle=muscle)
        
        # 5. Generar graficos uno por uno
        print(f"\n[GRAFICOS] Visualizando senales EMG procesadas...")
        plot_emg_processed_signals(signals, time_vec, fs_real, mvc_results)
        
        print(f"\n[GRAFICOS] Visualizando aceleraciones...")
        plot_accelerations(accel_data, time_vec, fs_real)
        
        print(f"\n[GRAFICOS] Visualizando giroscopios...")
        plot_gyroscopes(gyro_data, time_vec, fs_real)
        
        print(f"\n[GRAFICOS] Visualizando magnetometros...")
        plot_magnetometers(mag_data, time_vec, fs_real)
        
        print(f"\n[COMPLETO] Todas las visualizaciones generadas exitosamente")
    except Exception as e:
        print(f"\n[ERROR] Fallo critico: {e}")
        import traceback
        traceback.print_exc()