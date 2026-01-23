"""
Script de Análisis Completo: Tarea Desmontar
Autor: Alejandro Solar Iglesias
Fecha: Diciembre 2025
Versión: 2.0

Uso de la librería noraxon_analytics para:
- Carga automática de MVCs desde calibración
- Análisis de esfuerzo (%MVC) para TODOS los 12 músculos
- Detección de fatiga muscular (Frecuencia Mediana)
- Gráficos comparativos
"""

from noraxon_analytics import *
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ==============================================================================
# VALORES MVC (MÁXIMA CONTRACCIÓN VOLUNTARIA) - CALIBRACIÓN PREVIA
# Extraídos del archivo: 2025-12-17-11-24_Test_brazo_completo_MVC_results.json
# ==============================================================================
MVC_REFERENCE = {
    "FLEX.CARP.R": 392.48,           # Flexor Carpi Radialis
    "BRACHIORAD.": 251.36,           # Brachioradialis
    "FLEX.CARP.U": 287.45,           # Flexor Carpi Ulnaris
    "EXT.DIG.": 198.74,              # Extensor Digitorum
    "BICEPS BR.": 445.89,            # Biceps Brachii
    "LAT. TRICEPS": 512.34,          # Lateral Triceps
    "ANT.DELTOID": 367.92,           # Anterior Deltoid
    "POST.DELTOID": 289.45,          # Posterior Deltoid
    "MID DELT.": 401.23,             # Middle Deltoid
    "PECT. MAJOR": 478.56,           # Pectoralis Major
    "LAT.DORSI": 534.78,             # Latissimus Dorsi
    "ABDUCT.POL.": 156.89            # Abductor Pollicis
}

# Ruta del archivo CSV de tarea
TASK_FILE = r"C:\Users\alexs\Desktop\EMG_NORAVOX\Test_Medicion_Sin_Procesar\Solar_Iglesias,_Alejandro\2025-12-17-11-37_Test_brazo_completo_desmontar_01.csv"

FS = 2000.4  # Frecuencia de muestreo detectada

print("\n" + "="*80)
print(" ANÁLISIS BIOMECÁNICO COMPLETO - TAREA DESMONTAR")
print("="*80)

try:
    # =========================================================================
    # FASE 1: CARGA DE DATOS MULTICANAL
    # =========================================================================
    print("\n[1] Cargando datos del movimiento...")
    signals_dict, time_vec, fs_real, accel_dict, gyro_dict, mag_dict = load_noraxon_csv_multi(TASK_FILE)
    
    print(f"    ✓ Archivo cargado: {os.path.basename(TASK_FILE)}")
    print(f"    ✓ Frecuencia detectada: {fs_real} Hz")
    print(f"    ✓ Señales EMG: {len(signals_dict)} canales")
    print(f"    ✓ Duración: {time_vec[-1]:.2f} segundos")
    
    # =========================================================================
    # FASE 2: PROCESAMIENTO MULTICANAL (ESFUERZO)
    # =========================================================================
    print("\n[2] Procesando señales para cálculo de esfuerzo...")
    
    # Diccionarios para almacenar resultados
    results = {}
    envelopes = {}
    
    # Encabezado de tabla
    print(f"\n{'Músculo':<20} | {'RMS (uV)':<10} | {'IEMG (uV·s)':<12}")
    print("-" * 45)
    
    # Procesar cada canal EMG
    for muscle, raw_signal in signals_dict.items():
        # Pre-procesamiento SENIAM
        centered = remove_dc_offset(raw_signal)
        filtered = butter_bandpass_filter(centered, 20, 450, fs_real)
        envelope = compute_linear_envelope(filtered, fs_real, cutoff=6)
        
        # Obtener MVC de referencia
        mvc_val = MVC_REFERENCE.get(muscle, 300)  # Fallback si no existe
        
        # Normalización a %MVC usando función de librería
        norm_signal = normalize_to_mvc(envelope, mvc_val)
        
        # Extracción de características (función de librería)
        features = compute_time_domain_features_oleinikov(filtered, fs_real)
        
        # Almacenar resultados
        results[muscle] = {
            "mvc_ref": mvc_val,
            "norm_signal": norm_signal,
            "filtered_signal": filtered,
            "envelope": envelope,
            "features": features
        }
        envelopes[muscle] = envelope
        
        # Imprimir fila
        print(f"{muscle:<20} | {features['RMS']:>8.2f} uV  | {features['IEMG']:>8.1f} uV·s")
    
    print("-" * 45)
    
    # =========================================================================
    # FASE 3: ANÁLISIS DE FATIGA MUSCULAR (SELECTIVO)
    # =========================================================================
    print("\n[3] Analizando fatiga muscular (frecuencia mediana)...")
    print("    (Procesando los 3 músculos principales: Biceps, Triceps, Flexor Carpi)")
    
    main_muscles = ["BICEPS BR.", "LAT. TRICEPS", "FLEX.CARP.R"]
    fatigue_analysis = {}
    
    for muscle in main_muscles:
        if muscle in signals_dict:
            # Obtener la señal filtrada (SIN rectificar ni envolvente)
            raw_signal = signals_dict[muscle]
            centered = remove_dc_offset(raw_signal)
            filtered = butter_bandpass_filter(centered, 20, 450, fs_real)
            
            # Análisis de fatiga
            fatigue_data = analyze_muscle_fatigue(filtered, fs_real, window_sec=1.0, step_sec=0.5)
            fatigue_analysis[muscle] = fatigue_data
            
            slope = fatigue_data["Fatigue_Slope"]
            status = "FATIGADO" if fatigue_data["Is_Fatigued"] else "ESTABLE"
            
            print(f"\n    • {muscle}:")
            print(f"      Pendiente MDF: {slope:.4f} Hz/s [{status}]")
            print(f"      R² de tendencia: {fatigue_data['R_Squared']:.3f}")
    
    # =========================================================================
    # FASE 4: VISUALIZACIÓN CON FUNCIONES DE LA LIBRERÍA
    # =========================================================================
    print("\n[4] Visualizando señales EMG procesadas (función de librería)...")
    plot_emg_processed_signals(signals_dict, time_vec, fs_real, MVC_REFERENCE)
    
    print("\n[5] Visualizando datos de aceleración (función de librería)...")
    plot_accelerations(accel_dict, time_vec, fs_real)
    
    print("\n[6] Visualizando datos de giroscopio (función de librería)...")
    plot_gyroscopes(gyro_dict, time_vec, fs_real)
    
    print("\n[7] Visualizando datos de magnetómetro (función de librería)...")
    plot_magnetometers(mag_dict, time_vec, fs_real)
    
    # --- AUDITORÍA VISUAL DE FATIGA (3 PRINCIPALES) ---
    if fatigue_analysis:
        print("\n[8] Generando auditoría de ventana deslizante para músculos principales...")
        for muscle in main_muscles:
            if muscle in signals_dict:
                raw_signal = signals_dict[muscle]
                centered = remove_dc_offset(raw_signal)
                filtered = butter_bandpass_filter(centered, 20, 450, fs_real)
                envelope = compute_linear_envelope(filtered, fs_real, cutoff=6)
                visualize_mvc_sliding_window(envelope, time_vec, fs=fs_real, window_ms=500, title_muscle=muscle)
    
    # =========================================================================
    # FASE 5: RESUMEN EJECUTIVO
    # =========================================================================
    print("\n" + "="*80)
    print(" RESUMEN EJECUTIVO")
    print("="*80)
    
    # Búscar el músculo más exigido
    max_rms = max([results[m]['features']['RMS'] for m in results])
    most_demanding = [m for m in results if results[m]['features']['RMS'] == max_rms][0]
    
    # Búscar el más fatigado
    if fatigue_analysis:
        fatigued_muscles = [m for m in fatigue_analysis if fatigue_analysis[m]['Is_Fatigued']]
    else:
        fatigued_muscles = []
    
    print(f"\n Duración de tarea: {time_vec[-1]:.2f} segundos")
    print(f" Músculo más exigido: {most_demanding} ({max_rms:.2f} uV RMS)")
    print(f" Esfuerzo promedio global (RMS): {np.mean([results[m]['features']['RMS'] for m in results]):.2f} uV")
    print(f" Total de energía (suma IEMG): {sum([results[m]['features']['IEMG'] for m in results]):.1f} uV·s")
    
    if fatigued_muscles:
        print(f" Músculos con signos de fatiga: {', '.join(fatigued_muscles)}")
    else:
        print(f" Signos de fatiga: NO DETECTADOS")
    
    # Análisis de sobrecarga
    overloaded = [m for m in results if results[m]['features']['RMS'] > 100]
    if overloaded:
        print(f"\n  ADVERTENCIA: Músculos con RMS elevado (>100 uV):")
        for m in overloaded:
            print(f"    • {m}: {results[m]['features']['RMS']:.2f} uV")
    else:
        print(f"\n Activación muscular dentro de límites normales")
    
    print("\n" + "="*80)
    print(" Análisis completado exitosamente")
    print("="*80 + "\n")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
