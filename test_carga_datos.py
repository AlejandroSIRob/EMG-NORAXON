"""
Script de prueba rapida para verificar carga de datos
sin generar graficos (para testing en terminal)
"""

import os
import pandas as pd
import numpy as np
from noraxon_analytics import *

# Ruta del archivo
filename = r'C:\Users\alexs\Desktop\EMG_NORAVOX\Test_Medicion_Sin_Procesar\Solar_Iglesias,_Alejandro\2025-12-17-11-24_Test_brazo_completo_MVC.csv'

print("[TEST] Iniciando prueba de carga de datos...")

try:
    # Verificar que existe el archivo
    if not os.path.exists(filename):
        print(f"[ERROR] Archivo no encontrado: {filename}")
    else:
        print(f"[OK] Archivo encontrado: {os.path.basename(filename)}")
    
    # Carga basica de CSV
    fs = 2000.0
    skip_rows = 2
    
    with open(filename, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        # Frecuencia
        if len(lines) > 1:
            header_parts = lines[1].replace('"', '').split(';')
            if len(header_parts) > 2:
                fs = float(header_parts[2].strip().replace(',', '.'))
        print(f"[OK] Frecuencia de muestreo: {fs} Hz")
        
        # Encontrar headers
        for i, line in enumerate(lines):
            if 'time' in line.lower() and '(uV)' in line:
                skip_rows = i
                break
        print(f"[OK] Headers encontrados en fila {skip_rows}")
    
    # Cargar DataFrame
    df = pd.read_csv(filename, sep=';', decimal=',', skiprows=skip_rows, on_bad_lines='skip')
    print(f"[OK] DataFrame cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Detectar columnas
    df.columns = [str(c).replace('"', '').strip() for c in df.columns]
    
    emg_cols = [col for col in df.columns if '(uV)' in col]
    accel_cols = [col for col in df.columns if 'Accel' in col and 'mG' in col]
    gyro_cols = [col for col in df.columns if 'Gyro' in col and 'deg/s' in col]
    mag_cols = [col for col in df.columns if 'Mag' in col and 'mGauss' in col]
    
    print(f"[OK] Detectadas {len(emg_cols)} columnas EMG")
    print(f"[OK] Detectadas {len(accel_cols)} columnas de Aceleracion")
    print(f"[OK] Detectadas {len(gyro_cols)} columnas de Giroscopo")
    print(f"[OK] Detectadas {len(mag_cols)} columnas de Magnetometro")
    
    # Listar sensores EMG
    print("\n[EMG SENSORES]:")
    for col in emg_cols:
        muscle_name = col.split(' (uV)')[0].replace(' RT', '').strip()
        signal = df[col].values
        print(f"  - {muscle_name:<20} | Min: {signal.min():>8.2f} | Max: {signal.max():>8.2f} | Mean: {signal.mean():>8.2f}")
    
    # Verificar tiempo
    time_col = [c for c in df.columns if 'time' in c.lower()]
    if time_col:
        time_vec = df[time_col[0]].values
        print(f"\n[TIEMPO]:")
        print(f"  - Duracion: {time_vec[-1] - time_vec[0]:.2f} segundos")
        print(f"  - Muestras: {len(time_vec)}")
    
    print("\n[RESULTADO] PRUEBA EXITOSA - Script listo para ejecutar calibration_multi_sensor.py")

except Exception as e:
    print(f"\n[ERROR] Fallo en prueba: {e}")
    import traceback
    traceback.print_exc()
