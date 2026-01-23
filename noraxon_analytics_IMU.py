"""
Código para el análisis de datos de IMU (Unidad de Medición Inercial)
Autor: Alejandro Solar Iglesias
Fecha: Diciembre 2025
Versión: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. FUNCIÓN DE CARGA ROBUSTA
# =============================================================================
def load_imu_data(filepath):
    print(f"Leyendo archivo: {filepath}...")
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    header_idx = -1
    for i, line in enumerate(lines):
        if "time" in line and "Activity" in line:
            header_idx = i
            break
            
    if header_idx == -1:
        raise ValueError("Error: No se encontró la cabecera de datos.")

    raw_header = lines[header_idx].strip().split(';')
    header = [h.replace('"', '').strip() for h in raw_header]
    
    df = pd.read_csv(filepath, sep=';', decimal=',', names=header, skiprows=header_idx+1, engine='python')
    df = df.dropna(subset=['time'])
    return df

# =============================================================================
# 2. MOTOR DE FÍSICA (SISTEMA INTERNACIONAL)
# =============================================================================
def calcular_metricas_fisicas(df):
    acc_cols = [c for c in df.columns if 'Accel' in c]
    gyro_cols = [c for c in df.columns if 'Gyro' in c]
    mag_cols = [c for c in df.columns if 'Mag' in c]
    
    time = df['time'].values
    dt = np.mean(np.diff(time))

    # --- A. FUERZA (ACELERACIÓN) EN m/s² ---
    # 1. Convertir de mG a G
    acc_data_g = df[acc_cols].values / 1000.0 
    # 2. Convertir de G a m/s² (1 G = 9.81 m/s²)
    acc_data_ms2 = acc_data_g * 9.81

    # Cálculo de la Norma (Magnitud Total)
    # [REF: Saraf A, et al. (2023) - Vector Magnitude Feature]
    # https://www.sciencedirect.com/science/article/pii/S0957417422008144
    fuerza_ms2 = np.linalg.norm(acc_data_ms2, axis=1)

    # --- B. TORQUE (ACELERACIÓN ANGULAR) EN rad/s² ---
    gyro_deg = df[gyro_cols].values
    gyro_rad = np.deg2rad(gyro_deg)
    
    # Derivada (dω/dt)
    # [REF: Koopman B, et al. (2017) - Numerical Differentiation]
    # segmental angular acceleration is obtained by numerically differentiating the angular velocity value.
    # https://ieeexplore.ieee.org/document/7222528
    # Usamos np.gradient que usa diferencias centrales (más preciso que np.diff)
    # Aceleración = d(Velocidad) / dt
    alpha_vec = np.gradient(gyro_rad, axis=0) / dt
    torque_proxy = np.linalg.norm(alpha_vec, axis=1)

    # --- C. MAGNETÓMETRO ---
    mag_data = df[mag_cols].values
    mag_norm = np.linalg.norm(mag_data, axis=1)

    return time, fuerza_ms2, torque_proxy, mag_norm


# =============================================================================
# 3. MÓDULO DE CINEMÁTICA PARA OPENSIM (NUEVO)
# =============================================================================
try:
    from ahrs.filters import Madgwick
    AHRS_AVAILABLE = True
except ImportError:
    AHRS_AVAILABLE = False
    print(" ADVERTENCIA: Librería 'ahrs' no instalada. Ejecuta 'pip install ahrs'")

def calcular_orientacion_para_opensim(df, fs=2000):
    """
    Toma el DataFrame crudo de Noraxon y calcula los Cuaterniones (q0,q1,q2,q3)
    usando fusión de sensores (Madgwick) para cada sensor detectado.
    
    Retorna: DataFrame listo para exportar como .STO
    """
    if not AHRS_AVAILABLE: return None
    
    print(f" Calculando Orientación (Quaterniones) a {fs} Hz...")
    
    # Configuración del algoritmo de fusión
    madgwick = Madgwick(frequency=fs, beta=0.1)
    
    # Preparar DataFrame de salida
    sto_data = pd.DataFrame()
    sto_data['time'] = df['time']
    
    # Identificar sensores automáticamente (Accel 1, Accel 2, etc.)
    sensores = set()
    for col in df.columns:
        if "Accel" in col:
            try:
                # Extraer número: "Ultium... Accel 1 Ax..." -> "1"
                partes = col.split('.')
                parte_sensor = [p for p in partes if 'Accel' in p][0]
                num = parte_sensor.split('Accel')[1].split()[0]
                sensores.add(num)
            except: pass
            
    # Procesar cada sensor
    for sensor_id in sorted(sensores):
        try:
            # 1. Extraer Aceleración (mG -> m/s²)
            # Buscamos columnas dinámicamente
            ax = df[[c for c in df.columns if f"Accel {sensor_id} Ax" in c][0]].values
            ay = df[[c for c in df.columns if f"Accel {sensor_id} Ay" in c][0]].values
            az = df[[c for c in df.columns if f"Accel {sensor_id} Az" in c][0]].values
            acc = np.column_stack((ax, ay, az)) * (9.81 / 1000.0)
            
            # 2. Extraer Giroscopio (deg/s -> rad/s)
            gx = df[[c for c in df.columns if f"Gyro {sensor_id} Gx" in c][0]].values
            gy = df[[c for c in df.columns if f"Gyro {sensor_id} Gy" in c][0]].values
            gz = df[[c for c in df.columns if f"Gyro {sensor_id} Gz" in c][0]].values
            gyr = np.column_stack((gx, gy, gz)) * (np.pi / 180.0)
            
            # 3. Calcular Cuaterniones (Q)
            # Q = [w, x, y, z]
            Q = np.zeros((len(acc), 4))
            Q[0] = [1.0, 0.0, 0.0, 0.0] 
            
            # Iteración (Madgwick)
            for t in range(1, len(acc)):
                Q[t] = madgwick.updateIMU(Q[t-1], gyr[t], acc[t])
            
            # 4. Guardar con nombres para OpenSim
            base = f"imu_{sensor_id}"
            sto_data[f'{base}_q0'] = Q[:, 0]
            sto_data[f'{base}_q1'] = Q[:, 1]
            sto_data[f'{base}_q2'] = Q[:, 2]
            sto_data[f'{base}_q3'] = Q[:, 3]
            
        except IndexError:
            print(f" Saltando Sensor {sensor_id} (datos incompletos)")
            continue
            
    return sto_data