"""
FUSION MULTIMODAL: FUERZA (OnRobot) + EMG (Noraxon) + CINEMÁTICA (OpenSim)
-------------------------------------------------------------------------
Autor: Alejandro Solar Iglesias
Objetivo: Sincronizar y analizar datos heterogéneos para Tesis/TFM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Importamos tus librerías científicas
import noraxon_analytics as na
import synergy_lib as sl

# --- CONFIGURACIÓN ---
FS_MASTER = 2000.0  # Frecuencia maestra (usualmente la de los EMG es la más alta)

def cargar_fuerza_onrobot(csv_path):
    """Carga datos del script hexFT.py"""
    print(f"-> Cargando Fuerza: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    # El script hexFT guarda columnas: Time_s, SampleCount, Fx, Fy, Fz...
    # Aseguramos que el tiempo empiece en 0 relativo
    df['Time_s'] = df['Time_s'] - df['Time_s'].iloc[0]
    return df

def cargar_cinematica_opensim(mot_path):
    """Carga resultados de Cinemática Inversa (.mot) de OpenSim"""
    print(f"-> Cargando Cinemática: {os.path.basename(mot_path)}")
    
    # OpenSim .mot tiene un header variable, usualmente 'endheader' marca el fin
    with open(mot_path, 'r') as f:
        lines = f.readlines()
        skip = 0
        for i, line in enumerate(lines):
            if 'endheader' in line:
                skip = i + 1
                break
    
    df = pd.read_csv(mot_path, sep='\t', skiprows=skip)
    return df

def sincronizar_senales(df_fuerza, df_emg_accel, umbral_fuerza=2.0, umbral_accel=1.5):
    """
    ALGORITMO DE SINCRONIZACIÓN (EL "CLAQUEO")
    Busca el primer pico significativo en Fuerza (Fz) y en Acelerometría (Accel)
    y calcula el desfase (lag) para alinearlos.
    """
    print("\n[SINCRONIZACIÓN] Buscando evento de impacto (Claqueta)...")
    
    # 1. Detectar pico en Fuerza (Fz suele ser negativa al presionar, usamos abs)
    # Asumimos que los golpes de sincro ocurren en los primeros 10 segundos
    fz_slice = df_fuerza[df_fuerza['Time_s'] < 10]['Fz'].abs()
    idx_f = fz_slice[fz_slice > umbral_fuerza].first_valid_index()
    
    if idx_f is None:
        print("   [ADVERTENCIA] No se detectó pico de fuerza. Asumiendo T=0.")
        t_fuerza = 0
    else:
        t_fuerza = df_fuerza.loc[idx_f, 'Time_s']
        print(f"   - Impacto Fuerza detectado en T={t_fuerza:.3f}s")

    # 2. Detectar pico en Acelerómetro (Usamos la magnitud de un sensor clave, ej: Mano)
    # df_emg_accel suele venir como diccionario de numpy arrays desde tu librería.
    # Convertimos a DataFrame temporal para buscar
    # Buscamos en el primer sensor de aceleración disponible
    keys_accel = [k for k in df_emg_accel.keys() if 'Accel' in k]
    if not keys_accel:
        print("   [ERROR] No hay datos de acelerometría para sincronizar.")
        return 0
    
    acc_data = df_emg_accel[keys_accel[0]] # Tomamos el primero
    # Crear eje de tiempo temporal para accel
    t_accel_vec = np.arange(len(acc_data)) / FS_MASTER
    
    # Buscar pico (Umbral en Gs o mG)
    # Noraxon suele dar mG. 1500 mG = 1.5 G
    idx_a = np.where(np.abs(acc_data) > (umbral_accel * 1000))[0] # Convertir a mG si necesario
    
    if len(idx_a) == 0:
        print("   [ADVERTENCIA] No se detectó pico de aceleración. Asumiendo T=0.")
        t_accel = 0
    else:
        t_accel = t_accel_vec[idx_a[0]]
        print(f"   - Impacto Accel detectado en T={t_accel:.3f}s")
        
    # 3. Calcular Desfase (Shift)
    # Si Fuerza ocurre en t=5 y Accel en t=8, EMG va adelantada 3s.
    # Queremos alinear todo al tiempo de la Fuerza (nuestro reloj maestro robótico)
    lag = t_accel - t_fuerza
    print(f"   -> DESFASE CALCULADO: {lag:.3f}s (Se ajustará el EMG/IMU)")
    
    return lag

def remapear_a_frecuencia_maestra(tiempo_maestro, tiempo_sen, senal_val):
    """Interpolación para poner todas las señales en el mismo vector de tiempo"""
    f = interp1d(tiempo_sen, senal_val, kind='linear', bounds_error=False, fill_value=0)
    return f(tiempo_maestro)

def main_fusion():
    # --- 1. DEFINIR RUTAS DE ARCHIVOS  ---
    # Cambia esto por tus rutas reales o usa argparse
    path_fuerza = "datos_sensor_fuerza.csv"
    path_emg = "datos_noraxon.csv"
    # path_ik = "Resultados_Linux/ik_result.mot" # Opcional si ya procesaste OpenSim
    
    print("=== INICIANDO FUSIÓN DE DATOS MULTIMODAL ===")
    
    # A. Cargar Datos
    try:
        df_force = cargar_fuerza_onrobot(path_fuerza)
        
        # Usamos tu librería noraxon para cargar todo (EMG + Accel)
        emg_dict, time_emg, fs_emg, accel_dict, _, _ = na.load_noraxon_csv_multi(path_emg)
        
    except FileNotFoundError as e:
        print(f"[ERROR CRÍTICO] Falta archivo: {e}")
        return

    # B. Sincronización
    # Usamos los acelerómetros de Noraxon para sincronizar con la Fuerza
    lag_emg = sincronizar_senales(df_force, accel_dict)
    
    # Ajustamos el tiempo del EMG
    # Si lag es positivo (EMG empezó después), restamos. 
    # La lógica exacta depende de quién empezó a grabar primero. 
    # Asumimos aquí: Alineamos el pico EMG al tiempo 0 del pico Fuerza.
    time_emg_sync = time_emg - lag_emg

    # C. Construcción del DataFrame Maestro
    # Usaremos el vector de tiempo de la FUERZA como base (ej. 1000Hz)
    # O creamos uno nuevo regular a 2000Hz (FS_MASTER)
    duracion = min(df_force['Time_s'].max(), time_emg_sync[-1])
    t_master = np.arange(0, duracion, 1/FS_MASTER)
    
    df_master = pd.DataFrame({'Time': t_master})
    
    print("\n[PROCESAMIENTO] Re-muestreado y fusión de señales...")
    
    # 1. Insertar Fuerza (Interpolada)
    for col in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
        df_master[f'Force_{col}'] = remapear_a_frecuencia_maestra(
            t_master, df_force['Time_s'], df_force[col]
        )
        
    # 2. Insertar EMG (Procesado con tu pipeline)
    for musculo, raw_signal in emg_dict.items():
        # A. Quitar Offset
        sig_centered = na.remove_dc_offset(raw_signal)
        # B. Filtro Pasa-Banda (20-450)
        sig_filt = na.butter_bandpass_filter(sig_centered, 20, 450, fs_emg)
        # C. Envolvente (6Hz)
        sig_env = na.compute_linear_envelope(sig_filt, fs_emg, cutoff=6)
        
        # D. Interpolar al tiempo maestro
        df_master[f'EMG_{musculo}'] = remapear_a_frecuencia_maestra(
            t_master, time_emg_sync, sig_env
        )

    # D. Análisis de la Tarea (Ej. Corte)
    # Detectar cuando Fz supera 5 Newtons (Contacto con carne)
    mask_corte = df_master['Force_Fz'].abs() > 5.0
    
    if mask_corte.any():
        print(f"\n[ANÁLISIS] Detectadas fases de contacto (Fuerza > 5N)")
        df_corte = df_master[mask_corte]
        
        mean_force = df_corte['Force_Fz'].mean()
        peak_force = df_corte['Force_Fz'].min() # Negativo hacia abajo
        print(f"   - Fuerza Media de Corte: {mean_force:.2f} N")
        print(f"   - Pico Máximo: {peak_force:.2f} N")
        
        # Correlación EMG vs Fuerza en esa fase
        print("   - Correlación Muscular con Fuerza Z:")
        corrs = {}
        for col in df_master.columns:
            if 'EMG_' in col:
                r = df_corte['Force_Fz'].corr(df_corte[col])
                corrs[col] = r
        
        # Mostrar Top 3 músculos implicados en la fuerza
        sorted_muscles = sorted(corrs.items(), key=lambda item: abs(item[1]), reverse=True)
        for name, r in sorted_muscles[:3]:
            print(f"     * {name}: r = {r:.3f}")

        # E. Sinergias en la fase de corte
        # Preparamos dataframe solo de EMG para tu librería synergy_lib
        emg_cols = [c for c in df_master.columns if 'EMG_' in c]
        df_emg_only = df_corte[emg_cols].copy()
        # Renombramos quitando prefijo para que quede limpio en el gráfico
        df_emg_only.columns = [c.replace('EMG_', '') for c in df_emg_only.columns]
        
        # Llamamos a tu librería (adaptada para recibir DF directo)
        print("\n[SINERGIAS] Calculando módulos de control durante el corte...")
        res_optimo = sl.buscar_sinergias_optimas(df_emg_only)
        sl.generar_ranking_y_graficos(df_emg_only, res_optimo, "Resultados_Fusion", "Corte_Carne_Real")

    # F. Visualización Final
    plt.figure(figsize=(12, 8))
    
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df_master['Time'], df_master['Force_Fz'], 'k', label='Fuerza Z (N)')
    ax1.set_title("Fuerza de Interacción")
    ax1.grid(True)
    ax1.legend()
    
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    # Graficar solo los 3 músculos más activos
    top_muscles = [m[0] for m in sorted_muscles[:3]] if 'sorted_muscles' in locals() else []
    for col in top_muscles:
        ax2.plot(df_master['Time'], df_master[col], label=col.replace('EMG_', ''))
    ax2.set_title("Activación Muscular (Top 3)")
    ax2.grid(True)
    ax2.legend()
    
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    # Si tuvieras datos de IK (ángulos), irían aquí
    # Por ahora graficamos Fuerza Resultante vs Suma EMG
    emg_sum = df_master[[c for c in df_master.columns if 'EMG_' in c]].sum(axis=1)
    ax3.plot(df_master['Time'], emg_sum, 'r--', label='Suma EMG Total')
    ax3.set_title("Esfuerzo Muscular Total Estimado")
    ax3.set_xlabel("Tiempo (s)")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_fusion()