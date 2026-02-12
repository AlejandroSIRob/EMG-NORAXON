"""
FUSION MULTIMODAL AUTOMÁTICA (V14 - FINAL ROBUST FIX)
---------------------------------------------------------
Autor: Alejandro Solar Iglesias
Objetivo: 
  1. Procesado Dual: Raw (2000Hz) -> Procesado (100Hz).
  2. Sincronización: Fuerza + EMG + Xsens.
  3. ANÁLISIS CIENTÍFICO: Ejecuta automáticamente NMF y VAF usando 'synergy_lib'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import glob
import sys

# --- IMPORTACIÓN DE LIBRERÍAS PROPIAS ---
try:
    import noraxon_analytics as na
    import synergy_lib as lab  # <--- IMPORTAMOS TU LIBRERÍA DE SINERGIAS
    print("[INFO] Librerías científicas cargadas correctamente.")
except ImportError as e:
    print(f"[ERROR] Falta una librería en la carpeta: {e}")
    sys.exit()

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
RUTA_CARPETA_TOMA = r"C:\Users\alexs\Desktop\MUESTRAS\V1"
FS_MASTER = 100.0        
FS_EMG_ORIG = na.DEFAULT_FS # 2000 Hz

# Configuración para Sinergias (Nombres bonitos para los gráficos)
# NOTA: Las claves deben coincidir con los nombres generados en el paso 2
SENSOR_MAP_ANALISIS = {
    'EMG_RT FLEX.CARP.R (uV)': 'Flexor Rad.',
    'EMG_RT BRACHIORAD. (uV)': 'Braquiorradial',
    'EMG_RT EXT.DIG. (uV)': 'Extensor',
    'EMG_RT BICEPS BR. (uV)': 'Bíceps',
    'EMG_RT LAT. TRICEPS (uV)': 'Tríceps',
    'EMG_RT ANT.DELTOID (uV)': 'Delt. Ant.',
    'EMG_RT MID DELT. (uV)': 'Delt. Med.',
    'EMG_RT POST.DELTOID (uV)': 'Delt. Post.',
    'EMG_RT FLEX.CARP.U (uV)': 'Flexor Uln.',
    'EMG_RT PECT. MAJOR (uV)': 'Pectoral',
    'EMG_RT INFRASPIN. (uV)': 'Infraespinoso',
    'EMG_RT EXT.CARP.ULN. (uV)': 'Ext. Ulnar',
    'EMG_RT LAT.DORSI (uV)': 'Dorsal'
}

# =========================================================
# 2. MOTOR DE BÚSQUEDA Y UTILIDADES
# =========================================================
def buscar_archivo(patron_glob):
    archivos = glob.glob(patron_glob)
    return archivos[0] if archivos else None

def preparar_rutas():
    base_name = os.path.basename(RUTA_CARPETA_TOMA)
    print(f"\n--- INICIANDO PROCESADO MAESTRO: {base_name} ---")
    out_dir = os.path.join(RUTA_CARPETA_TOMA, "PROCESADO_COMPLETO")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # Localizar archivos críticos
    f_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "FUERZA", "*.csv"))
    # Fallback: buscar en la raíz si no está en FUERZA
    if not f_path: f_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "*v1.csv")) # Patrón amplio para fuerza

    e_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "EMG", "*.csv"))
    # Fallback: buscar en la raíz si no está en EMG
    if not e_path: e_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "*v1.csv")) 

    k_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "PROCESADO-Xsens", "*.sto"))
    
    return f_path, e_path, k_path, out_dir

def resample_signal(signal, old_time, new_time):
    f = interp1d(old_time, signal, kind='linear', fill_value="extrapolate")
    return f(new_time)

# =========================================================
# 3. MAIN (Procesado + Sinergias)
# =========================================================
def main():
    path_f, path_e, path_k, out_dir = preparar_rutas()

    if not (path_f and path_e):
        print("[ERROR] Faltan archivos de Fuerza o EMG. Abortando.")
        print(f"   Fuerza: {path_f}")
        print(f"   EMG:    {path_e}")
        return

    # --- A. CARGA Y SINCRONIZACIÓN (CORREGIDO) ---
    print("1. Cargando Datos...")
    
    # --- CORRECCIÓN FUERZA ROBUSTA ---
    try:
        # Intentamos cargar con header en fila 0 (estándar)
        df_force = pd.read_csv(path_f, skiprows=0)
        
        # Limpiar espacios en nombres de columnas
        df_force.columns = [str(c).strip() for c in df_force.columns]
        
        # Detección inteligente de la columna Time
        time_col_name = None
        candidates = ['Time', 'time', 'Time_s', 'time_s', 'Time (s)']
        for cand in candidates:
            if cand in df_force.columns:
                time_col_name = cand
                break
        
        if not time_col_name:
            # Si no encuentra el nombre, usa la PRIMERA columna por defecto
            time_col_name = df_force.columns[0]
            print(f"   [AVISO] No se encontró columna 'Time'. Usando la 1ra columna: '{time_col_name}'")

        force_cols = [c for c in df_force.columns if 'Force' in c or 'F' in c or 'T' in c] 
        # Filtro estricto para Fx, Fy, Fz...
        force_cols = [c for c in force_cols if c in ['Fx','Fy','Fz','Tx','Ty','Tz'] or 'Force' in c]
        
        # Renombrar y seleccionar
        df_force = df_force[[time_col_name] + force_cols].rename(columns={time_col_name: 'Time_F'})
        
    except Exception as e:
        print(f"[ERROR CRÍTICO AL LEER FUERZA]: {e}")
        return
    
    # EMG (Usando Noraxon Analytics)
    try:
        emg_dict, time_emg, fs_emg, _, _, _ = na.load_noraxon_csv_multi(path_e)
    except Exception as e:
        print(f"[ERROR CRÍTICO AL LEER EMG]: {e}")
        return
    
    # Kinematics (Xsens)
    if path_k:
        try:
            df_kin = pd.read_csv(path_k, sep='\t', skiprows=11) # Ajustar según STO
        except:
            df_kin = pd.DataFrame() # Fallback vacío
    else:
        print("[AVISO] No hay Kinematics. Se procesará sin Xsens.")

    # --- B. FUSIÓN (Resampling a 100Hz) ---
    print(f"2. Fusionando a {FS_MASTER} Hz...")
    
    # Definir vector de tiempo maestro
    t_start = df_force['Time_F'].min()
    t_end = df_force['Time_F'].max()
    
    if pd.isna(t_start) or pd.isna(t_end):
        print("[ERROR] La columna de tiempo en Fuerza contiene valores inválidos.")
        return

    t_master = np.arange(t_start, t_end, 1/FS_MASTER)
    
    data_fusion = {'Time': t_master}
    
    # 1. Resample Fuerza
    for col in force_cols:
        clean_col = "F_" + col.split('.')[-1] if '.' in col else col
        if clean_col in ['Fx', 'Fy', 'Fz']: clean_col = 'F_' + clean_col
        data_fusion[clean_col] = resample_signal(df_force[col].values, df_force['Time_F'].values, t_master)

    # 2. Resample y Procesado EMG (Envolvente)
    # CORRECCIÓN DE CLAVES PARA ANÁLISIS CIENTÍFICO
    for muscle, signal in emg_dict.items():
        # Limpiar nombre del músculo para evitar duplicados como "RT RT"
        muscle_clean = muscle.replace("RT ", "").replace("LT ", "").strip()
        
        # Procesado rápido (Rectificación)
        sig_proc = np.abs(signal) 
        
        # Generamos la clave en formato "EMG_RT MUSCULO (uV)"
        key_name = f"EMG_RT {muscle_clean} (uV)"
        data_fusion[key_name] = resample_signal(sig_proc, time_emg, t_master)

    # 3. Resample Kinematics (Si existe)
    if path_k and not df_kin.empty:
        try:
            time_k = df_kin.iloc[:,0].values # Asumimos col 0 es tiempo
            for col in df_kin.columns[1:]:
                # Filtrar columnas relevantes (Rotaciones, ángulos)
                if 'imu' in col.lower() or 'angle' in col.lower():
                     data_fusion[col] = resample_signal(df_kin[col].values, time_k, t_master)
        except Exception as e:
            print(f"[AVISO] Error procesando Kinematics: {e}")

    # --- C. GUARDADO DATASET MAESTRO ---
    df_final = pd.DataFrame(data_fusion)
    output_csv = os.path.join(out_dir, "DATASET_MAESTRO.csv")
    df_final.to_csv(output_csv, index=False)
    print(f"   [OK] Dataset Maestro guardado: {output_csv}")

    # =========================================================
    # 4. MÓDULO CIENTÍFICO (SYNERGY LIB INTEGRATION)
    # =========================================================
    print("\n=============================================")
    print("   INICIANDO ANÁLISIS DE SINERGIAS (NMF)    ")
    print("=============================================")
    
    try:
        # 1. Preparar DataFrame para Synergy Lib
        cols_emg_existentes = [c for c in SENSOR_MAP_ANALISIS.keys() if c in df_final.columns]
        
        if not cols_emg_existentes:
            print("[ADVERTENCIA] No se detectaron columnas EMG coincidentes para el análisis.")
            print("Esperaba:", list(SENSOR_MAP_ANALISIS.keys())[:3], "...")
            print("Encontré:", [c for c in df_final.columns if 'EMG' in c][:3], "...")
            return

        df_emg_analysis = df_final[cols_emg_existentes].rename(columns=SENSOR_MAP_ANALISIS)
        df_emg_analysis = df_emg_analysis.clip(lower=0)

        # 2. Ejecutar Pipeline de 'synergy_lib'
        task_name = "MAESTRO_ANALYSIS"
        
        print("-> A. Verificando Calidad de Señal...")
        lab.reportar_calidad_senal(df_emg_analysis)
        
        print("-> B. Analizando Redundancia...")
        lab.analizar_redundancia_pearson(df_emg_analysis, out_dir, task_name)
        
        print("-> C. Calculando Sinergias Óptimas...")
        resultado_optimo = lab.buscar_sinergias_optimas(df_emg_analysis)
        
        print("-> D. Generando Reportes Visuales...")
        df_ranking = lab.generar_ranking_y_graficos(df_emg_analysis, resultado_optimo, out_dir, task_name)
        
        print("\n   --- RESULTADO FINAL ---")
        print(f"   Sinergias Óptimas: {resultado_optimo['n']} (VAF: {resultado_optimo['vaf']:.2f}%)")
        print("   Top 3 Músculos:")
        print(df_ranking.head(3).to_string(index=False))

    except Exception as e:
        print(f"[ERROR EN ANÁLISIS CIENTÍFICO]: {e}")
        import traceback
        traceback.print_exc()

    print("\n[FIN] Ejecución completa.")

if __name__ == "__main__":
    main()