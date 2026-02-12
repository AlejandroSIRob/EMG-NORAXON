"""
FUSION MULTIMODAL AUTOMÁTICA (V11 - Full Scientific Suite)
---------------------------------------------------------
Autor: Alejandro Solar Iglesias
Objetivo: 
  - Procesado Dual: Raw (2000Hz) para visualización y Procesado (100Hz) para datos.
  - Sincronización triple: Fuerza, EMG y Xsens.
  - Dashboards de Control y Validación de Calidad (SNR).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import glob
import sys

# IMPORTAMOS TU LIBRERÍA CIENTÍFICA
try:
    import noraxon_analytics as na
except ImportError:
    print("[ERROR] No se encuentra 'noraxon_analytics.py'. Asegúrate de que esté en la misma carpeta.")
    sys.exit()

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
RUTA_CARPETA_TOMA = r"C:\Users\alexs\Desktop\MUESTRAS\V1"
FS_MASTER = 100.0        
FS_EMG_ORIG = na.DEFAULT_FS # 2000 Hz

# =========================================================
# 2. MOTOR DE BÚSQUEDA
# =========================================================
def buscar_archivo(patron_glob):
    archivos = glob.glob(patron_glob)
    return archivos[0] if archivos else None

def preparar_rutas():
    print(f"\n--- INICIANDO PROCESADO MAESTRO: {os.path.basename(RUTA_CARPETA_TOMA)} ---")
    out_dir = os.path.join(RUTA_CARPETA_TOMA, "PROCESADO_COMPLETO")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # Localizar archivos críticos
    f_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "FUERZA", "*.csv"))
    e_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "EMG", "*.csv"))
    k_path = buscar_archivo(os.path.join(RUTA_CARPETA_TOMA, "PROCESADO-Xsens", "*.sto"))
    
    return f_path, e_path, k_path, out_dir

# =========================================================
# 3. MAIN (Procesado Dual Completo)
# =========================================================
def main():
    path_f, path_e, path_k, out_dir = preparar_rutas()
    if None in [path_f, path_e, path_k]:
        print("[ERROR] Faltan archivos. Revisa que existan las carpetas FUERZA, EMG y PROCESADO-Xsens con sus archivos.")
        return

    # --- CARGA ---
    df_f = pd.read_csv(path_f)
    df_f['Time_s'] -= df_f['Time_s'].iloc[0]
    
    df_k = pd.read_csv(path_k, sep='\t', skiprows=5)
    
    df_e_raw = pd.read_csv(path_e, sep=';', decimal=',', quotechar='"', skiprows=3, low_memory=False)
    df_e_raw.columns = [c.replace('"', '').strip() for c in df_e_raw.columns]
    
    # Gestión robusta de la columna de tiempo (String vs Float)
    if df_e_raw['time'].dtype == object:
        time_e_orig = df_e_raw['time'].str.replace(',', '.').astype(float).values
    else:
        time_e_orig = df_e_raw['time'].values

    # --- SINCRONIZACIÓN ---
    # Detectamos el pico máximo para alinear el evento de impacto ("Clap")
    t_f_sync = df_f.loc[df_f['Fz'].abs().idxmax(), 'Time_s']
    t_e_sync = time_e_orig[df_e_raw['RT LAT. TRICEPS (uV)'].abs().idxmax()]
    t_k_sync = 21.32 # Valor de referencia Xsens obtenido manualmente para V1

    print(f"-> Sincronización calculada: F:{t_f_sync:.2f}s | E:{t_e_sync:.2f}s | K:{t_k_sync:.2f}s")

    # --- DATASET MAESTRO (100Hz) ---
    # Definimos una ventana desde 2 segundos antes del impacto hasta el final de la toma
    t_master = np.arange(-2.0, df_f['Time_s'].max() - t_f_sync, 1/FS_MASTER)
    df_out = pd.DataFrame({'Time': t_master})
    f_remap = lambda t, y: interp1d(t, y, bounds_error=False, fill_value=0)(t_master)

    # Alinear Fuerza
    for c in ['Fx', 'Fy', 'Fz']: 
        df_out[f'F_{c}'] = f_remap(df_f['Time_s'] - t_f_sync, df_f[c])

    # --- PROCESADO EMG CIENTÍFICO (DUAL) ---
    blacklist = ['time', 'Activity', 'Marker', 'Sync', 'Switch', 'Ultium EMG.Switch 1 (On)']
    emg_cols = [c for c in df_e_raw.columns if 'uV' in c and not any(b in c for b in blacklist)]
    dict_raw_2000hz = {}

    print("\nANÁLISIS DE CALIDAD Y PROCESADO (Standards Noraxon/SENIAM):")
    for col in emg_cols:
        raw_val = df_e_raw[col].astype(float).values
        
        # 1. Centrado (Remove DC Offset)
        centered = na.remove_dc_offset(raw_val)
        dict_raw_2000hz[col] = centered # Guardamos para la gráfica de alta resolución

        # 2. Filtrado y Envolvente para el Dataset Maestro
        filt = na.butter_bandpass_filter(centered, na.CUTOFF_LOW, na.CUTOFF_HIGH, FS_EMG_ORIG)
        env = na.compute_linear_envelope(filt, FS_EMG_ORIG, cutoff=na.ENVELOPE_CUTOFF)
        df_out[f'EMG_{col}'] = f_remap(time_e_orig - t_e_sync, env)
        
        # 3. Reporte de Calidad en consola
        qa = na.calculate_signal_quality_snr(centered)
        print(f" > {col:22} | SNR: {qa['SNR_dB']:4.1f} dB | {qa['Status']}")

    # --- KINEMATICS (XSENS) ---
    for c in [x for x in df_k.columns if 'imu' in x]:
        qs = df_k[c].str.split(',', expand=True).astype(float)
        for i in range(4): 
            df_out[f'{c}_q{i}'] = f_remap(df_k['time'] - t_k_sync, qs[i])

    # --- GUARDAR Y VISUALIZAR ---
    csv_path = os.path.join(out_dir, "DATASET_MAESTRO.csv")
    df_out.to_csv(csv_path, index=False)
    
    generar_imagenes_suite(df_out, dict_raw_2000hz, time_e_orig, t_e_sync, out_dir)
    print(f"\n[ÉXITO] Todo generado correctamente en: {out_dir}")

# =========================================================
# 4. SUITE DE VISUALIZACIÓN (4 GRÁFICAS)
# =========================================================
def generar_imagenes_suite(df, dict_raw, t_orig, t_sync_e, folder):
    emg_cols = [c for c in df.columns if 'EMG' in c]
    n_musc = len(emg_cols)
    
    # --- 1. DESGLOSE PROCESADO (Envolventes 6Hz) ---
    
    fig1, axes1 = plt.subplots(n_musc, 1, figsize=(12, 2 * n_musc), sharex=True)
    if n_musc == 1: axes1 = [axes1]
    for i, col in enumerate(emg_cols):
        axes1[i].plot(df['Time'], df[col], color='green', linewidth=1.5)
        axes1[i].set_title(f"PROCESADA (Env 6Hz): {col.replace('EMG_', '')}", loc='left', fontsize=10, fontweight='bold')
        axes1[i].axvline(0, color='black', linestyle='--', linewidth=1.5)
        axes1[i].set_ylabel("uV")
        axes1[i].grid(True, alpha=0.2)
    axes1[-1].set_xlabel("Tiempo (s)")
    plt.tight_layout(); plt.savefig(os.path.join(folder, "DESGLOSE_1_PROCESADO.png"), dpi=200)

    # --- 2. DESGLOSE RAW (Frecuencia Nativa 2000Hz) ---
    
    fig2, axes2 = plt.subplots(n_musc, 1, figsize=(12, 2 * n_musc), sharex=True)
    if n_musc == 1: axes2 = [axes2]
    t_rel_raw = t_orig - t_sync_e
    for i, (name, signal) in enumerate(dict_raw.items()):
        axes2[i].plot(t_rel_raw, signal, color='gray', linewidth=0.5, alpha=0.8)
        axes2[i].set_title(f"RAW (2000Hz): {name}", loc='left', fontsize=10, fontweight='bold')
        axes2[i].axvline(0, color='black', linestyle='--', linewidth=1.5)
        axes2[i].set_xlim(-1, 5) # Zoom para ver el detalle del impacto
        axes2[i].set_ylabel("uV")
        axes2[i].grid(True, alpha=0.2)
    axes2[-1].set_xlabel("Tiempo (s)")
    plt.tight_layout(); plt.savefig(os.path.join(folder, "DESGLOSE_2_RAW.png"), dpi=200)

    # --- 3. DASHBOARD CONTROL (Fuerza + Velocidad) ---
    fig3, (axf, axv) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axf.plot(df['Time'], df['F_Fz'], 'r', label='Fuerza Fz (N)')
    axf.axvline(0, color='black', linestyle='--', linewidth=2)
    axf.set_title("Dinámica de Carga (Eje Z)"); axf.grid(True); axf.legend()
    
    h_cols = [c for c in df.columns if 'hand_r_imu_q' in c]
    if len(h_cols) == 4:
        # Calculamos magnitud del cambio de orientación como proxy de velocidad
        vel = np.linalg.norm(np.diff(df[h_cols].values, axis=0, prepend=df[h_cols].values[0:1]), axis=1)
        axv.plot(df['Time'], vel, 'b', label='Velocidad Mano (IMU)')
        axv.axvline(0, color='black', linestyle='--', linewidth=2)
        axv.set_title("Cinemática del Segmento Mano"); axv.grid(True); axv.legend()
    axes3 = axv.set_xlabel("Tiempo (s)")
    plt.tight_layout(); plt.savefig(os.path.join(folder, "DASHBOARD_CONTROL.png"), dpi=200)

    # --- 4. VALIDACIÓN ZOOM (Sincronización al milisegundo) ---
    
    plt.figure(figsize=(10, 5))
    # Normalizamos señales para comparar el "timing" visualmente
    plt.plot(df['Time'], df['F_Fz']/df['F_Fz'].max(), 'r', label='Fuerza (Norm)')
    if n_musc > 0:
        plt.plot(df['Time'], df[emg_cols[0]]/df[emg_cols[0]].max(), 'g', alpha=0.5, label='EMG Ref (Norm)')
    
    plt.axvline(0, color='black', linestyle='--', linewidth=2, label='T=0 (Punto Maestro)')
    plt.xlim(-0.5, 0.5) # Zoom muy fuerte de 500ms
    plt.title("Auditoría de Sincronización (Zoom +/- 500ms)")
    plt.legend(); plt.grid(True); plt.xlabel("Tiempo (s)")
    plt.savefig(os.path.join(folder, "VALIDACION_SYNC_ZOOM.png"), dpi=200)
    plt.close('all')

if __name__ == "__main__":
    main()