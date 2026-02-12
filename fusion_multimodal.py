"""
FUSION MULTIMODAL AUTOMÁTICA (V22 - Global Sync & Composite EMG)
---------------------------------------------------------
Autor: Alejandro Solar Iglesias
Mejoras Críticas:
  - EMG: Crea una señal compuesta (Suma de músculos de impacto) y busca el 
    máximo global, no local. Esto corrige desfases grandes entre dispositivos.
  - VISUALIZACIÓN: Normaliza las señales en las gráficas para verificar la 
    sincronización visualmente (0 a 1).
  - FUERZA: Mantiene la lógica de Triple Clap / Primer Contacto.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import os
import glob
import sys

# INTENTO DE IMPORTAR LIBRERÍA CIENTÍFICA
try:
    import noraxon_analytics as na
except ImportError:
    print("[ERROR] No se encuentra 'noraxon_analytics.py'.")
    sys.exit()

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
FS_MASTER = 100.0        
FS_EMG_ORIG = na.DEFAULT_FS 

# Configuración de Búsqueda
SEARCH_WINDOW_S = 60.0      # Analizar hasta 60s de grabación
SKIP_START_S = 2.0          # Ignorar los primeros 2s (ruido botón)

# Umbrales Fuerza
UMBRAL_CONTACTO_N = 1.0     
MIN_DIST_CLAP_S = 0.15      
FRACCIONES_UMBRAL = [0.5, 0.3, 0.1] 

# =========================================================
# 2. GESTIÓN DE ARCHIVOS
# =========================================================
def buscar_archivo(patron_glob):
    archivos = glob.glob(patron_glob)
    return archivos[0] if archivos else None

def preparar_rutas(ruta_base):
    if not os.path.exists(ruta_base):
        print(f"[ERROR] Ruta no existe: {ruta_base}")
        sys.exit(1)
        
    print(f"\n--- PROCESANDO: {os.path.basename(ruta_base)} ---")
    out_dir = os.path.join(ruta_base, "PROCESADO_COMPLETO")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    f_path = buscar_archivo(os.path.join(ruta_base, "FUERZA", "*.csv"))
    e_path = buscar_archivo(os.path.join(ruta_base, "EMG", "*.csv"))
    k_path = buscar_archivo(os.path.join(ruta_base, "PROCESADO-Xsens", "*.sto"))
    
    missing = []
    if not f_path: missing.append("Carpeta FUERZA")
    if not e_path: missing.append("Carpeta EMG")
    if not k_path: missing.append("Carpeta Xsens")
    
    if missing:
        print(f"[ERROR CRÍTICO] Faltan archivos en: {', '.join(missing)}")
        return None, None, None, None

    return f_path, e_path, k_path, out_dir

# =========================================================
# 3. MOTORES DE SINCRONIZACIÓN
# =========================================================

def detectar_t0_fuerza(df_fuerza):
    """
    Encuentra el instante del golpe principal en la señal de fuerza.
    """
    # 1. Inversión si es necesario (para que picos sean positivos)
    if df_fuerza['Fz'].mean() < -5.0: 
        df_fuerza['Fz'] = df_fuerza['Fz'] * -1
    
    # 2. Tara y Magnitud
    offset = df_fuerza['Fz'].iloc[0:20].mean()
    fz = (df_fuerza['Fz'] - offset).values
    fz_abs = np.abs(fz)
    time = df_fuerza['Time_s'].values
    
    # 3. Ventana de búsqueda
    mask = (time > SKIP_START_S) & (time < SEARCH_WINDOW_S)
    fz_search = fz_abs[mask]
    time_search = time[mask]
    
    if len(fz_search) == 0: return time[0]
    
    max_val = np.max(fz_search)
    
    # ESTRATEGIA A: TRIPLE CLAP (Adaptativo)
    for frac in FRACCIONES_UMBRAL:
        umbral = max(max_val * frac, 5.0)
        peaks, _ = find_peaks(fz_search, height=umbral, distance=int(MIN_DIST_CLAP_S * 100))
        
        if len(peaks) >= 3:
            t1, t3 = time_search[peaks[0]], time_search[peaks[2]]
            if (t3 - t1) < 5.0:
                print(f"[SYNC FUERZA] Triple Clap detectado (Pico={fz_search[peaks[0]]:.1f}N) -> T={t1:.2f}s")
                return t1

    # ESTRATEGIA B: PRIMER CONTACTO (Si no hay claps claros)
    mask_contact = fz_search > UMBRAL_CONTACTO_N
    if np.any(mask_contact):
        idx = np.argmax(mask_contact)
        t_first = time_search[idx]
        print(f"[SYNC FUERZA] Modo Primer Contacto -> T={t_first:.2f}s")
        return t_first
    
    # ESTRATEGIA C: MAX PEAK (Fallback)
    t_max = time_search[np.argmax(fz_search)]
    print(f"[SYNC FUERZA] Fallback Max Peak -> T={t_max:.2f}s")
    return t_max

def detectar_t0_emg(df_emg, time_emg):
    """
    Encuentra el 'Estallido Muscular' principal sumando los canales clave.
    Busca en TODO el archivo (primeros 30s) para corregir desfases grandes.
    """
    # 1. Identificar canales de "Impacto" (Tríceps, Pectoral, Flexores)
    # Si no existen, usa todos los disponibles.
    target_keywords = ['TRICEPS', 'PECT', 'FLEX', 'DELT']
    cols_impact = []
    
    all_emg_cols = [c for c in df_emg.columns if '(uV)' in c and 'Switch' not in c]
    
    for col in all_emg_cols:
        if any(key in col.upper() for key in target_keywords):
            cols_impact.append(col)
            
    if not cols_impact: cols_impact = all_emg_cols # Fallback a todos
    
    # 2. Construir Señal Compuesta (Suma de Envolventes Rápidas)
    # Usamos una envolvente rápida (20Hz) para detectar el pico con precisión temporal
    composite_signal = np.zeros(len(time_emg))
    
    for col in cols_impact:
        raw = df_emg[col].astype(float).values
        # Rectificación y normalización simple
        rect = np.abs(raw - np.mean(raw))
        mx = np.max(rect)
        if mx > 0:
            composite_signal += (rect / mx) # Suma normalizada
            
    # 3. Buscar el Pico Máximo en la ventana de búsqueda
    # Asumimos que el golpe es el evento de mayor activación muscular conjunta
    mask = (time_emg > SKIP_START_S) & (time_emg < SEARCH_WINDOW_S)
    
    if np.sum(mask) == 0: return 0.0
    
    comp_search = composite_signal[mask]
    time_search = time_emg[mask]
    
    # Suavizado ligero para evitar picos de ruido de 1 muestra
    # (Promedio móvil simple de 50ms)
    window_smooth = int(0.05 * FS_EMG_ORIG)
    comp_smooth = np.convolve(comp_search, np.ones(window_smooth)/window_smooth, mode='same')
    
    idx_max = np.argmax(comp_smooth)
    t_sync = time_search[idx_max]
    
    print(f"[SYNC EMG] Estallido Muscular Compuesto detectado -> T={t_sync:.2f}s")
    return t_sync

def detectar_t0_xsens(df_k):
    cols_hand = [c for c in df_k.columns if 'hand' in c.lower() or 'mano' in c.lower()]
    if not cols_hand: return 0.0
    
    col_target = cols_hand[0]
    time = df_k['time'].values
    
    try:
        if df_k[col_target].dtype == object:
            val = df_k[col_target].astype(str).str.split(',', expand=True)[0].astype(float).values
        else:
            val = df_k[col_target].values
        motion = np.abs(np.diff(val, prepend=val[0]))
    except: return 0.0
    
    mask = (time > SKIP_START_S) & (time < SEARCH_WINDOW_S)
    if np.sum(mask) == 0: return 0.0
    
    mot_search = motion[mask]
    time_search = time[mask]
    
    # Buscar Triple Gesto o Max
    peaks, _ = find_peaks(mot_search, height=np.max(mot_search)*0.3, distance=10)
    
    if len(peaks) >= 3:
        if (time_search[peaks[2]] - time_search[peaks[0]]) < 5.0:
            print(f"[SYNC XSENS] Triple Gesto detectado -> T={time_search[peaks[0]]:.2f}s")
            return time_search[peaks[0]]
            
    idx_max = np.argmax(mot_search)
    print(f"[SYNC XSENS] Movimiento Máximo detectado -> T={time_search[idx_max]:.2f}s")
    return time_search[idx_max]

# =========================================================
# 4. MAIN
# =========================================================
def main():
    if len(sys.argv) < 2:
        print("Uso: python fusion.py <RUTA>")
        sys.exit(1)
    
    ruta_toma = sys.argv[1] 
    path_f, path_e, path_k, out_dir = preparar_rutas(ruta_toma)
    if None in [path_f, path_e, path_k]: return

    try:
        # --- CARGA ---
        df_f = pd.read_csv(path_f)
        df_f['Time_s'] -= df_f['Time_s'].iloc[0]
        if df_f['Fz'].min() < -50: df_f['Fz'] = df_f['Fz'] * -1
        
        df_k = pd.read_csv(path_k, sep='\t', skiprows=5)
        
        df_e_raw = pd.read_csv(path_e, sep=';', decimal=',', quotechar='"', skiprows=3, low_memory=False)
        df_e_raw.columns = [c.replace('"', '').strip() for c in df_e_raw.columns]
        
        if df_e_raw['time'].dtype == object:
            time_e_orig = df_e_raw['time'].str.replace(',', '.').astype(float).values
        else:
            time_e_orig = df_e_raw['time'].values

        # --- SINCRONIZACIÓN INDEPENDIENTE ---
        # Calculamos el T=0 ideal para CADA dispositivo por separado
        t_f_sync = detectar_t0_fuerza(df_f)
        t_e_sync = detectar_t0_emg(df_e_raw, time_e_orig)
        t_k_sync = detectar_t0_xsens(df_k)

        print(f"-> PUNTOS CERO: Fuerza={t_f_sync:.2f}s | EMG={t_e_sync:.2f}s | Xsens={t_k_sync:.2f}s")

        # --- FUSIÓN Y REMUESTREO (100 Hz) ---
        # Creamos una base de tiempo común de -2s a +Fin
        t_max = df_f['Time_s'].max() - t_f_sync
        t_master = np.arange(-2.0, t_max, 1/FS_MASTER)
        df_out = pd.DataFrame({'Time': t_master})
        f_remap = lambda t, y: interp1d(t, y, bounds_error=False, fill_value=0)(t_master)

        # 1. Fuerza (Alineada a su T0)
        for c in ['Fx', 'Fy', 'Fz']: 
            df_out[f'F_{c}'] = f_remap(df_f['Time_s'] - t_f_sync, df_f[c])

        # 2. EMG (Alineado a su T0)
        emg_cols = [c for c in df_e_raw.columns if 'uV' in c and 'Switch' not in c]
        dict_raw = {} # Para gráfica raw

        for col in emg_cols:
            raw = df_e_raw[col].astype(float).values
            cent = na.remove_dc_offset(raw)
            dict_raw[col] = cent
            # Procesado estándar (Filtro + Envolvente)
            filt = na.butter_bandpass_filter(cent, 20, 450, FS_EMG_ORIG)
            env = na.compute_linear_envelope(filt, FS_EMG_ORIG, 6)
            # Alinear usando el t_e_sync calculado
            df_out[f'EMG_{col}'] = f_remap(time_e_orig - t_e_sync, env)

        # 3. Xsens (Alineado a su T0)
        for c in [x for x in df_k.columns if 'imu' in x]:
            try:
                if df_k[c].dtype == object:
                    qs = df_k[c].astype(str).str.split(',', expand=True).astype(float)
                    for i in range(4): 
                        if i < qs.shape[1]: df_out[f'{c}_q{i}'] = f_remap(df_k['time'] - t_k_sync, qs[i])
                else:
                    df_out[f'{c}'] = f_remap(df_k['time'] - t_k_sync, df_k[c])
            except: pass

        # --- GUARDAR ---
        df_out.to_csv(os.path.join(out_dir, "DATASET_MAESTRO.csv"), index=False)
        generar_imagenes_suite(df_out, dict_raw, time_e_orig, t_e_sync, out_dir)
        print(f"[EXITO] {os.path.basename(ruta_toma)} procesada.\n")
        
    except Exception as e:
        print(f"[ERROR EXCEPCIÓN] Fallo en {ruta_toma}: {e}\n")

# =========================================================
# 5. VISUALIZACIÓN
# =========================================================
def generar_imagenes_suite(df, dict_raw, t_orig, t_sync_e, folder):
    emg_cols = [c for c in df.columns if 'EMG' in c]
    n_musc = len(emg_cols)
    
    # 1. PROCESADO
    fig1, axes1 = plt.subplots(n_musc, 1, figsize=(12, 2 * n_musc), sharex=True)
    if n_musc == 1: axes1 = [axes1]
    for i, col in enumerate(emg_cols):
        axes1[i].plot(df['Time'], df[col], color='green', lw=1.2)
        axes1[i].set_title(f"PROCESADA: {col.replace('EMG_', '')}", loc='left', fontsize=9)
        axes1[i].axvline(0, color='k', linestyle='--', lw=1)
    plt.tight_layout(); plt.savefig(os.path.join(folder, "DESGLOSE_1_PROCESADO.png"), dpi=100)
    plt.close(fig1)

    # 2. RAW
    fig2, axes2 = plt.subplots(n_musc, 1, figsize=(12, 2 * n_musc), sharex=True)
    if n_musc == 1: axes2 = [axes2]
    t_raw = t_orig - t_sync_e
    for i, (name, sig) in enumerate(dict_raw.items()):
        axes2[i].plot(t_raw, sig, color='gray', linewidth=0.5)
        axes2[i].set_title(f"RAW: {name}", loc='left', fontsize=9)
        axes2[i].axvline(0, color='k', linestyle='--', lw=1)
        axes2[i].set_xlim(-1, 5)
    plt.tight_layout(); plt.savefig(os.path.join(folder, "DESGLOSE_2_RAW.png"), dpi=100)
    plt.close(fig2)

    # 3. CONTROL NORMALIZADO (Para ver sync real)
    plt.figure(figsize=(10, 6))
    
    # Fuerza Normalizada
    f_norm = df['F_Fz'] / (df['F_Fz'].max() + 1e-6)
    plt.plot(df['Time'], f_norm, 'r', label='Fuerza (Norm)', lw=2)
    
    # EMG Promedio Normalizado (Suma de todos)
    if n_musc > 0:
        avg_emg = df[emg_cols].mean(axis=1)
        e_norm = avg_emg / (avg_emg.max() + 1e-6)
        plt.plot(df['Time'], e_norm, 'g', alpha=0.6, label='EMG Promedio (Norm)', lw=1.5)
        
    plt.axvline(0, color='k', linestyle='--', label='T=0 (Sincronización)')
    plt.title("VALIDACIÓN DE SINCRONIZACIÓN (Señales Normalizadas 0-1)")
    plt.legend(loc='upper right')
    plt.xlim(-1.0, 1.0) # Zoom de 2 segundos alrededor del golpe
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder, "DASHBOARD_CONTROL.png"), dpi=100)
    plt.close()

if __name__ == "__main__":
    main()