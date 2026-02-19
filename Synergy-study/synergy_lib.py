"""
LIBRERÍA CIENTÍFICA DE ANÁLISIS EMG (Funcional)
------------------------------------------------------------------
Autor: Alejandro Solar Iglesias
Fecha: Enero 2026

REFERENCIAS CIENTÍFICAS Y METODOLÓGICAS (CITAS EXACTAS):

[1] Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. 
    Nature, 401(6755). -> [Pág. 789]: Reglas de actualización multiplicativa (solver='mu').

[2] Pertusa Llopis, A. M. (2020). Estudio de sinergias musculares durante actividades de pedaleo. 
    Universidad de Alicante. -> [Sec. 5.6.1, Pág. 30]: Modelo E = W x H. [Sec. 6.4.2, Pág. 47]: Criterio VAF.

[3] Yokoyama, H., et al. (2019). Muscle Synergy Analysis: A practical protocol. Bio-protocol 9(10).
    -> [Step 3.2]: Filtros Pasa-Banda (20-450 Hz).
    -> [Step 3.6]: Criterio de selección (Global VAF > 90%).

[4] Konrad, P. (2005). The ABC of EMG. Noraxon Inc. USA.
    -> [Pág. 12]: Calidad de señal y línea base (<5uV). 
    -> [Pág. 27]: Procesamiento de Envolvente (RMS/Smoothing).

[5] Torres-Oviedo, G., & Ting, L. H. (2007). Muscle synergies characterizing muscle activation. 
    J. Neurophysiol, 98(4). -> [Pág. 2145, Eq. 1]: Fórmula matemática del VAF.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
import sys
# Esto le dice a Python que busque en la carpeta anterior la libreria
ruta_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_lib not in sys.path:
    sys.path.append(ruta_lib)
# ---------------------------------------
import noraxon_analytics as na

# --- CONFIGURACIÓN GLOBAL (Basada en Literatura) ---

# [Yokoyama 2019, Step 3.2]: "Band-pass filtered (20–450 Hz) to remove movement artifacts."
FILTER_LOW = 20    
FILTER_HIGH = 450  
ENVELOPE_HZ = 6    # [Konrad 2005, Pág. 27]: Suavizado para simular la contracción mecánica.

# [Konrad 2005, Pág. 12]: "Good baseline noise should be low... < 5uV ideally."
NOISE_THRESHOLD_UV = 5.0 

# --- FUNCIONES DE CARGA Y PROCESAMIENTO ---

def cargar_y_procesar_datos(folder_path, emg_map):
    """
    Carga, filtra y concatena señales EMG siguiendo el protocolo de Yokoyama (2019).
    Retorna un DataFrame limpio y activo.
    """
    print(f"-> Cargando datos desde: {folder_path}...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    accumulated_signals = {name: [] for name in emg_map.values()}
    
    files_processed = 0
    
    for file_path in csv_files:
        # Excluir calibraciones
        if any(x in file_path.lower() for x in ["mvc", "rom", "calib", "dados"]): continue
        try:
            signals, _, fs_file, _, _, _ = na.load_noraxon_csv_multi(file_path)
            if not signals: continue
            
            for k_full, data in signals.items():
                for k_map, v_clean in emg_map.items():
                    if k_map in k_full:
                        # [Yokoyama 2019, Step 3.2]: Filtrado Pasa-Banda
                        sig_filt = na.butter_bandpass_filter(na.remove_dc_offset(data), FILTER_LOW, FILTER_HIGH, fs_file)
                        # [Konrad 2005, Pág. 27]: Envolvente Lineal
                        env = na.compute_linear_envelope(sig_filt, fs_file, cutoff=ENVELOPE_HZ)
                        accumulated_signals[v_clean].append(env)
                        break
            files_processed += 1
        except Exception as e:
            print(f"   [AVISO] Error en archivo {os.path.basename(file_path)}: {e}")

    # Concatenación
    final_data = {k: np.concatenate(v) for k, v in accumulated_signals.items() if v}
    if not final_data: 
        raise ValueError("No se encontraron datos válidos en la carpeta.")
        
    min_len = min(len(v) for v in final_data.values())
    df = pd.DataFrame({k: v[:min_len] for k, v in final_data.items()})
    
    # Filtro de Silencio Basal [Konrad 2005]
    df_active = df[df.mean(axis=1) > NOISE_THRESHOLD_UV]
    
    print(f"   [DATOS] Archivos usados: {files_processed}. Muestras útiles: {len(df_active)}")
    return df_active

# --- FUNCIONES DE ANÁLISIS PREVIO ---

def reportar_calidad_senal(df_emg):
    """
    Auditoría de Ruido Basal.
    Referencia: [Konrad 2005, Pág. 12 "Signal Check"].
    """
    print("\n   --- 1. AUDITORÍA DE CALIDAD (SNR) [Konrad 2005] ---")
    quality_report = []
    for sensor in df_emg.columns:
        stats = na.calculate_signal_quality_snr(df_emg[sensor].values)
        noise = stats['Noise_Floor_uV']
        calidad = "CRÍTICA " if noise > 15.0 else "ÓPTIMA "
        quality_report.append({'Sensor': sensor, 'Ruido_uV': round(noise, 2), 'Calidad': calidad})

    df_q = pd.DataFrame(quality_report).sort_values(by='Ruido_uV')
    # print(df_q.to_string(index=False)) # Descomentar si se quiere ver en consola
    return df_q

def analizar_redundancia_pearson(df_active, output_folder, task_name):
    """
    Análisis de Redundancia Funcional (Pearson).
    Criterio Estadístico: r > 0.90 indica correlación muy fuerte (Evans, 1996).
    """
    print("\n   --- 2. ANÁLISIS DE REDUNDANCIA (PEARSON) ---")
    corr = df_active.corr()
    
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Redundancia Funcional - {task_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Pearson_{task_name}.png"), dpi=300)
    print("   [VISUALIZAR] Se abrirá la matriz de Pearson. Cierra la ventana para continuar...")
    #plt.show() 
    plt.close()
    
    found_redundancy = False
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr.iloc[i, j] > 0.90:
                print(f"   [ALERTA] Alta Redundancia: {cols[i]} y {cols[j]} (r={corr.iloc[i, j]:.2f})")
                found_redundancy = True
    
    if not found_redundancy:
        print("   [OK] No se detectó redundancia crítica.")

# --- FUNCIONES MATEMÁTICAS CENTRALES (NNMF + VAF) ---

def calcular_vaf(original, reconstructed):
    """
    Calcula el VAF (Variance Accounted For).
    Referencia: [Torres-Oviedo 2007, Pág. 2145, Eq. 1].
    """
    numerator = np.sum((original - reconstructed) ** 2)
    denominator = np.sum(original ** 2)
    vaf = (1 - numerator / denominator) * 100
    return max(vaf, 0)

def extraer_sinergias_nnmf(df_emg, n_synergies):
    """
    Algoritmo NNMF (Non-negative Matrix Factorization).
    Referencia: [Lee & Seung 1999, Pág. 789].
    """
    X = df_emg.values.T 
    X = np.maximum(X, 0) # [Lee & Seung 1999]: Restricción de no-negatividad.
    
    model = NMF(n_components=n_synergies, init='random', random_state=42, 
                solver='mu', max_iter=3000, tol=1e-4)
    
    W = model.fit_transform(X)
    H = model.components_
    
    X_rec = np.dot(W, H)
    vaf = calcular_vaf(X, X_rec)
    
    return W, H, vaf

def buscar_sinergias_optimas(df_active):
    """
    Itera para encontrar el número de sinergias que cumple VAF > 90%.
    Referencia: [Yokoyama 2019, Step 3.6] y [Torres-Oviedo 2007].
    """
    print("\n   --- 3. BÚSQUEDA DE SINERGIAS ÓPTIMAS (VAF > 90%) ---")
    best_result = None
    
    for n in range(1, 9):
        W, H, vaf = extraer_sinergias_nnmf(df_active, n)
        print(f"     -> {n} Sinergias | VAF: {vaf:.2f}%")
        
        if vaf > 90.0:
            print(f"     [!] Criterio alcanzado con {n} sinergias.")
            best_result = {'n': n, 'W': W, 'vaf': vaf}
            break
    else:
        best_result = {'n': n, 'W': W, 'vaf': vaf}
        
    return best_result

# --- FUNCIONES DE VISUALIZACIÓN Y RANKING ---

def generar_ranking_y_graficos(df_active, best_result, output_folder, task_name):
    """
    Genera los gráficos de Matriz W y Ranking de Sensores.
    Guarda los resultados en la carpeta especificada.
    """
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    n_syn = best_result['n']
    W = best_result['W']
    vaf = best_result['vaf']
    muscle_names = list(df_active.columns)
    
    # Normalización
    W_norm = W / W.max(axis=0)

    # 1. Gráfico Matriz W
    plt.figure(figsize=(12, 6))
    x = np.arange(len(muscle_names))
    width = 0.8 / n_syn
    for i in range(n_syn):
        plt.bar(x + i*width, W_norm[:, i], width=width, label=f'Sinergia {i+1}')
    plt.xticks(x + width*(n_syn-1)/2, muscle_names, rotation=45, ha='right')
    plt.title(f'Estructura de Sinergias ({task_name}) - VAF: {vaf:.1f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"W_Sinergias_{task_name}.png"), dpi=300)
    print("   [VISUALIZAR] Se abrirá el gráfico de Sinergias. Cierra la ventana para continuar...")
    #plt.show()
    plt.close()

    # 2. Ranking
    total_importance = np.sum(W_norm, axis=1)
    df_rank = pd.DataFrame({'Musculo': muscle_names, 'Score': total_importance}).sort_values(by='Score', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Musculo', hue='Musculo', data=df_rank, palette='viridis', legend=False)
    plt.title(f'Ranking Sensores - {task_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Ranking_{task_name}.png"), dpi=300)
    print("   [VISUALIZAR] Se abrirá el Ranking. Cierra la ventana para continuar...")
    #plt.show()
    plt.close()
    
    return df_rank