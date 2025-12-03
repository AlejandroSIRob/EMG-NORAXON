"""
EMG CSV Analysis Tool
Alejandro Solar Iglesias 
=====================
Análisis completo de señales EMG desde archivos CSV.
Extrae características temporales, espectrales y detecta fatiga.

Uso:
    python EMG_CSV.py <ruta_csv> [--output <dir_salida>] [--plot] [--save-features]

Ejemplo:
    python EMG_CSV.py "..\\Pruebas\\2025-12-02-15-49_Free_Capture-3.csv" --plot --save-features
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, correlate


# --- CONFIGURACIÓN POR DEFECTO ---
DEFAULT_FS = 2000  # Frecuencia de muestreo (Hz)
CUTOFF_LOW = 20    # Paso alto (Hz)
CUTOFF_HIGH = 450  # Paso bajo (Hz)
ENVELOPE_CUTOFF = 6  # Paso bajo para envolvente (Hz)
FATIGUE_THRESHOLD = 0.9  # 90% de MDF inicial = fatiga detectada


# ============================================================================
# SECCIÓN 1: FILTRADO Y PROCESAMIENTO
# ============================================================================

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """
    Aplica filtro pasa banda Butterworth a la señal.
    
    Args:
        data: Señal de entrada
        lowcut: Frecuencia de corte baja (Hz)
        highcut: Frecuencia de corte alta (Hz)
        fs: Frecuencia de muestreo (Hz)
        order: Orden del filtro
    
    Returns:
        Señal filtrada
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def get_envelope(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """
    Obtiene la envolvente de la señal (rectificación + suavizado).
    
    Args:
        data: Señal filtrada
        cutoff: Frecuencia de corte del filtro paso bajo (Hz)
        fs: Frecuencia de muestreo (Hz)
        order: Orden del filtro
    
    Returns:
        Envolvente de la señal
    """
    abs_data = np.abs(data)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, abs_data)


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """Elimina la componente DC (media) de la señal."""
    return data - np.mean(data)


# ============================================================================
# SECCIÓN 2: CARACTERÍSTICAS EN DOMINIO DEL TIEMPO
# ============================================================================

def compute_time_domain_features(signal: np.ndarray, fs: int) -> Dict[str, float]:
    """
    Calcula características en el dominio del tiempo.
    
    Características:
        - RMS: Root Mean Square (Potencia de la señal)
        - MAV: Mean Absolute Value (Amplitud media)
        - WL: Waveform Length (Complejidad / Cambios rápidos)
        - ZC: Zero Crossings (Cruces por cero, aprox. frecuencia)
        - IEMG: Integral EMG (Energía acumulada)
    
    Args:
        signal: Señal EMG
        fs: Frecuencia de muestreo
    
    Returns:
        Diccionario con características
    """
    features = {}
    
    # 1. RMS
    features['RMS'] = float(np.sqrt(np.mean(signal**2)))
    
    # 2. MAV (Mean Absolute Value)
    features['MAV'] = float(np.mean(np.abs(signal)))
    
    # 3. WL (Waveform Length)
    features['WL'] = float(np.sum(np.abs(np.diff(signal))))
    
    # 4. ZC (Zero Crossings)
    features['ZC'] = float(np.sum(np.diff(np.sign(signal)) != 0))
    
    # 5. IEMG (Integral EMG)
    features['IEMG'] = float(np.sum(np.abs(signal)))
    
    # 6. Peak (Valor máximo)
    features['Peak'] = float(np.max(np.abs(signal)))
    
    # 7. DAMV (Difference Absolute Mean Value)
    diff = np.abs(np.diff(signal))
    features['DAMV'] = float(np.mean(diff)) if len(diff) > 0 else 0.0
    
    return features


# ============================================================================
# SECCIÓN 3: CARACTERÍSTICAS EN DOMINIO DE LA FRECUENCIA
# ============================================================================

def compute_frequency_domain_features(signal: np.ndarray, fs: int, nperseg: int = 1024) -> Dict[str, float]:
    """
    Calcula características en el dominio de la frecuencia (PSD - Welch).
    
    Características:
        - MNF: Mean Frequency (Frecuencia media ponderada)
        - MDF: Median Frequency (Frecuencia que divide el espectro en 2 mitades)
        - Peak Freq: Frecuencia dominante (máxima potencia)
        - Total Power: Potencia total integrada
    
    Args:
        signal: Señal EMG
        fs: Frecuencia de muestreo
        nperseg: Puntos por segmento Welch
    
    Returns:
        Diccionario con características espectrales
    """
    features = {}
    
    f, Pxx = welch(signal, fs, nperseg=min(nperseg, len(signal)))
    
    # 1. MNF (Mean Frequency)
    total_power = np.sum(Pxx)
    features['MNF'] = float(np.sum(f * Pxx) / total_power) if total_power > 0 else 0.0
    
    # 2. MDF (Median Frequency)
    cumulative_power = np.cumsum(Pxx)
    idx_median = np.where(cumulative_power >= total_power / 2)[0]
    if len(idx_median) > 0:
        features['MDF'] = float(f[idx_median[0]])
    else:
        features['MDF'] = float(f[-1])
    
    # 3. Peak Frequency
    idx_peak = np.argmax(Pxx)
    features['Peak_Freq'] = float(f[idx_peak])
    
    # 4. Total Power
    features['Total_Power'] = float(total_power)
    
    # 5. Power Distribution (% potencia < 100 Hz)
    idx_100 = np.where(f <= 100)[0]
    if len(idx_100) > 0:
        power_low = np.sum(Pxx[idx_100])
        features['Power_Below_100Hz'] = float((power_low / total_power) * 100) if total_power > 0 else 0.0
    else:
        features['Power_Below_100Hz'] = 0.0
    
    return features


# ============================================================================
# SECCIÓN 4: ANÁLISIS DE FATIGA
# ============================================================================

def detect_fatigue(segments: List[np.ndarray], fs: int, threshold: float = 0.9) -> Dict[str, any]:
    """
    Detecta fatiga muscular comparando MDF al inicio vs. final.
    
    Criterio: Si MDF_final < MDF_inicial * threshold -> Fatiga
    
    Args:
        segments: Lista de segmentos de señal (ej. inicio, medio, final)
        fs: Frecuencia de muestreo
        threshold: Umbral de fatiga (0.9 = 90% de MDF inicial)
    
    Returns:
        Diccionario con análisis de fatiga
    """
    results = {}
    mdfs = []
    
    for i, seg in enumerate(segments):
        f, Pxx = welch(seg, fs, nperseg=min(1024, len(seg)))
        cumsum = np.cumsum(Pxx)
        total = np.sum(Pxx)
        idx = np.where(cumsum >= total / 2)[0]
        if len(idx) > 0:
            mdf = f[idx[0]]
        else:
            mdf = f[-1]
        mdfs.append(mdf)
        results[f'MDF_Segment_{i}'] = float(mdf)
    
    # Análisis de tendencia
    if len(mdfs) >= 2:
        mdf_drop_percent = ((mdfs[0] - mdfs[-1]) / mdfs[0]) * 100 if mdfs[0] > 0 else 0
        results['MDF_Drop_%'] = float(mdf_drop_percent)
        results['Fatigue_Detected'] = mdf_drop_percent > ((1 - threshold) * 100)
    else:
        results['Fatigue_Detected'] = False
    
    return results


# ============================================================================
# SECCIÓN 5: ANÁLISIS DE SINCRONIZACIÓN (CROSS-CORRELATION)
# ============================================================================

def calculate_delay(signal1: np.ndarray, signal2: np.ndarray, fs: int) -> Dict[str, float]:
    """
    Calcula retraso (lag) entre dos señales usando correlación cruzada.
    
    Args:
        signal1: Primera señal
        signal2: Segunda señal
        fs: Frecuencia de muestreo
    
    Returns:
        Diccionario con retraso en ms
    """
    corr = correlate(signal1, signal2, mode='full')
    lag_samples = np.argmax(corr) - (len(signal1) - 1)
    lag_seconds = lag_samples / fs
    lag_ms = lag_seconds * 1000
    
    return {
        'Lag_Samples': int(lag_samples),
        'Lag_Seconds': float(lag_seconds),
        'Lag_Ms': float(lag_ms)
    }


# ============================================================================
# SECCIÓN 6: FUNCIONES PRINCIPALES DE ANÁLISIS
# ============================================================================

def analyze_emg_signal(
    signal: np.ndarray,
    time: np.ndarray,
    fs: int = DEFAULT_FS,
    lowcut: float = CUTOFF_LOW,
    highcut: float = CUTOFF_HIGH,
    envelope_cutoff: float = ENVELOPE_CUTOFF
) -> Dict[str, any]:
    """
    Realiza análisis completo de una señal EMG.
    
    Args:
        signal: Señal EMG cruda
        time: Vector de tiempo
        fs: Frecuencia de muestreo
        lowcut: Frecuencia de corte baja
        highcut: Frecuencia de corte alta
        envelope_cutoff: Frecuencia de corte para envolvente
    
    Returns:
        Diccionario con todos los análisis
    """
    results = {
        'metadata': {
            'duration_s': float(time[-1] - time[0]),
            'num_samples': len(signal),
            'fs': fs,
            'lowcut': lowcut,
            'highcut': highcut
        },
        'raw_signal': {
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'min': float(np.min(signal)),
            'max': float(np.max(signal))
        }
    }
    
    # Procesamiento
    signal_dc = remove_dc_offset(signal)
    signal_filtered = butter_bandpass_filter(signal_dc, lowcut, highcut, fs)
    signal_envelope = get_envelope(signal_filtered, envelope_cutoff, fs)
    
    # Normalización
    mvc_value = np.max(signal_envelope)
    if mvc_value == 0:
        mvc_value = 1
    signal_normalized = signal_envelope / mvc_value
    
    # Características
    results['time_domain'] = compute_time_domain_features(signal_filtered, fs)
    results['frequency_domain'] = compute_frequency_domain_features(signal_filtered, fs)
    
    # Envolvente
    results['envelope'] = {
        'max': float(np.max(signal_envelope)),
        'mean': float(np.mean(signal_envelope)),
        'std': float(np.std(signal_envelope)),
        'mvc_value': float(mvc_value)
    }
    
    # Fatiga (dividir en 3 partes)
    n = len(signal_filtered)
    seg1 = signal_filtered[0:n//3]
    seg2 = signal_filtered[n//3:2*n//3]
    seg3 = signal_filtered[2*n//3:]
    results['fatigue'] = detect_fatigue([seg1, seg2, seg3], fs)
    
    # Procesadas
    results['processed_signals'] = {
        'filtered': signal_filtered,
        'envelope': signal_envelope,
        'normalized': signal_normalized
    }
    
    return results


def load_csv(file_path: str, fs: int = DEFAULT_FS) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Carga datos de archivo CSV de Noraxon.
    
    Asume formato: sep=';', decimal=',', encabezado en línea 3.
    
    Args:
        file_path: Ruta al archivo CSV
        fs: Frecuencia de muestreo esperada
    
    Returns:
        (signal, time, column_name)
    """
    df = pd.read_csv(file_path, sep=';', decimal=',', header=2)
    
    # Buscar columna EMG (contiene "uV")
    emg_cols = [col for col in df.columns if '(uV)' in col]
    if not emg_cols:
        raise ValueError(f"No se encontró columna de EMG (uV) en {file_path}")
    
    emg_col = emg_cols[0]
    time_col = 'time'
    
    if time_col not in df.columns:
        raise ValueError(f"No se encontró columna 'time' en {file_path}")
    
    signal = df[emg_col].values
    time = df[time_col].values
    
    return signal, time, emg_col


def plot_analysis(results: Dict, time: np.ndarray, title: str = "Análisis EMG", save_path: Optional[str] = None):
    """
    Visualiza el análisis completo con 6 subplots.
    
    Args:
        results: Diccionario de resultados del análisis
        time: Vector de tiempo
        title: Título general
        save_path: Ruta para guardar figura (opcional)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    raw = results['raw_signal']
    signals = results['processed_signals']
    time_feat = results['time_domain']
    freq_feat = results['frequency_domain']
    fatigue = results['fatigue']
    
    # 1. Señal Cruda
    ax = axes[0, 0]
    ax.plot(time, np.arange(len(time)), color='gray', alpha=0.6)
    ax.plot(time, signals['filtered'], color='blue', label='Filtrada', linewidth=0.8)
    ax.set_title('Señal Cruda y Filtrada (20-450 Hz)')
    ax.set_ylabel('Amplitud (µV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Envolvente y Normalizada
    ax = axes[0, 1]
    ax.plot(time, signals['envelope'], color='green', label='Envolvente', linewidth=1.2)
    ax.plot(time, signals['normalized'], color='red', label='Normalizada (0-1)', linewidth=1.2)
    ax.set_title('Envolvente y Señal Normalizada')
    ax.set_ylabel('Amplitud')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Características de Tiempo
    ax = axes[1, 0]
    time_keys = ['RMS', 'MAV', 'Peak', 'WL']
    time_vals = [time_feat.get(k, 0) for k in time_keys]
    ax.bar(time_keys, time_vals, color=['blue', 'green', 'red', 'orange'])
    ax.set_title('Características Dominio del Tiempo')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Características de Frecuencia
    ax = axes[1, 1]
    freq_keys = ['MNF', 'MDF', 'Peak_Freq']
    freq_vals = [freq_feat.get(k, 0) for k in freq_keys]
    ax.bar(freq_keys, freq_vals, color=['purple', 'brown', 'pink'])
    ax.set_title('Características Dominio de la Frecuencia')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Espectro Welch
    ax = axes[2, 0]
    f, Pxx = welch(signals['filtered'], fs=2000, nperseg=1024)
    ax.semilogy(f, Pxx)
    ax.set_title('Espectro de Potencia (Welch)')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Potencia')
    ax.grid(True, alpha=0.3)
    
    # 6. Análisis de Fatiga (MDF por segmento)
    ax = axes[2, 1]
    mdf_vals = [fatigue.get(f'MDF_Segment_{i}', 0) for i in range(3)]
    seg_names = ['Inicio', 'Medio', 'Final']
    ax.plot(seg_names, mdf_vals, 'o-', color='darkred', linewidth=2, markersize=8)
    ax.set_title(f"Análisis de Fatiga (MDF Drop: {fatigue.get('MDF_Drop_%', 0):.1f}%)")
    ax.set_ylabel('MDF (Hz)')
    ax.grid(True, alpha=0.3)
    
    if fatigue.get('Fatigue_Detected', False):
        ax.text(1, max(mdf_vals) * 0.9, '⚠ FATIGA', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def save_features_to_csv(results: Dict, output_path: str):
    """
    Guarda todas las características extraídas en un archivo CSV.
    
    Args:
        results: Diccionario de resultados
        output_path: Ruta del archivo de salida
    """
    features_flat = {}
    
    # Aplanar el diccionario anidado
    def flatten(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            elif not isinstance(v, np.ndarray):
                features_flat[new_key] = v
    
    flatten(results)
    
    # Guardar como CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Value'])
        for k, v in features_flat.items():
            if not isinstance(v, bool):
                writer.writerow([k, v])
            else:
                writer.writerow([k, str(v)])
    
    print(f"✓ Características guardadas en: {output_path}")


def print_summary(results: Dict, file_name: str):
    """
    Imprime resumen de análisis en consola.
    
    Args:
        results: Diccionario de resultados
        file_name: Nombre del archivo analizado
    """
    print("\n" + "="*70)
    print(f"ANÁLISIS EMG: {file_name}")
    print("="*70)
    
    meta = results['metadata']
    print(f"\n METADATOS")
    print(f"  Duración: {meta['duration_s']:.2f} s")
    print(f"  Muestras: {meta['num_samples']}")
    print(f"  Fs: {meta['fs']} Hz")
    print(f"  Banda: {meta['lowcut']}-{meta['highcut']} Hz\n")
    
    raw = results['raw_signal']
    print(f" SEÑAL CRUDA")
    print(f"  Media: {raw['mean']:.2f} µV")
    print(f"  Desv. Est.: {raw['std']:.2f} µV")
    print(f"  Min/Max: {raw['min']:.2f} / {raw['max']:.2f} µV\n")
    
    time_feat = results['time_domain']
    print(f" CARACTERÍSTICAS TIEMPO")
    print(f"  RMS: {time_feat['RMS']:.3f}")
    print(f"  MAV: {time_feat['MAV']:.3f}")
    print(f"  Peak: {time_feat['Peak']:.3f}")
    print(f"  WL: {time_feat['WL']:.1f}")
    print(f"  ZC: {time_feat['ZC']:.0f}\n")
    
    freq_feat = results['frequency_domain']
    print(f" CARACTERÍSTICAS FRECUENCIA")
    print(f"  MNF: {freq_feat['MNF']:.2f} Hz")
    print(f"  MDF: {freq_feat['MDF']:.2f} Hz")
    print(f"  Peak Freq: {freq_feat['Peak_Freq']:.2f} Hz")
    print(f"  Total Power: {freq_feat['Total_Power']:.2e}\n")
    
    env = results['envelope']
    print(f" ENVOLVENTE")
    print(f"  Max: {env['max']:.3f}")
    print(f"  Media: {env['mean']:.3f}")
    print(f"  MVC: {env['mvc_value']:.3f}\n")
    
    fatigue = results['fatigue']
    print(f" FATIGA MUSCULAR")
    print(f"  MDF Inicio: {fatigue.get('MDF_Segment_0', 0):.2f} Hz")
    print(f"  MDF Medio: {fatigue.get('MDF_Segment_1', 0):.2f} Hz")
    print(f"  MDF Final: {fatigue.get('MDF_Segment_2', 0):.2f} Hz")
    print(f"  Caída MDF: {fatigue.get('MDF_Drop_%', 0):.1f}%")
    if fatigue.get('Fatigue_Detected', False):
        print(f"   FATIGA DETECTADA")
    else:
        print(f"   Músculo fresco")
    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Análisis completo de señales EMG desde archivos CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python EMG_CSV.py "archivo.csv"
  python EMG_CSV.py "archivo.csv" --plot --save-features
  python EMG_CSV.py "archivo.csv" --output ./results/
        """
    )
    
    parser.add_argument('csv_file', help='Ruta al archivo CSV')
    parser.add_argument('--fs', type=int, default=DEFAULT_FS, help='Frecuencia de muestreo (Hz)')
    parser.add_argument('--output', default=None, help='Directorio de salida para resultados')
    parser.add_argument('--plot', action='store_true', help='Mostrar gráficos')
    parser.add_argument('--save-features', action='store_true', help='Guardar características en CSV')
    parser.add_argument('--lowcut', type=float, default=CUTOFF_LOW, help='Frecuencia de corte baja (Hz)')
    parser.add_argument('--highcut', type=float, default=CUTOFF_HIGH, help='Frecuencia de corte alta (Hz)')
    
    args = parser.parse_args()
    
    # Validar archivo
    if not os.path.exists(args.csv_file):
        print(f"Error: Archivo no encontrado: {args.csv_file}")
        return
    
    # Crear directorio de salida
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    try:
        # Cargar datos
        print(f"Cargando: {args.csv_file}")
        signal, time, col_name = load_csv(args.csv_file, fs=args.fs)
        print(f"✓ Cargadas {len(signal)} muestras en {len(time)} puntos de tiempo")
        
        # Análisis
        print("Analizando señal...")
        results = analyze_emg_signal(
            signal, time, fs=args.fs,
            lowcut=args.lowcut, highcut=args.highcut
        )
        print("Análisis completado")
        
        # Resumen
        file_name = Path(args.csv_file).stem
        print_summary(results, file_name)
        
        # Guardar características
        if args.save_features and args.output:
            csv_out = os.path.join(args.output, f"{file_name}_features.csv")
            save_features_to_csv(results, csv_out)
        
        # Plots
        if args.plot:
            plot_title = f"Análisis EMG: {col_name}"
            plot_path = None
            if args.output:
                plot_path = os.path.join(args.output, f"{file_name}_analysis.png")
            plot_analysis(results, time, title=plot_title, save_path=plot_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
