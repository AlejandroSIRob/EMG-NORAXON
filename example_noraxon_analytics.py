"""
Script de Análisis EMG Completo (Alineado con librería noraxon_analytics v1.0)
Autor: Alejandro Solar Iglesias
Fecha: Diciembre 2025
Versión: 1.0
"""

# Importar librería personalizada (asegúrate de que noraxon_analytics.py esté en la misma carpeta)
try:
    from noraxon_analytics import *
except ImportError:
    print(" ERROR: No se encuentra 'noraxon_analytics.py'.")
    exit()

import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURACIÓN
# Usamos r'' para rutas de Windows para evitar errores de caracteres
filename = r'C:\Users\alexs\Desktop\EMG_NORAVOX\Pruebas\2025-12-10-10-38_Calibracion-Biceps-20251210.csv'
# Si el archivo no existe en esa ruta exacta, intenta buscarlo en la carpeta actual
if not os.path.exists(filename):
    filename = "2025-12-10-10-38_Calibracion-Biceps-20251210.csv"

print(f"\n=== INICIANDO PROTOCOLO PARA: {filename} ===\n")

try:
    # ---------------------------------------------------------
    # FASE 1: CARGA Y CONTROL DE CALIDAD (QA)
    # ---------------------------------------------------------
    # Cargar datos usando la función de la librería
    raw_emg, time, emg_col_name = load_csv(filename, fs=DEFAULT_FS)
    
    # Eliminar DC Offset (Vital para cálculos de amplitud) [REF: ABC of EMG p.23]
    emg_centered = remove_dc_offset(raw_emg)

    # CHEQUEO DE CALIDAD (SNR) - Antes de procesar nada
    qa_metrics = calculate_signal_quality_snr(emg_centered)
    print(f"1. CONTROL DE CALIDAD:")
    print(f"   Estado: {qa_metrics['Status']}")
    print(f"   Ruido Basal: {qa_metrics['Noise_Floor_uV']:.2f} uV | SNR: {qa_metrics['SNR_dB']:.1f} dB")
    
    if qa_metrics['Noise_Floor_uV'] > 15:
        print("    ADVERTENCIA: Ruido alto. Verificar impedancia de electrodos.")
    print("-" * 60)

    # ---------------------------------------------------------
    # FASE 2: PRE-PROCESAMIENTO DE SEÑAL
    # ---------------------------------------------------------
    # Filtrado Pasa-Banda (20-450 Hz) [REF: SENIAM / Noraxon]
    filtered_emg = butter_bandpass_filter(emg_centered, CUTOFF_LOW, CUTOFF_HIGH, DEFAULT_FS)
    
    # Envolvente Lineal (6 Hz) [REF: ABC of EMG p.29]
    envelope = compute_linear_envelope(filtered_emg, DEFAULT_FS, cutoff=ENVELOPE_CUTOFF)
    print("2. PRE-PROCESAMIENTO: [OK] (Filtros SENIAM y Envolvente aplicados)")

    # ---------------------------------------------------------
    # FASE 3: NORMALIZACIÓN DE AMPLITUD (% MVC)
    # ---------------------------------------------------------
    # Estimación de MVC usando el método ROBUSTO (Ventana 500ms + Percentil 95)
    # [REF: ABC of EMG p.32 y Zhao et al. 2024]
    mvc_ref = calculate_mvc_value(envelope, DEFAULT_FS, window_ms=500, method="95perc")

    print("\n[VISUALIZACIÓN] Generando auditoría de la ventana MVC...")
    visualize_mvc_sliding_window(envelope, time, fs=DEFAULT_FS, window_ms=500)
    
    # Normalización
    emg_normalized = normalize_to_mvc(envelope, mvc_ref)
    print(f"3. NORMALIZACIÓN:     [OK] (Ref. MVC estimada [95%]: {mvc_ref:.2f} uV)")

    # ---------------------------------------------------------
    # FASE 4: EXTRACCIÓN DE CARACTERÍSTICAS (FEATURES)
    # ---------------------------------------------------------
    # A) Globales (Todo el archivo)
    time_feats = compute_time_domain_features_oleinikov(filtered_emg, DEFAULT_FS)
    freq_feats = compute_frequency_domain_features_oleinikov(filtered_emg, DEFAULT_FS)
    
    # B) Continuas (Ventanas deslizantes para ver evolución temporal)
    print("4. ANÁLISIS TEMPORAL: [OK] Generando ventanas deslizantes (200ms)...")
    df_continuous = extract_features_continuous(filtered_emg, DEFAULT_FS, window_ms=200, step_ms=50)

    # ---------------------------------------------------------
    # FASE 5: ANÁLISIS DE FATIGA (SPECTRAL SHIFT)
    # ---------------------------------------------------------
    # Detectar caída de la Frecuencia Mediana [REF: ABC of EMG p.51]
    # Importante: Usar señal filtrada, NO la envolvente
    fatigue_data = analyze_muscle_fatigue(filtered_emg, DEFAULT_FS)
    print(f"5. ANÁLISIS FATIGA:   [OK] Tendencia MDF: {fatigue_data['Fatigue_Slope']:.4f} Hz/s")

    # =========================================================
    # REPORTE FINAL DE RESULTADOS
    # =========================================================
    print("\n" + "="*60)
    print(f" REPORTE BIOMECÁNICO: {emg_col_name}")
    print("="*60)
    
    print("\n[A] INTENSIDAD Y ESFUERZO")
    print(f"    Pico de Fuerza (% MVC): {np.max(emg_normalized):.1f} %")
    print(f"    Potencia Media (RMS):   {time_feats['RMS']:.2f} uV")
    print(f"    Energía Total (IEMG):   {time_feats['IEMG']:.2f} uV*s")

    print("\n[B] CARACTERÍSTICAS ESPECTRALES (Oleinikov)")
    print(f"    Frec. Media (MNF):      {freq_feats['FREQ_MNF']:.2f} Hz")
    print(f"    Frec. Pico (PKF):       {freq_feats['FREQ_Peak_Hz']:.2f} Hz")
    if 'WAVE_db4_L4_Energy' in freq_feats:
        print(f"    Wavelet Energy (db4):   {freq_feats['WAVE_db4_L4_Energy']:.2f}")

    print("\n[C] ESTADO DE FATIGA MUSCULAR")
    if fatigue_data['Is_Fatigued']:
        print(f"     FATIGA DETECTADA. La MDF cae a razón de {fatigue_data['Fatigue_Slope']:.3f} Hz/s")
    else:
        print(f"     SIN FATIGA EVIDENTE. Espectro estable.")

    print("="*60)

    # =========================================================
    # VISUALIZACIÓN GRÁFICA PROFESIONAL
    # =========================================================
    print("\nGenerando paneles gráficos...")

    # PANEL 1: Calidad y Procesamiento (Cruda vs Filtrada)
    plot_signals(
        time, 
        [emg_centered, filtered_emg], 
        labels=["EMG Cruda (Centrada)", "EMG Filtrada (20-450Hz)"],
        title="1. Procesamiento de Señal (Reducción de Ruido)",
        ylabel="Amplitud (uV)",
        figsize=(12, 6),
        subplots=True
    )

    # PANEL 2: Activación Muscular Normalizada (% MVC) - Clave Clínica
    plt.figure(figsize=(12, 5))
    plt.plot(time, emg_normalized, color='tab:orange', label='Activación (% MVC)')
    plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Límite 100% MVC')
    plt.title(f"2. Esfuerzo Muscular Normalizado (Ref: {mvc_ref:.0f} uV)")
    plt.ylabel("% Contracción Máxima")
    plt.xlabel("Tiempo (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # PANEL 3: Fisiología (Fatiga y Potencia)
    if not df_continuous.empty:
        # Creamos una figura manual para tener ejes dobles (Potencia vs Frecuencia)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Eje Izquierdo: RMS (Potencia)
        color = 'tab:blue'
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Potencia RMS (uV)', color=color)
        ax1.plot(df_continuous['Time_s'], df_continuous['RMS'], color=color, alpha=0.6, label='RMS Power')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Eje Derecho: MDF (Fatiga) -> Extraemos MDF de la función de fatiga
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Frecuencia Mediana (Hz)', color=color)
        ax2.plot(fatigue_data['Time_Points'], fatigue_data['MDF_Values'], color=color, linewidth=2, label='MDF (Fatiga)')
        
        # Línea de tendencia de fatiga
        trend = fatigue_data['Fatigue_Slope'] * fatigue_data['Time_Points'] + fatigue_data['Initial_MDF']
        ax2.plot(fatigue_data['Time_Points'], trend, color='black', linestyle='--', alpha=0.7, label='Tendencia Fatiga')
        
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("3. Relación Potencia vs. Fatiga (Spectral Shift)", fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.show()
    # PANEL 4: VISUALIZACIÓN TÉCNICA (RMS vs SEÑAL FILTRADA)
    # ======================================================
    plt.figure(figsize=(12, 6))
    
    # 1. Graficar la señal de fondo (Filtrada)
    # Usamos alpha=0.3 para que sea semitransparente y gris
    plt.plot(time, filtered_emg, color='gray', alpha=0.3, label='EMG Filtrada (Raw)')

    # 2. Graficar la RMS encima
    # Nota: Usamos df_continuous['Time_s'] porque la RMS tiene menos puntos que la raw
    plt.plot(df_continuous['Time_s'], df_continuous['RMS'], 
             color='red', linewidth=2, label='RMS (Potencia Media)')

    plt.title("4. Análisis de Activación: Señal Cruda vs. Potencia RMS")
    plt.ylabel("Amplitud (uV)")
    plt.xlabel("Tiempo (s)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n ERROR CRÍTICO DURANTE EL ANÁLISIS: {e}")
    import traceback 
    traceback.print_exc()