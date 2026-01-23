"""
Código de Calibración MVC para EMG de Noraxon Ultium.
Autor: Alejandro Solar Iglesias
Fecha: Diciembre 2025
Versión: 1.0
"""

"""
GUÍA COMPLETA PARA CALIBRACIÓN MVC (Maximum Voluntary Contraction)
===================================================================

PARTE 1: PROTOCOLO FÍSICO DE CALIBRACIÓN
=========================================

Basado en las mejores prácticas científicas publicadas en:

1. **MDPI Sensors 2025 (Fuentes del Toro et al.)**: "The Impact of Normalization 
   Procedures on Surface Electromyography (sEMG) Data Integrity"
   - Recomienda MVC como gold standard para normalización
   - Contracciones máximas isométricas de 5 segundos
   - Repetir 3 veces con 2 minutos de descanso entre repeticiones
   
2. **MDPI Signals 2024 (Zhao et al.)**: "Standardizing EMG Pipelines for Muscle 
   Synergy Analysis"
   - Recomienda filtro pasa-banda 20-450 Hz (estándar SENIAM)
   - Envolvente lineal con filtro pasa-baja 6-10 Hz
   - Método del percentil 95 es más robusto que el máximo absoluto

3. **Delsys Tutorial - The Use of sEMG in Biomechanics**:
   - Contracciones isométricas en posiciones estandarizadas
   - Mantener contracción 3-5 segundos
   - Evitar movimientos compensatorios

══════════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════════

PARTE 2: PROCESAMIENTO Y CÁLCULO DE MVC
========================================

**METODOLOGÍA DE ANÁLISIS** (basada en papers científicos):

1. **Pre-procesamiento**:
   - Filtro pasa-banda: 20-450 Hz (Butterworth 4º orden)
   - Remoción de offset DC
   - Rectificación de onda completa (valor absoluto)

2. **Cálculo de envolvente lineal**:
   - Filtro pasa-baja: 6 Hz (Butterworth 4º orden)
   - Preserva la dinámica de activación muscular

3. **Selección de ventana de análisis**:
    CRÍTICO: No usar toda la grabación
   
   Según los papers:
   - Identificar la fase de meseta (segundos 1-4 de cada contracción)
   - Usar ventana deslizante de 500 ms (1000 muestras a 2000 Hz)
   - Extraer el MVC de esta ventana específica

4. **Métodos de estimación MVC** (3 opciones):

   a) **Máximo absoluto** (peak):
      - MVC = max(envolvente_en_ventana)
      - Simple pero sensible a artefactos/ruido
      - Útil si señal muy limpia
   
   b) **Percentil 95** (95perc) -  RECOMENDADO:
      - MVC = percentile(envolvente_en_ventana, 95)
      - Robusto ante outliers y picos espurios
      - Método estándar en literatura actual
      - Referencia: Zhao et al., Signals 2024
   
   c) **Media del 10% superior** (mean_top10):
      - MVC = mean(top_10_percent(envolvente_en_ventana))
      - Balance entre robustez y representatividad
      - Menos sensible que peak, más estable que percentil

5. **Selección del mejor MVC**:
   - Si múltiples repeticiones: tomar el MÁXIMO entre todas
   - Verificar consistencia (<10% variación entre repeticiones)

══════════════════════════════════════════════════════════════════════════

REFERENCIAS CIENTÍFICAS
========================

1. Fuentes del Toro S, Aranda-Ruiz J. (2025). "The Impact of Normalization 
   Procedures on Surface Electromyography (sEMG) Data Integrity". 
   Sensors, 25(9), 2668. https://doi.org/10.3390/s25092668
   
   → Recomienda: MVC con contracciones isométricas de 5 seg, 3 repeticiones
   → Ventana de análisis: fase de meseta (segundos 1-4)
   → Filtrado: 20-450 Hz pasa-banda, envolvente 6 Hz

2. Zhao K, Jin Y, Feng Y, Li J, Zhou Y. (2024). "Standardizing EMG Pipelines 
   for Muscle Synergy Analysis". Signals, 6(4), 68.
   https://doi.org/10.3390/signals6040068
   
   → Recomienda: Percentil 95 más robusto que máximo
   → Procesamiento: bandpass 20-450 Hz, lowpass 6-10 Hz
   → Normalización: MAX, AVE o UVA con EVAF criterion

3. Hermens HJ, et al. (2000). "European Recommendations for Surface 
   ElectroMyoGraphy (SENIAM)". 
   http://www.seniam.org
   
   → Estándar para colocación de electrodos
   → Metodología MVC estandarizada
   → Posiciones anatómicas específicas por músculo

4. Delsys Inc. "The Use of Surface EMG in Biomechanics" Tutorial.
   https://delsys.com/downloads/TUTORIAL/the-use-of-semg-in-biomechanics.pdf
   
   → Protocolos de prueba MVC
   → Criterios de validación
   → Troubleshooting común

══════════════════════════════════════════════════════════════════════════
"""

# Importar librería personalizada
from noraxon_analytics import *
import matplotlib.pyplot as plt
import numpy as np

# 1. CONFIGURACIÓN
filename = r'C:\Users\alexs\Desktop\EMG_NORAVOX\Movimientos\2025-12-15-16-50_AntFManoD_MD.csv'
FS = 2000  # Frecuencia de muestreo típica de Noraxon
print(f" INICIANDO PROTOCOLO PARA: {filename}...\n")

try:
    # ---------------------------------------------------------
    # FASE 1: CARGA DE DATOS
    # ---------------------------------------------------------
    # La función load_csv ya maneja el formato de Noraxon
    raw_emg, time, emg_col_name = load_csv(filename, fs=FS)
    print(f"1. CARGA: [OK] Músculo detectado: {emg_col_name}")

    # ---------------------------------------------------------
    # FASE 2: PRE-PROCESAMIENTO (Pipeline Científico)
    # ---------------------------------------------------------
    # A) Quitar Offset (Bias)
    emg_centered = remove_dc_offset(raw_emg)

    # B) Chequeo de Calidad (SNR)
    qa = calculate_signal_quality_snr(emg_centered)
    print(f"   |__ Calidad: {qa['Status']} (Ruido: {qa['Noise_Floor_uV']:.2f} uV)")
    
    if qa['Noise_Floor_uV'] > 15:
        print("   ADVERTENCIA: Ruido basal alto. Revisar impedancia.")

    # C) Filtrado Pasa-Banda (20-450 Hz)
    emg_filtered = butter_bandpass_filter(emg_centered, 20, 450, FS)

    # D) Envolvente Lineal (6 Hz)
    emg_envelope = compute_linear_envelope(emg_filtered, FS, cutoff=6)
    
    print("2. PROCESAMIENTO: [OK] Filtros SENIAM + Envolvente aplicados.")

    # ---------------------------------------------------------
    # FASE 3: CÁLCULO DE MVC (Método Robusto)
    # ---------------------------------------------------------
    # Aquí usamos tu nueva función que aplica Ventana Móvil + Percentil 95
    mvc_value = calculate_mvc_value(emg_envelope, fs=FS, window_ms=500, method="peak")
    
    print("-" * 50)
    print(f" RESULTADO MVC FINAL: {mvc_value:.2f} uV")
    print("-" * 50)

    # ---------------------------------------------------------
    # FASE 4: VISUALIZACIÓN DE CONTROL
    # ---------------------------------------------------------
    print("\n[VISUALIZACIÓN] Generando gráficos de auditoría de ventana...")
    
    visualize_mvc_sliding_window(emg_envelope, time, fs=FS, window_ms=500)
    
    print("   -> Verifica en el gráfico que la zona VERDE esté en la meseta estable.")

except Exception as e:
    print(f"\n ERROR CRÍTICO: {e}")
    # Esto ayuda a ver dónde falló exactamente
    import traceback
    traceback.print_exc()