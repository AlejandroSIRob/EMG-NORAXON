"""
Script de Análisis de Tarea: Montar/Desmontar Cabeza de Muñeca
Autor: Alejandro Solar Iglesias
Fecha: Diciembre 2025
Versión: 1.0
"""

from noraxon_analytics import *
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO
# ==============================================================================
# Nombre exacto de tu archivo de tarea
# Usamos r'' para que Windows lea bien las barras invertidas
FILENAME = r"C:\Users\alexs\Desktop\EMG_NORAVOX\Movimientos\2025-12-10-14-19_MontarDesmontarCabezaMuñeca-AntebrazoFlexor.csv"

# VALOR DE REFERENCIA (MVC) - OBTENIDO DE TU CALIBRACIÓN PREVIA
# Flexor Carpi Radialis: 392.48 uV

# VALOR DE REFERENCIA (MVC) - Flexor Carpi Radialis
MVC_REF_UWORD = 392.48 
FS = 2000 

print(f"\n{'='*60}")
print(f" ANÁLISIS DE TAREA COMPLETO (AMPLITUD + FATIGA)")
print(f"{'='*60}\n")

try:
    # ---------------------------------------------------------
    # FASE 1: CARGA Y LIMPIEZA
    # ---------------------------------------------------------
    raw_emg, time, col_name = load_csv(FILENAME, fs=FS)
    
    # A. Señal para Fatiga (Filtrada pero NO rectificada/envolvente)
    # Rango 20-450 Hz estándar SENIAM para espectro
    emg_centered = remove_dc_offset(raw_emg)
    filtered_emg = butter_bandpass_filter(emg_centered, 20, 450, FS)
    
    # B. Señal para Amplitud (Envolvente Lineal)
    # Cutoff 6Hz para simular la biomecánica del movimiento
    envelope = compute_linear_envelope(filtered_emg, FS, cutoff=6)
    
    print(f"1. PROCESAMIENTO: [OK] Señales generadas (Espectral y Amplitud).")

    # ---------------------------------------------------------
    # FASE 2: NORMALIZACIÓN REAL (% MVC)
    # ---------------------------------------------------------
    norm_emg = (envelope / MVC_REF_UWORD) * 100
    peak_task = np.max(norm_emg)
    mean_task = np.mean(norm_emg)
    iemg = np.sum(np.abs(filtered_emg)) / FS 

    # ---------------------------------------------------------
    # FASE 3: ANÁLISIS DE FATIGA (MDF)
    # ---------------------------------------------------------
    # ¡OJO! Pasamos 'filtered_emg', NO 'envelope' ni 'norm_emg'
    print("   > Calculando fatiga espectral (puede tardar unos segundos)...")
    fatigue_data = analyze_muscle_fatigue(filtered_emg, FS, window_sec=1.0, step_sec=0.5)
    
    slope = fatigue_data["Fatigue_Slope"]
    is_fatigued = fatigue_data["Is_Fatigued"]
    
    print(f"2. CÁLCULOS: [OK] MVC y Fatiga procesados.")
    
    # ---------------------------------------------------------
    # FASE 4: REPORTE DE RESULTADOS
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(f" REPORTE BIOMECÁNICO: {col_name}")
    print("-"*40)
    print(" >> DOMINIO DEL TIEMPO (Esfuerzo):")
    print(f"    - Pico de Esfuerzo:   {peak_task:.2f} % MVC")
    print(f"    - Esfuerzo Medio:     {mean_task:.2f} % MVC")
    print(f"    - Energía Total:      {iemg:.2f} uV*s")
    
    print("\n >> DOMINIO DE LA FRECUENCIA (Fatiga):")
    print(f"    - Pendiente MDF:      {slope:.4f} Hz/s")
    print(f"    - R² (Consistencia):  {fatigue_data['R_Squared']:.2f}")
    
    # Interpretación Experta
    print("\n [CONCLUSIÓN CLÍNICA]:")
    if is_fatigued:
        print(" [ALERTA] SIGNOS DE FATIGA METABÓLICA.")
        print(" La frecuencia mediana está cayendo, indicando reducción en la velocidad")
        print(" de conducción muscular, posiblemente por acumulación de ácido láctico.")
    else:
        print(" [OK] ESTADO MUSCULAR ESTABLE.")
        print(" No hay caída significativa en la frecuencia mediana.")

    if peak_task > 40 and is_fatigued:
        print(" RECOMENDACIÓN: Aumentar tiempos de descanso entre ciclos.")

    # ---------------------------------------------------------
    # FASE 5: GRÁFICOS (3 PANELES) - VERSIÓN CORREGIDA
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 10))
    
    # Panel 1: Señal Cruda (Calidad)
    plt.subplot(3, 1, 1)
    plt.plot(time, filtered_emg, color='gray', alpha=0.5, label='EMG Filtrada (20-450Hz)')
    plt.title(f"1. Calidad de Señal Raw: {col_name}")
    plt.ylabel("uV")
    plt.grid(True, alpha=0.3)
    
    # Panel 2: Esfuerzo Normalizado (% MVC)
    plt.subplot(3, 1, 2)
    plt.plot(time, norm_emg, color='tab:blue', linewidth=1.5, label='Esfuerzo')
    plt.axhline(peak_task, color='red', linestyle=':', label=f'Pico {peak_task:.1f}%')
    plt.axhline(mean_task, color='green', linestyle='--', label=f'Media {mean_task:.1f}%')
    plt.fill_between(time, 0, 30, color='green', alpha=0.1, label='Zona Confort')
    plt.title("2. Intensidad de Esfuerzo (% MVC)")
    plt.ylabel("% MVC")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Panel 3: Análisis de Fatiga 
    plt.subplot(3, 1, 3)
    if len(fatigue_data["MDF_Values"]) > 0:
        # Puntos MDF
        plt.scatter(fatigue_data["Time_Points"], fatigue_data["MDF_Values"], 
                   color='purple', alpha=0.5, s=20, label='MDF (Ventana 1s)')
        
        # --- CORRECCIÓN DE INGENIERÍA ---
        # Si el diccionario no trae 'Intercept', lo calculamos matemáticamente:
        # y = mx + b  ->  b = media(y) - m * media(x)
        if "Intercept" in fatigue_data:
            intercept_val = fatigue_data["Intercept"]
        else:
            intercept_val = np.mean(fatigue_data["MDF_Values"]) - slope * np.mean(fatigue_data["Time_Points"])
            
        # Línea de Tendencia
        y_pred = slope * fatigue_data["Time_Points"] + intercept_val
        # --------------------------------
        
        color_line = 'red' if is_fatigued else 'green'
        label_trend = f'Tendencia: {slope:.3f} Hz/s'
        
        plt.plot(fatigue_data["Time_Points"], y_pred, color=color_line, 
                linewidth=2, linestyle='--', label=label_trend)
        
        plt.title("3. Fatiga Muscular (Evolución de Frecuencia Mediana)")
        plt.ylabel("Frecuencia (Hz)")
        plt.xlabel("Tiempo (s)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Señal demasiado corta para análisis de fatiga", 
                 ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n ERROR CRÍTICO: {e}")
    import traceback
    traceback.print_exc()