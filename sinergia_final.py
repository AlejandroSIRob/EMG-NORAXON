"""
MAIN: ANÁLISIS MULTITAREA DE SELECCIÓN DE SENSORES
Usa la librería científica funcional 'lib_sinergias_cientifica.py'
"""
import pandas as pd
import os
import synergy_lib as lab  # Importamos tu librería como 'lab'

# 1. CONFIGURACIÓN DE TAREAS Y RUTAS
# Ajusta estas rutas a tus carpetas reales
TASKS_CONFIG = {
    # "ENSAMBLADO": r"C:\Ruta\A\Datos_Ensamblado",
    # "CORTE_CARNE": r"C:\Ruta\A\Datos_Corte",
    # "DESMOLDEADO": r"C:\Ruta\A\Datos_Desmoldeado",
    
    # EJEMPLO CON CARPETA ACTUAL PARA PROBAR:
    "PRUEBA_BRAZO": r"C:\Users\alexs\Desktop\EMG_NORAVOX\Test_Medicion_Sin_Procesar\Solar_Iglesias,_Alejandro"
}

# 2. MAPEO DE SENSORES (Los 11 finales)
MY_SENSORS = {
    'FLEX.CARP.R': 'Flexor Rad.', 'BRACHIORAD.': 'Braquiorradial',
    'EXT.DIG.': 'Extensor', 'BICEPS BR.': 'Bíceps',
    'LAT. TRICEPS': 'Tríceps', 'ANT.DELTOID': 'Delt. Ant.',
    'MID DELT.': 'Delt. Med.', 'POST.DELTOID': 'Delt. Post.',
    'FLEX.CARP.U': 'Flexor Uln.', 'PECT. MAJOR': 'Pectoral',
    'LAT.DORSI': 'Dorsal'
}

def main():
    print("=== INICIO DEL ANÁLISIS MULTITAREA CIENTÍFICO ===")
    
    reporte_general = []

    for task_name, folder_path in TASKS_CONFIG.items():
        print(f"\n=============================================")
        print(f"PROCESANDO TAREA: {task_name}")
        print(f"=============================================")
        
        output_dir = r"C:\Users\alexs\Desktop\EMG_NORAVOX\Resultados_Completo_Sinergias"
        
        try:
            # A. Cargar Datos y Filtrar (Yokoyama 2019)
            df_active = lab.cargar_y_procesar_datos(folder_path, MY_SENSORS)
            
            # B. Auditoría de Calidad (Konrad 2005)
            lab.reportar_calidad_senal(df_active)
            
            # C. Análisis de Redundancia (Pearson)
            lab.analizar_redundancia_pearson(df_active, output_dir, task_name)
            
            # D. Extracción de Sinergias (NNMF + VAF > 90%)
            resultado_optimo = lab.buscar_sinergias_optimas(df_active)
            
            # E. Generar Gráficos y Ranking
            df_ranking = lab.generar_ranking_y_graficos(df_active, resultado_optimo, output_dir, task_name)
            
            print(f"   [ÉXITO] Resultados guardados en: {output_dir}")
            
            # F. Guardar resumen para Excel
            top_5 = df_ranking.head(5)['Musculo'].tolist()
            bottom_3 = df_ranking.tail(3)['Musculo'].tolist()
            
            reporte_general.append({
                'Tarea': task_name,
                'N_Sinergias': resultado_optimo['n'],
                'VAF_Final': round(resultado_optimo['vaf'], 2),
                'Top_Sensores': ", ".join(top_5),
                'Descartables': ", ".join(bottom_3)
            })

        except Exception as e:
            print(f"   [ERROR CRÍTICO] No se pudo procesar {task_name}: {e}")

    # --- GENERAR EXCEL FINAL ---
    if reporte_general:
        df_final = pd.DataFrame(reporte_general)
        print("\n\n=== TABLA COMPARATIVA FINAL ===")
        print(df_final.to_string(index=False))
    
    print("\n[FIN] Ejecución completada.")

if __name__ == "__main__":
    main()