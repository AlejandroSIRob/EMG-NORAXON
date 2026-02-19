"""
MAIN: ANÁLISIS MULTITAREA AUTOMATIZADO (ARGUMENTOS)
Usa la librería científica funcional 'synergy_lib.py'
"""
import pandas as pd
import os
import sys  
import synergy_lib as lab

# 1. MAPEO DE SENSORES
MY_SENSORS = {
    'FLEX.CARP.R': 'Flexor Rad.',
    'BRACHIORAD.': 'Braquiorradial',
    'EXT.DIG.': 'Extensor Dig.',     
    'EXT.CARP.ULN.': 'Extensor Uln.', 
    'BICEPS BR.': 'Bíceps',
    'LAT. TRICEPS': 'Tríceps',
    'ANT.DELTOID': 'Delt. Ant.',
    'MID DELT.': 'Delt. Med.',
    'POST.DELTOID': 'Delt. Post.',
    'FLEX.CARP.U': 'Flexor Uln.',
    'PECT. MAJOR': 'Pectoral',
    'INFRASPIN.': 'Infraespinoso'     
}

def main():
    # --- LEER ARGUMENTOS DEL .BAT ---
    if len(sys.argv) > 1:
        # Si el .bat nos manda una carpeta, usamos esa
        input_folder_root = sys.argv[1] # Ej: C:\Users\...\MUESTRAS\V1
        
        # Asumimos que dentro de V1 hay una carpeta "EMG". 
        target_folder = os.path.join(input_folder_root, "EMG") 
        
        # Nombre de la tarea basado en la carpeta (Ej: V1)
        task_name = os.path.basename(input_folder_root)
        
        TASKS_CONFIG = {task_name: target_folder}
        
        # Carpeta de salida: Dentro de la propia carpeta V1 para ser ordenados
        output_dir = os.path.join(input_folder_root, "RESULTADOS_SINERGIAS")
        
    else:
        # Modo manual (por si lo ejecutas tú solo para probar)
        print("[MODO MANUAL] No se recibieron argumentos. Usando ruta por defecto.")
        TASKS_CONFIG = {
            "PRUEBA_MANUAL": r"C:\Users\alexs\Desktop\MUESTRAS\V1\EMG"
        }
        output_dir = r"C:\Users\alexs\Desktop\RESULTADOS_MANUALES"

    # --- INICIO DEL PROCESO ---
    print(f"=== PROCESANDO: {list(TASKS_CONFIG.keys())[0]} ===")
    
    reporte_general = []

    for task_name, folder_path in TASKS_CONFIG.items():
        # Crear carpeta de resultados si no existe
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        try:
            # A. Cargar Datos
            df_active = lab.cargar_y_procesar_datos(folder_path, MY_SENSORS)
            
            # B. Calidad
            lab.reportar_calidad_senal(df_active)
            
            # C. Redundancia (Pasamos output_dir para guardar ahí las fotos)
            lab.analizar_redundancia_pearson(df_active, output_dir, task_name)
            
            # D. Sinergias
            resultado_optimo = lab.buscar_sinergias_optimas(df_active)
            
            # E. Gráficos
            df_ranking = lab.generar_ranking_y_graficos(df_active, resultado_optimo, output_dir, task_name)
            
            print(f"   [ÉXITO] Resultados guardados en: {output_dir}")
            
            # F. Resumen
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
            print(f"   [ERROR CRÍTICO] {task_name}: {e}")

    # --- GUARDAR EXCEL INDIVIDUAL POR TOMA ---
    if reporte_general:
        df_final = pd.DataFrame(reporte_general)
        excel_path = os.path.join(output_dir, f"Informe_{task_name}.csv")
        df_final.to_csv(excel_path, index=False)
        print(f"\n[FIN] Informe CSV guardado en: {excel_path}")

if __name__ == "__main__":
    main()