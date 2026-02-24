import pandas as pd
import glob
import os
import sys
from collections import Counter

def generar_resumen_descartables(carpeta_raiz):
    print("\n" + "="*70)
    print(" GENERANDO TABLA RESUMEN GLOBAL Y ANÁLISIS DE DESCARTABLES")
    print("="*70)
    
    # Busca todos los informes CSV dentro de la carpeta MUESTRAS y subcarpetas
    patron_busqueda = os.path.join(carpeta_raiz, '**', 'Informe_*.csv')
    archivos_informe = glob.glob(patron_busqueda, recursive=True)
    
    if not archivos_informe:
        print("[AVISO] No se encontraron archivos de informe para resumir.")
        return
    
    lista_datos = []
    todos_descartables = []
    
    for archivo in archivos_informe:
        try:
            df = pd.read_csv(archivo)
            if 'Tarea' in df.columns and 'Descartables' in df.columns:
                tarea = df['Tarea'].iloc[0]
                descartables_str = str(df['Descartables'].iloc[0])
                
                # Guardar dato para la tabla general
                lista_datos.append({
                    'Grabación (Tarea)': tarea,
                    'Músculos Descartables': descartables_str
                })
                
                # Extraer y limpiar los nombres de los músculos para el contador
                if descartables_str.lower() != 'nan' and descartables_str.strip():
                    # Separar por comas y quitar espacios en blanco
                    musculos = [m.strip() for m in descartables_str.split(',') if m.strip()]
                    todos_descartables.extend(musculos)
                    
        except Exception as e:
            pass # Si falla un archivo, lo salta silenciosamente
            
    if not lista_datos:
        return

    # ---------------------------------------------------------
    # 1. TABLA DETALLE POR GRABACIÓN
    # ---------------------------------------------------------
    df_resumen = pd.DataFrame(lista_datos)
    df_resumen = df_resumen.sort_values(by='Grabación (Tarea)').reset_index(drop=True)
    
    print("\n--- DETALLE POR GRABACIÓN ---")
    print(df_resumen.to_string(index=False))
    
    # ---------------------------------------------------------
    # 2. TABLA RANKING (MÁS COMUNES)
    # ---------------------------------------------------------
    print("\n--- RANKING DE MÚSCULOS MÁS DESCARTADOS ---")
    if todos_descartables:
        # Cuenta cuántas veces aparece cada músculo
        conteo = Counter(todos_descartables)
        # Lo convierte a una tabla
        df_conteo = pd.DataFrame(conteo.items(), columns=['Músculo', 'Veces Descartado'])
        # Lo ordena de mayor a menor frecuencia
        df_conteo = df_conteo.sort_values(by='Veces Descartado', ascending=False).reset_index(drop=True)
        print(df_conteo.to_string(index=False))
    else:
        df_conteo = pd.DataFrame(columns=['Músculo', 'Veces Descartado'])
        print("Ningún músculo descartable encontrado.")
        
    print("\n" + "="*70)
    
    # ---------------------------------------------------------
    # 3. GUARDAR LOS ARCHIVOS EN LA CARPETA MUESTRAS
    # ---------------------------------------------------------
    ruta_salida_detalle = os.path.join(carpeta_raiz, 'RESUMEN_DESCARTABLES_TOTAL.csv')
    ruta_salida_ranking = os.path.join(carpeta_raiz, 'RANKING_DESCARTABLES_COMUNES.csv')
    
    df_resumen.to_csv(ruta_salida_detalle, index=False, encoding='utf-8-sig')
    df_conteo.to_csv(ruta_salida_ranking, index=False, encoding='utf-8-sig')
    
    print(f"[ÉXITO] Tabla Detalle guardada en:  {ruta_salida_detalle}")
    print(f"[ÉXITO] Tabla Ranking guardada en:  {ruta_salida_ranking}\n")

if __name__ == "__main__":
    # Lee la carpeta desde el argumento que le pasa el .bat (En tu caso será MUESTRAS)
    if len(sys.argv) > 1:
        carpeta = sys.argv[1]
    else:
        carpeta = "."
        
    generar_resumen_descartables(carpeta)