# 📊 Estudio de Sinergias Musculares (EMG Synergy Analysis)

Este repositorio contiene un *pipeline* algorítmico automatizado para el procesamiento de señales electromiográficas (EMG) provenientes de sistemas Noraxon. Su objetivo principal es extraer **sinergias musculares**, analizar la calidad de la señal y generar un ranking para optimizar la cantidad de sensores necesarios en estudios biomecánicos o de control robótico.

## 📁 Estructura del Proyecto

El directorio está organizado de la siguiente manera:

    Synergy-study/
    │
    ├── 📂 Documentation_vSpanish/  # Documentación técnica y metodológica
    │   └── Estudio_Sinergía.pdf    # Informe detallado de la metodología (NNMF, VAF, etc.)
    │
    ├── 📂 Samples/                 # Carpeta de entrada (INPUT)
    │   └── 📂 EMG/                 # Colocar aquí los archivos .csv crudos de Noraxon
    │       └── 2026-02-04-19-37_v1.csv
    │
    ├── 📂 EX-Results/              # Carpeta de salida (OUTPUT)
    │   ├── Informe_PRUEBA_MANUAL.csv      # Resumen numérico con VAF y ranking
    │   ├── Pearson_PRUEBA_MANUAL.png      # Matriz de calor (Redundancia)
    │   ├── Ranking_PRUEBA_MANUAL.png      # Gráfico de clasificación de músculos
    │   └── W_Sinergias_PRUEBA_MANUAL.png  # Gráfico de pesos de las sinergias
    │
    ├── 📜 synergy_lib.py           # Librería científica core (NNMF, Pearson, Filtros)
    ├── 📜 synergy.py               # Script principal ejecutable
    └── 📜 run.bat                  # Script de ejecución rápida en lote

## ⚙️ ¿Qué hace este algoritmo?

El sistema ejecuta 3 etapas de validación científica de forma automática:

1. **Auditoría de Calidad (SNR):** Evalúa el ruido basal de cada sensor.
2. **Detección de Redundancia:** Calcula la correlación de Pearson para detectar músculos que aportan información repetida ($r > 0.90$).
3. **Extracción de Sinergias (NNMF):** Descompone la señal para encontrar los "comandos de control" principales del sistema nervioso, iterando hasta lograr una Varianza Explicada (VAF) superior al 90%. Al final, genera un ranking de importancia muscular.

## 🚀 Cómo usarlo

1. **Preparar datos:** Coloca los archivos `.csv` exportados de Noraxon dentro de la carpeta `Samples/EMG/`.
2. **Ejecutar el pipeline:**
   - **Opción A (Recomendada):** Haz doble clic en el archivo `run.bat`. Esto procesará automáticamente todas las muestras que encuentre configuradas.
   - **Opción B (Consola):** Abre una terminal y ejecuta el script de Python apuntando a tu carpeta de muestras:
     
         python synergy.py "C:\Ruta\A\Tus\Datos"

3. **Revisar resultados:** Abre la carpeta `EX-Results/` para ver los gráficos generados y el archivo CSV con el diagnóstico final, indicando qué sensores son críticos y cuáles pueden ser descartados.

## 📦 Requisitos

Asegúrate de tener instalado Python 3.x y las siguientes librerías:

    pip install pandas numpy matplotlib seaborn scikit-learn

*(Nota: El sistema intentará cargar la librería externa `noraxon_analytics.py` si está disponible en el directorio padre para el filtrado avanzado de EMG).*