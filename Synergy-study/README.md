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
    ├── 📂 EX-Results/              # Carpeta de salida individual (OUTPUT)
    │   ├── Informe_PRUEBA_MANUAL.csv      # Resumen numérico con VAF y ranking
    │   ├── Pearson_PRUEBA_MANUAL.png      # Matriz de calor (Redundancia)
    │   ├── Ranking_PRUEBA_MANUAL.png      # Gráfico de clasificación de músculos
    │   └── W_Sinergias_PRUEBA_MANUAL.png  # Gráfico de pesos de las sinergias
    │
    ├── 📜 synergy_lib.py           # Librería científica core (NNMF, Pearson, Filtros)
    ├── 📜 synergy.py               # Script analítico principal
    ├── 📜 generar_tabla_resumen.py # Script recolector de estadísticas globales
    └── 📜 run.bat                  # Script ejecutable de procesamiento masivo en lote

## ⚙️ ¿Qué hace este algoritmo?

El sistema ejecuta 4 etapas de validación científica de forma automática:

1. **Auditoría de Calidad (SNR):** Evalúa el ruido basal de cada sensor.
2. **Detección de Redundancia:** Calcula la correlación de Pearson para detectar músculos que aportan información repetida ($r > 0.90$).
3. **Extracción de Sinergias (NNMF):** Descompone la señal para encontrar los "comandos de control" principales del sistema nervioso, iterando hasta lograr una Varianza Explicada (VAF) superior al 90%. Genera un ranking de importancia muscular.
4. **Análisis Estadístico Global:** Al finalizar el procesamiento masivo, escanea todos los informes generados y crea una tabla maestra determinando, matemáticamente, qué músculos son estadísticamente inútiles (descartables) para el protocolo del laboratorio.

## 🚀 Cómo usarlo

1. **Preparar datos:** Coloca tus carpetas de muestras (ej. `V1`, `V2`) en la ruta configurada en el `.bat`.
2. **Ejecutar el pipeline:**
   - **Opción A (Recomendada):** Haz doble clic en el archivo `run.bat`. Esto procesará automáticamente todas las muestras, generará los gráficos individuales y, al final, ejecutará el resumen global.
   - **Opción B (Consola):** Abre una terminal y ejecuta el script de Python apuntando a tu carpeta de muestras:
     ```bash
     python synergy.py "C:\Ruta\A\Tus\Datos"
     ```

3. **Revisar resultados:** - **A nivel individual:** Abre la carpeta `EX-Results/` (o la carpeta `RESULTADOS_SINERGIAS` de tu toma) para ver las gráficas y el informe CSV específico de ese sujeto.
   - **A nivel global:** En la carpeta raíz de tus muestras se habrán generado automáticamente dos archivos maestros:
     - `RESUMEN_DESCARTABLES_TOTAL.csv`: Una tabla detallando qué se descartó en cada ensayo.
     - `RANKING_DESCARTABLES_COMUNES.csv`: Un ranking global con el conteo de los sensores más inútiles del estudio, listo para justificar decisiones de hardware en artículos o tesis.

## 📦 Requisitos

Asegúrate de tener instalado Python 3.x y las siguientes librerías:

    pip install pandas numpy matplotlib seaborn scikit-learn

*(Nota: El sistema intentará cargar la librería externa `noraxon_analytics.py` si está disponible en el directorio padre para el filtrado avanzado de EMG).*