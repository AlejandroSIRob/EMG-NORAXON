# 🔗 Biomech-Multimodal-Fusion (EMG + IMU + Force)

Este repositorio contiene la herramienta de infraestructura computacional para la **sincronización y fusión de datos biomecánicos multimodales**. Alinea temporalmente tres fuentes de datos de hardware distinto en un único dataset maestro, utilizando como anclaje eventos físicos de impacto (Jerk/Aceleración).

## ⚙️ ¿Qué hace este algoritmo (Pipeline)?

El script `fusion_multimodal.py` procesa carpetas de tomas crudas y busca un patrón de sincronización físico: **3 impactos secos consecutivos en menos de 2.5 segundos**.

1. **Sensor de Fuerza (OnRobot):** Detecta picos de fuerza (Newtons) correspondientes a los golpes físicos.
2. **Sistema EMG (Noraxon):** Detecta el artefacto electromagnético o el pico de activación de la señal compuesta (suma de envolventes lineales).
3. **IMU Cinemática (Xsens):** Calcula la segunda derivada de la velocidad (Jerk) para aislar el momento exacto de la sacudida en la mano/herramienta.

Una vez encontrados y validados los 3 eventos, el algoritmo establece el anclaje temporal ($T=0$ en el tercer impacto) y re-mapea (interpola) todas las señales a una frecuencia maestra de **100 Hz**, generando un dataset unificado.

## 📁 Estructura Esperada de Directorios

Para que el script por lotes funcione correctamente, tus datos deben estar organizados de la siguiente manera:

    /EMG_NORAVOX                    # Directorio Padre
    │
    ├── 📜 noraxon_analytics.py     # Librería científica (Requerida en el nivel superior)
    │
    ├── /SYNC                       # ESTE REPOSITORIO
    │   ├── 📜 fusion_multimodal.py 
    │   ├── 📜 Ejecutar_Fusion_multimodal.bat
    │   └── 📜 README.md
    │
    └── /MUESTRAS                   # Carpeta raíz de datos (Configurable en el .bat)
        ├── /V1
        │   ├── /EMG                # CSV exportado de Noraxon
        │   ├── /FUERZA             # CSV exportado del sensor de fuerza
        │   └── /PROCESADO-Xsens    # STO generado por MT Manager / OpenSim
        ├── /V2
        └── /V...

## 🚀 Cómo usarlo

### Paso 1: Configurar la ruta
Abre el archivo `Ejecutar_Fusion_multimodal.bat` con un bloc de notas y asegúrate de que la variable `CARPETA_RAIZ` apunte a la carpeta donde tienes guardadas tus tomas (V1, V2, etc.):

    set "CARPETA_RAIZ=C:\Users\tu_usuario\Desktop\MUESTRAS"

### Paso 2: Ejecución masiva
Haz doble clic sobre `Ejecutar_Fusion_multimodal.bat`. 
El script de Windows buscará automáticamente todas las subcarpetas que empiecen por "V", creará una carpeta de salida y ejecutará el código en Python para cada una de ellas.

### Paso 3: Revisar Resultados
Al finalizar, dentro de cada toma (ej. `V1`), se habrá creado una nueva carpeta llamada `PROCESADO_COMPLETO` que contendrá:
* `DATASET_MAESTRO.csv`: El archivo final con todas las señales (Fuerza, EMG, IMUs) alineadas en el mismo eje de tiempo.
* `REPORTE_EMG.png`: Gráfico de validación visual de la sincronización muscular vs. fuerza.
* `REPORTE_IMU.png`: Gráfico de validación visual de los sensores inerciales vs. fuerza.

## 📦 Requisitos Técnicos

Asegúrate de tener instalado Python 3.x y las siguientes librerías:

    pip install pandas numpy matplotlib scipy

*Nota: El script está programado para buscar y cargar automáticamente la librería local `noraxon_analytics.py` desde el directorio padre (`..`). Si no la encuentra, aplicará un procesado EMG básico de contingencia.*