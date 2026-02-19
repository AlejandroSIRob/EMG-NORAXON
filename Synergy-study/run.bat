@echo off
setlocal enabledelayedexpansion

:: ========================================================
:: CONFIGURACIÓN
:: ========================================================

:: 1. Carpeta raíz donde están tus carpetas de datos V1, V2, etc.
set "CARPETA_RAIZ=C:\Users\alexs\Desktop\MUESTRAS"

:: 2. Nombre de tu script de Python (debe estar en la misma carpeta que este .bat)
set "SCRIPT_PYTHON=synergy.py"

echo ========================================================
echo      PROCESAMIENTO MASIVO DE SINERGIAS MUSCULARES
echo ========================================================
echo.

:: ========================================================
:: BUCLE PRINCIPAL
:: ========================================================

:: Busca todas las carpetas que empiecen por "V" dentro de MUESTRAS
for /D %%d in ("%CARPETA_RAIZ%\V*") do (
    echo --------------------------------------------------------
    echo [PROCESANDO TOMA]: %%~nxd
    echo RUTA DETECTADA: "%%d"
    
    :: Comprobar si existe la carpeta EMG dentro (Seguridad)
    if exist "%%d\EMG" (
        
        :: Limpieza previa (Opcional: borra resultados antiguos si quieres)
        if exist "%%d\RESULTADOS_SINERGIAS" (
            echo Limpiando resultados anteriores...
            del /Q "%%d\RESULTADOS_SINERGIAS\*.*"
        )
        
        :: LLAMADA AL PYTHON
        :: Le pasamos la ruta completa de la carpeta "V..." como argumento 1
        python %SCRIPT_PYTHON% "%%d"
        
        if !errorlevel! equ 0 (
            echo [OK] %%~nxd analizado con exito.
        ) else (
            echo [ERROR] Fallo en el script de Python para %%~nxd.
        )
        
    ) else (
        echo [AVISO] Se salta %%~nxd porque no tiene carpeta "EMG".
    )
)

echo.
echo ========================================================
echo                 FIN DEL PROCESO MASIVO
echo ========================================================
pause