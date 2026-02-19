@echo off
setlocal enabledelayedexpansion

:: 1. CONFIGURA AQUÍ LA CARPETA DONDE ESTÁN TUS TOMAS (V1, V2, V3...)
set "CARPETA_RAIZ=C:\Users\alexs\Desktop\MUESTRAS"

echo ========================================================
echo      PROCESAMIENTO BIOMECANICO DE VARIAS MUESTRAS (NORAXON + XSENS + SENSOR FUERZA)
echo ========================================================
echo.

:: 2. BUCLE: Busca todas las carpetas que empiecen por "V" (V1, V2, V10...)
for /D %%d in ("%CARPETA_RAIZ%\V*") do (
    echo --------------------------------------------------------
    echo [PROCESANDO TOMA]: %%~nxd
    echo RUTA: "%%d"
    
    :: 3. LLAMADA AL SCRIPT DE PYTHON
    :: Le pasamos la ruta de la carpeta encontrada como argumento
    :: Vaciar carpeta de salida PROCESADO_COMPLETO antes de procesar
    if exist "%%d\PROCESADO_COMPLETO" (
        echo Eliminando carpeta PROCESADO_COMPLETO en %%~nxd...
        rd /s /q "%%d\PROCESADO_COMPLETO"
    )
    mkdir "%%d\PROCESADO_COMPLETO"

    python fusion_multimodal.py "%%d"
    
    if !errorlevel! equ 0 (
        echo [OK] Toma %%~nxd procesada correctamente.
    ) else (
        echo [ERROR] Fallo al procesar %%~nxd. Revisa los archivos.
    )
)

echo.
echo ========================================================
echo                 FIN DEL PROCESO
echo ========================================================
pause