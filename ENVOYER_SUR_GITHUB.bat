@echo off
TITLE "Envoi vers GitHub - EVRPD Casablanca"
cd /d "%~dp0"

echo ==============================================================================
echo   ENVOI AUTOMATIQUE VERS VOTRE GITHUB (imanehammadi2-etu-oss)
echo ==============================================================================
echo.

set /p TOKEN="Collez votre Personal Access Token GitHub (ou appuyez sur Entree pour la connexion classique) : "

if "%TOKEN%"=="" (
    "C:\Users\MSI\AppData\Local\MinGit\cmd\git.exe" remote set-url origin https://github.com/imanehammadi2-etu-oss/EVRPD-Casablanca.git
) else (
    "C:\Users\MSI\AppData\Local\MinGit\cmd\git.exe" remote set-url origin https://%TOKEN%@github.com/imanehammadi2-etu-oss/EVRPD-Casablanca.git
)

echo.
echo Envoi en cours vers la branche main...
"C:\Users\MSI\AppData\Local\MinGit\cmd\git.exe" push -u origin main

echo.
echo ==============================================================================
echo   Termine ! Rendez-vous sur https://github.com/imanehammadi2-etu-oss/EVRPD-Casablanca
echo ==============================================================================
pause
