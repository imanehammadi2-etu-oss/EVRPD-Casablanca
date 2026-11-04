@echo off
TITLE "Plateforme EVRPD Casablanca - Docker Launcher"
echo ==============================================================================
echo   LANCEMENT DE L'APPLICATION EVRPD CASABLANCA AVEC DOCKER
echo ==============================================================================
echo.

REM Verifier si Docker est installe et demarre
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Docker n'est pas demarre ou n'est pas installe !
    echo Veuillez installer et lancer Docker Desktop, puis relancer ce script.
    echo.
    pause
    exit /b 1
)

echo [1/2] Construction et demarrage du conteneur Docker (Port 8515)...
docker compose up -d --build

echo.
echo [2/2] Conteneur demarre avec succes !
echo ==============================================================================
echo   L'application est disponible aux adresses :
echo   - Local :      http://localhost:8515
echo   - Reseau IP :  http://192.168.1.42:8515 ou http://192.168.1.43:8515
echo ==============================================================================
echo.

REM Ouvrir automatiquement le navigateur par defaut
start http://localhost:8515

echo Appuyez sur une touche pour afficher les logs en direct (Ctrl+C pour quitter)...
docker compose logs -f
