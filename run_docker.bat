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

echo [1/2] Construction et demarrage du conteneur Docker...
docker compose up -d --build

echo.
echo [2/2] Conteneur demarre avec succes !
echo ==============================================================================
echo   L'application est disponible a l'adresse :
echo   http://localhost:8501
echo ==============================================================================
echo.

REM Ouvrir automatiquement le navigateur par defaut
start http://localhost:8501

echo Appuyez sur une touche pour afficher les logs en direct (Ctrl+C pour quitter)...
docker compose logs -f
