@echo off
TITLE "Plateforme EVRPD Casablanca - Lancement Local Python"
cd /d "%~dp0"

echo ==============================================================================
echo   LANCEMENT DE L'APPLICATION EVRPD CASABLANCA (MODE LOCAL PYTHON)
echo ==============================================================================
echo.

echo [1/3] Verification de Python...
python --version
if errorlevel 1 goto NO_PYTHON

echo.
echo [2/3] Verification des bibliotheques Python...
python -m pip install -r requirements.txt --quiet

echo.
echo [3/3] Demarrage du serveur Web Streamlit sur le port 8515...
echo ==============================================================================
echo   L'application s'ouvre sur : http://localhost:8515
echo   Accessible egalement sur le reseau local (ex: http://192.168.1.42:8515 ou http://192.168.1.43:8515)
echo   (Pour fermer l'application, fermez simplement cette fenetre)
echo ==============================================================================
echo.

REM Lancement du navigateur
start http://localhost:8515

REM Execution de Streamlit sur le port 8515
python -m streamlit run streamlit_app.py --server.port=8515 --server.address=0.0.0.0
goto END

:NO_PYTHON
echo [ERREUR] Python n'est pas installe ou n'est pas dans votre PATH Windows.
echo Rendez-vous sur https://www.python.org pour installer Python 3.10 ou superieur.
pause

:END
pause
