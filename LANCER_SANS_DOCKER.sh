#!/bin/bash
set -e

echo "=============================================================================="
echo "  LANCEMENT DIRECT DE L'APPLICATION (MODE PYTHON LOCAL SANS DOCKER)"
echo "=============================================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERREUR] python3 n'est pas installé !"
    exit 1
fi

echo "[1/3] Installation des dépendances..."
pip3 install -r requirements.txt --quiet

echo "[2/3] Démarrage du serveur Web Streamlit..."
echo "L'application s'ouvre sur : http://localhost:8501"
echo ""

streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
