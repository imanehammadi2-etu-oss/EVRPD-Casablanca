#!/bin/bash
set -e

echo "=============================================================================="
echo "  LANCEMENT DIRECT DE L'APPLICATION (MODE PYTHON LOCAL SANS DOCKER)"
echo "=============================================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERREUR] python3 n'est pas installe !"
    exit 1
fi

echo "[1/3] Installation des dependances..."
pip3 install -r requirements.txt --quiet

echo "[2/3] Demarrage du serveur Web Streamlit sur le port 8515..."
echo "L'application s'ouvre sur : http://localhost:8515"
echo "Accessible sur le reseau : http://192.168.1.42:8515 ou http://192.168.1.43:8515"
echo ""

streamlit run streamlit_app.py --server.port=8515 --server.address=0.0.0.0
