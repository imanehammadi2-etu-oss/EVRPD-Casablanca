#!/bin/bash
set -e

echo "=============================================================================="
echo "  LANCEMENT DE L'APPLICATION EVRPD CASABLANCA AVEC DOCKER"
echo "=============================================================================="
echo ""

if ! command -v docker &> /dev/null; then
    echo "[ERREUR] Docker n'est pas installe sur ce systeme !"
    echo "Veuillez installer Docker puis relancer ce script."
    exit 1
fi

echo "[1/2] Construction et demarrage du conteneur Docker (Port 8515)..."
docker compose up -d --build

echo ""
echo "[2/2] Conteneur demarre avec succes !"
echo "=============================================================================="
echo "  L'application est disponible aux adresses :"
echo "  - Local :      http://localhost:8515"
echo "  - Reseau IP :  http://192.168.1.42:8515 ou http://192.168.1.43:8515"
echo "=============================================================================="
echo ""

# Tentative d'ouverture automatique du navigateur
if which xdg-open > /dev/null; then
    xdg-open http://localhost:8515
elif which open > /dev/null; then
    open http://localhost:8515
fi

echo "Affichage des logs en direct (Ctrl+C pour quitter sans arreter le conteneur) :"
docker compose logs -f
