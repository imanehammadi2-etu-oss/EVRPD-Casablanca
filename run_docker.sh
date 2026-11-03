#!/bin/bash
set -e

echo "=============================================================================="
echo "  LANCEMENT DE L'APPLICATION EVRPD CASABLANCA AVEC DOCKER"
echo "=============================================================================="
echo ""

if ! command -v docker &> /dev/null; then
    echo "[ERREUR] Docker n'est pas installé sur ce système !"
    echo "Veuillez installer Docker puis relancer ce script."
    exit 1
fi

echo "[1/2] Construction et démarrage du conteneur Docker..."
docker compose up -d --build

echo ""
echo "[2/2] Conteneur démarré avec succès !"
echo "=============================================================================="
echo "  L'application est disponible à l'adresse :"
echo "  http://localhost:8501"
echo "=============================================================================="
echo ""

# Tentative d'ouverture automatique du navigateur
if which xdg-open > /dev/null; then
    xdg-open http://localhost:8501
elif which open > /dev/null; then
    open http://localhost:8501
fi

echo "Affichage des logs en direct (Ctrl+C pour quitter sans arrêter le conteneur) :"
docker compose logs -f
