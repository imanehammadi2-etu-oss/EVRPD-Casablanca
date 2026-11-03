# ==============================================================================
# Dockerfile - Plateforme EVRPD & Prévision IA Casablanca (Streamlit)
# ==============================================================================

# Image de base légère et optimisée Python 3.11
FROM python:3.11-slim

# Empêcher la création de fichiers .pyc et forcer l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système minimales (curl pour le healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
# Installation préalable de PyTorch version CPU pour minimiser la taille de l'image (gain de ~3 Go)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copie de l'intégralité du projet et des données
COPY . .

# Port d'écoute de l'application Web Streamlit
EXPOSE 8501

# Vérification de santé du conteneur (Healthcheck natif Streamlit)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Démarrage direct du dashboard
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
