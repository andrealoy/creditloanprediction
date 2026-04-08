# 1. Utiliser une image Python légère
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_URL=http://127.0.0.1:8000/predict
ENV MODEL_SOURCE=auto
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV MLFLOW_REGISTRY_URI=sqlite:////app/mlflow.db

# 3. Copier le fichier des dépendances
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier le reste du code et le modèle
COPY app.py .
COPY entrypoint.sh .
COPY main.py .
COPY models ./models

RUN chmod +x /app/entrypoint.sh

# 6. Exposer les ports du backend et du frontend
EXPOSE 8000
EXPOSE 8501

# 7. Lancer l'API et le frontend Streamlit
CMD ["/app/entrypoint.sh"]