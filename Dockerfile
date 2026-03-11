# 1. Utiliser une image Python légère
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier le fichier des dépendances
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier le reste du code et le modèle
COPY main.py .
COPY models ./models

# 6. Exposer le port que FastAPI utilise
EXPOSE 8000

# 7. Lancer l'API avec uvicorn
# On utilise 0.0.0.0 pour que l'API soit accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]