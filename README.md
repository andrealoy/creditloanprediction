# Prédiction de défaut de crédit (Projet MLOps)

## Aperçu
Ce projet vise à prédire la probabilité de défaut de paiement pour des prêts personnels dans un contexte de banque de détail. Il s’inscrit dans une démarche MLOps de bout en bout.

## Architecture
- **Frontend** : Streamlit (interface utilisateur)
- **Backend** : FastAPI (API de prédiction)
- **Modèle** : Random Forest entraîné sur des données de crédit
- **Tracking** : MLflow
- **Déploiement** : Docker + AWS (ECR / ECS)

## Fonctionnalités
- Saisie des données client
- Prédiction de la probabilité de défaut
- Affichage du niveau de risque et de la probabilité
- Architecture basée sur API (pas de chargement direct du modèle côté frontend)

## Lancement en local

### Backend
uvicorn main:app --reload

### Frontend
streamlit run app.py

## Lancement avec Docker

### Build
docker build -t creditloanprediction:latest .

### Run
docker run --rm -p 8000:8000 -p 8501:8501 creditloanprediction:latest

Le backend FastAPI est alors disponible sur `http://localhost:8000` et le frontend Streamlit sur `http://localhost:8501`.

## Points MLOps
- Séparation frontend / backend / modèle
- Communication via endpoint `/predict`
- URL d’API configurable pour le déploiement
- Gestion robuste des requêtes (timeout)
- Chargement par défaut du meilleur modèle enregistré dans MLflow (`best_credit_loan_model`, stage `Production`)

## Source du modèle

Par défaut, l'API charge le modèle depuis la registry MLflow locale du projet (`mlflow.db` + `mlruns/`).
L'inférence utilise directement les features attendues par le meilleur modèle enregistré (`best_credit_loan_model`), ce qui évite les problèmes de portabilité du préprocesseur sérialisé dans le notebook.

Dans l'image Docker, le mode par défaut est `auto` : l'API utilise MLflow si une registry est disponible, sinon elle retombe sur les artefacts locaux du dossier `models/`. Cela évite de rendre le build dépendant de fichiers `mlruns/` non versionnés.
Le fallback local utilise l'artefact MLflow exporté du modèle de production embarqué dans `models/mlflow_best_credit_loan_model`.

Pour forcer un fallback sur les artefacts locaux `models/`, définir :

`MODEL_SOURCE=local`

Pour forcer strictement l'usage de MLflow, définir :

`MODEL_SOURCE=mlflow`

Pour réactiver explicitement le préprocesseur enregistré dans MLflow, définir :

`MLFLOW_USE_PREPROCESSOR=true`

## Contributions (équipe)
- Entraînement du modèle et suivi MLflow
- Développement de l’API FastAPI
- Développement de l’interface Streamlit
- CI/CD et déploiement AWS