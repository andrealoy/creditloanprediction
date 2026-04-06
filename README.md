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

## Points MLOps
- Séparation frontend / backend / modèle
- Communication via endpoint `/predict`
- URL d’API configurable pour le déploiement
- Gestion robuste des requêtes (timeout)

## Contributions (équipe)
- Entraînement du modèle et suivi MLflow
- Développement de l’API FastAPI
- Développement de l’interface Streamlit
- CI/CD et déploiement AWS