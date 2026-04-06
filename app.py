import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# Configurable API endpoint (supports local and AWS deployment)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")




st.title("Credit Loan Default Prediction")
st.write("Remplis les informations du client pour prédire le risque de défaut.")

credit_lines = st.number_input("Nombre de lignes de crédit", min_value=0)
loan_amt = st.number_input("Montant du prêt", min_value=0.0)
total_debt = st.number_input("Dette totale", min_value=0.0)
income = st.number_input("Revenu annuel", min_value=0.0)
years_employed = st.number_input("Années d'emploi", min_value=0.0)
fico_score = st.number_input("Score FICO", min_value=300, max_value=850)

if st.button("Prédire"):
    payload = {
        "credit_lines_outstanding": credit_lines,
        "loan_amt_outstanding": loan_amt,
        "total_debt_outstanding": total_debt,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score
    }

    try:
        with st.spinner("Calcul en cours..."):
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            prediction = result.get("prediction")
            proba = result.get("default_probability")
            risk_level = result.get("risk_level")

        if prediction == "Default":
            st.error(f"Risque de défaut ({risk_level}) - probabilité : {proba:.1%}")
        else:
            st.success(f"Pas de risque ({risk_level}) - probabilité : {proba:.1%}")

    except Exception as e:
        st.error(f"Erreur API : {e}")