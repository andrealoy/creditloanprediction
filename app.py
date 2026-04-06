import streamlit as st
import mlflow.sklearn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests

# --- Configurable API endpoint (supports local and AWS deployment)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# --- Style
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="wide"
)

# --- Page title and layout
st.title("Credit Loan Default Prediction")
st.caption("Remplis les informations du client pour prédire le risque de défaut.")
st.divider()

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Informations client")

    #credit_lines   = st.number_input("Nombre de lignes de crédit", min_value=0, value=3)
    loan_amt       = st.number_input("Montant du prêt restant (€)", min_value=0.0, value=20000.0, step=500.0)
    total_debt     = st.number_input("Dette totale restante (€)", min_value=0.0, value=8000.0, step=500.0)
    income         = st.number_input("Revenu annuel (€)", min_value=0.0, value=45000.0, step=1000.0)
    years_employed = st.number_input("Années d'emploi", min_value=0.0, value=5.0, step=0.5)
    fico_score     = st.slider("Score FICO", min_value=300, max_value=850, value=680)

    predict_btn = st.button("Prédire le risque", type="primary", use_container_width=True)

with col_result:
    st.subheader("Résultat")

    if predict_btn:

        payload = {
            #"credit_lines_outstanding": credit_lines,
            "loan_amt_outstanding": loan_amt,
            "total_debt_outstanding": total_debt,
            "income": income,
            "years_employed": years_employed,
            "fico_score": fico_score
        }

        try:
            with st.spinner("Calcul en cours..."):
                response = requests.post(API_URL, json=payload, timeout=5)
                st.write(f"Status API: {response.status_code}")
                response.raise_for_status()
                result = response.json()
                prediction = result.get("prediction")
                proba = result.get("default_probability")
                risk_level = result.get("risk_level")

            # --- Visuels
            # Recalcul dti and lti pour éviter le log
            dti_display = (total_debt / income) * 100  # en %
            lti_display = (loan_amt / income) * 100    # en %
            debt_to_loan = total_debt / loan_amt *100

            # Visuel de gauge
            gauge_color = "#e74c3c" if proba > 0.6 else "#f39c12" if proba > 0.3 else "#27ae60"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0,  30], "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fdebd0"},
                        {"range": [60, 100],"color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": gauge_color, "width": 3},
                        "thickness": 0.75,
                        "value": round(proba * 100, 1)
                    }
                }
            ))
            fig.update_layout(
                height=220,
                margin=dict(t=20, b=0, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)"
                )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Métriques
            m1, m2, m3 = st.columns(3)

            m1.metric("Ratio dette/revenu", f"{dti_display:.1f}%",
                    delta="+Élevé" if dti_display > 30 else "-OK",
                    delta_color="inverse")

            m2.metric("Ratio prêt/revenu", f"{lti_display:.1f}%",
                    delta="+Élevé" if lti_display > 30 else "-OK",
                    delta_color="inverse")

            m3.metric("Ratio dette/prêt", f"{debt_to_loan:.1f}%",
                    delta="+Élevé" if debt_to_loan > 30 else "-OK",
                    delta_color="inverse")

        except Exception as e:
            st.error(f"Erreur API : {e}")