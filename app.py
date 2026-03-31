import streamlit as st
import mlflow.sklearn
import pandas as pd
import numpy as np
#TODO: Implement app for AWS deployment

model = mlflow.sklearn.load_model("models:/best_credit_loan_model/1")

st.title("Credit Loan Default Prediction")
st.write("Remplis les informations du client pour prédire le risque de défaut.")

credit_lines = st.number_input("Nombre de lignes de crédit", min_value=0)
loan_amt = st.number_input("Montant du prêt", min_value=0.0)
total_debt = st.number_input("Dette totale", min_value=0.0)
income = st.number_input("Revenu annuel", min_value=0.0)
years_employed = st.number_input("Années d'emploi", min_value=0.0)
fico_score = st.number_input("Score FICO", min_value=300, max_value=850)

if st.button("Prédire"):
    input_data = pd.DataFrame([[credit_lines, loan_amt, total_debt, income, years_employed, fico_score]],
                               columns=["credit_lines_outstanding", "loan_amt_outstanding", 
                                        "total_debt_outstanding", "income", 
                                        "years_employed", "fico_score"])
    
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"Risque de défaut détecté - probabilité : {proba:.1%})")
    else:
        st.success(f"Pas de risque de défaut - probabilité : {proba:.1%})")