import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Configuration de la page
st.set_page_config(page_title="Prédiction Dépenses - KNN", layout="wide")

# 2. Chargement des composants sauvegardés
@st.cache_resource
def load_assets():
    model = joblib.load('modele_knn.pkl')
    scaler = joblib.load('scaler.pkl')
    features_list = joblib.load('features.pkl')
    return model, scaler, features_list

model, scaler, features_list = load_assets()

# 3. Interface utilisateur
st.title("💰 Simulateur de Prédiction des Dépenses")
st.markdown("""
Cette application utilise un modèle **K-Nearest Neighbors (KNN)** optimisé pour estimer les dépenses 
en fonction du profil financier.
""")

with st.sidebar:
    st.header("Paramètres d'entrée")
    # Variables Numériques
    bilan = st.number_input("Bilan Financier", value=50000.0)
    actifs = st.number_input("Actifs", value=20000.0)
    revenu = st.number_input("Revenu mensuel", value=3000.0)
    taux = st.slider("Taux d'intérêt (%)", 0.0, 10.0, 5.0)
    flux = st.number_input("Flux de trésorerie", value=10000.0)
    capital = st.number_input("Capital", value=40000.0)
    
    # Variables Catégorielles
    agence = st.selectbox("Agence", ['Agence_Centre', 'Agence_Sud', 'Agence_Nord'])
    banque = st.selectbox("Banque", ['Société Générale', 'UBA', 'Ecobank', 'BGFI'])
    lieu = st.selectbox("Lieu", ['Bafoussam', 'Douala', 'Yaoundé', 'Garoua'])

# 4. Préparation des données pour le modèle
if st.button("Lancer la prédiction"):
    # Création d'un dictionnaire avec les saisies
    input_data = {
        'bilan_financier': bilan,
        'actifs': actifs,
        'revenu': revenu,
        'taux_interet': taux,
        'flux_tresorerie': flux,
        'capital': capital,
        'agence': agence,
        'banque': banque,
        'lieu': lieu
    }
    
    # Conversion en DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Feature Engineering (doit être identique à l'entraînement)
    df_input['pression_depense'] = df_input['revenu'] / (df_input['actifs'] + 1)
    df_input['ratio_cash_capital'] = df_input['flux_tresorerie'] / (df_input['capital'] + 1)
    
    # Encodage (One-Hot Encoding)
    df_encoded = pd.get_dummies(df_input)
    
    # Aligner les colonnes avec celles vues durant l'entraînement
    # On crée les colonnes manquantes avec des 0
    for col in features_list:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # On réordonne pour correspondre exactement
    df_final = df_encoded[features_list]
    
    # Scaling
    X_scaled = scaler.transform(df_final)
    
    # Prédiction
    prediction = model.predict(X_scaled)[0]
    
    # Affichage du résultat
    st.success(f"### Dépenses estimées : **{prediction:,.2f} XAF**")
    
    # Petit indicateur visuel
    st.metric(label="Estimation", value=f"{prediction:,.0f} XAF")