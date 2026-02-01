import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. Configuration & Design
st.set_page_config(page_title="Executive Dashboard - Prédiction Dépenses", layout="wide")

# CSS pour un look épuré "Corporate"
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Chargement des données et modèles
@st.cache_resource
def load_assets():
    # Adapter les noms de fichiers si nécessaire
    model = joblib.load('modele_knn.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

@st.cache_data
def load_csv():
    return pd.read_csv('dataset_financier.csv')

model, scaler, features_list = load_assets()
df_hist = load_csv()

# 3. Sidebar - Paramètres de simulation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.header("⚙️ Paramètres Simulation")
    
    bilan = st.number_input("Bilan Financier", value=float(df_hist['bilan_financier'].mean()))
    actifs = st.number_input("Actifs", value=float(df_hist['actifs'].mean()))
    revenu = st.number_input("Revenu mensuel", value=float(df_hist['revenu'].mean()))
    taux = st.slider("Taux d'intérêt (%)", 0.0, 10.0, 5.0)
    flux = st.number_input("Flux de trésorerie", value=float(df_hist['flux_tresorerie'].mean()))
    capital = st.number_input("Capital", value=float(df_hist['capital'].mean()))
    
    st.divider()
    agence = st.selectbox("Agence", df_hist['agence'].unique())
    banque = st.selectbox("Banque", df_hist['banque'].unique())
    lieu = st.selectbox("Lieu", df_hist['lieu'].unique())
    
    predict_btn = st.button("📊 ANALYSER & PRÉDIRRE", use_container_width=True)

# 4. Corps Principal - Dashboard
st.title("🏛️ Direction Générale : Analyse des Coûts & Dépenses")
st.markdown("---")

# ONGLETS : Vue Globale vs Simulation Individuelle
tab1, tab2 = st.tabs(["📈 Vue d'ensemble Marché", "🎯 Simulateur de Performance"])

with tab1:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Dépense Moyenne", f"{df_hist['depenses'].mean():,.0f} XAF")
    col_b.metric("Revenu Moyen", f"{df_hist['revenu'].mean():,.0f} XAF")
    col_c.metric("Taux Moyen", f"{df_hist['taux_interet'].mean():.2f} %")

    st.markdown("### Répartition et Tendances")
    c1, c2 = st.columns(2)

    with c1:
        # Histogramme des dépenses
        fig_hist = px.histogram(df_hist, x="depenses", nbins=30, 
                                title="Distribution des Dépenses Historiques",
                                color_discrete_sequence=['#003366'])
        fig_hist.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        # Camembert des agences ou banques
        fig_pie = px.pie(df_hist, names='banque', title="Part de Marché par Institution",
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Courbe : Relation Revenu / Dépense
    st.markdown("### Analyse de Corrélation")
    fig_scatter = px.scatter(df_hist, x="revenu", y="depenses", color="agence",
                             trendline="ols", title="Relation Revenus vs Dépenses par Agence")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    if predict_btn:
        # --- LOGIQUE DE PRÉDICTION ---
        input_data = {
            'bilan_financier': bilan, 'actifs': actifs, 'revenu': revenu,
            'taux_interet': taux, 'flux_tresorerie': flux, 'capital': capital,
            'agence': agence, 'banque': banque, 'lieu': lieu
        }
        df_input = pd.DataFrame([input_data])
        
        # Feature Engineering
        df_input['pression_depense'] = df_input['revenu'] / (df_input['actifs'] + 1)
        df_input['ratio_cash_capital'] = df_input['flux_tresorerie'] / (df_input['capital'] + 1)
        
        # Encodage & Alignement
        df_encoded = pd.get_dummies(df_input)
        for col in features_list:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_final = df_encoded[features_list]
        
        # Prediction
        X_scaled = scaler.transform(df_final)
        prediction = model.predict(X_scaled)[0]

        # AFFICHAGE RÉSULTAT STYLE DASHBOARD
        st.markdown(f"""
            <div style="background-color:#003366; padding:30px; border-radius:15px; color:white; text-align:center;">
                <h2 style="color:white; margin:0;">DÉPENSE ESTIMÉE</h2>
                <h1 style="font-size:50px; margin:10px;">{prediction:,.0f} <span style="font-size:20px;">XAF</span></h1>
                <p style="opacity:0.8;">Basé sur le profil de similarité KNN (K-Voisins)</p>
            </div>
        """, unsafe_allow_html=True)

        # Comparaison visuelle : où se situe la prédiction ?
        st.markdown("### Positionnement par rapport au marché")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Niveau de dépense vs Maximum Historique"},
            gauge = {
                'axis': {'range': [None, df_hist['depenses'].max()]},
                'bar': {'color': "#FF4B4B"},
                'steps': [
                    {'range': [0, df_hist['depenses'].mean()], 'color': "#E8F5E9"},
                    {'range': [df_hist['depenses'].mean(), df_hist['depenses'].max()], 'color': "#FFEBEE"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': df_hist['depenses'].mean()
                }
            }
        ))
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("👈 Ajustez les paramètres dans la barre latérale et cliquez sur le bouton pour lancer l'analyse.")

st.caption("Fait par la Division Data Strategy - Confidentiel")