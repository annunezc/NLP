
import streamlit as st
import pandas as pd
import joblib

# Cargar modelo
model = joblib.load('../models/modelo_triage.pkl')

# Diccionario para codificar el motivo
motivo_map = {
    'Dolor torácico': 0, 'Fiebre': 1, 'Trauma': 2, 'Dolor abdominal': 3,
    'Cefalea': 4, 'Disnea': 5, 'Contusión': 6, 'Dolor lumbar': 7,
    'Vómitos': 8, 'Fractura': 9
}

# Título
st.title("🔎 Clasificador de Triage ESI")
st.write("Introduce los datos del paciente para obtener la clasificación de triage y la confianza del modelo.")

# Inputs
edad = st.slider("Edad", 0, 100, 30)
fc = st.slider("Frecuencia Cardíaca", 40, 180, 90)
pa = st.slider("Presión Arterial Sistólica", 60, 180, 110)
fr = st.slider("Frecuencia Respiratoria", 10, 40, 20)
sato2 = st.slider("Saturación de O2 (%)", 70, 100, 95)
glasgow = st.slider("Escala de Glasgow", 3, 15, 15)
dolor = st.slider("Dolor (0-10)", 0, 10, 5)
motivo = st.selectbox("Motivo de Consulta", list(motivo_map.keys()))
riesgo_vital = st.radio("¿Signos de Riesgo Vital?", ["Sí", "No"])
intervencion = st.radio("¿Requiere Intervención Urgente?", ["Sí", "No"])

# Procesar inputs
if st.button("Clasificar Triage"):
    input_df = pd.DataFrame([{
        'Edad': edad,
        'FC': fc,
        'PA': pa,
        'FR': fr,
        'SatO2': sato2,
        'Glasgow': glasgow,
        'Dolor': dolor,
        'Motivo': motivo_map[motivo],
        'Riesgo Vital': 1 if riesgo_vital == "Sí" else 0,
        'Intervención Urgente': 1 if intervencion == "Sí" else 0
    }])

    # Predicción
    pred = model.predict(input_df)[0]
    proba = max(model.predict_proba(input_df)[0])

    # Mostrar resultado
    st.success(f"Nivel de Triage ESI: {pred}")
    st.info(f"Confianza del modelo: {proba*100:.2f}%")
