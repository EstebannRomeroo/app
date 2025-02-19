import streamlit as st
import requests
import pandas as pd

# Cargar datos reales de viviendas en Madrid
@st.cache_resource
def load_data():
    file_path = "data_con_distritos.csv"
    df = pd.read_csv(file_path)
    df = df.sample(n=500)
    df_filtered = df[[
        "PRICE", "CONSTRUCTEDAREA", "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING",
        "HASPARKINGSPACE", "DISTRITO", "ROOMNUMBER_RECATEGORIZED", "BATHNUMBER_RECATEGORIZED",
        "CADASTRALQUALITYID_RECATEGORIZED", "LATITUDE", "FLATLOCATIONID_RECATEGORIZED"
    ]]
    df_filtered = df_filtered.rename(columns={
        "PRICE": "Precio (€)",
        "CONSTRUCTEDAREA": "Área (m²)",
        "HASTERRACE": "Tiene Terraza",
        "HASLIFT": "Tiene Ascensor",
        "HASAIRCONDITIONING": "Tiene Aire Acondicionado",
        "HASPARKINGSPACE": "Tiene Parking",
        "DISTRITO": "Distrito",
        "ROOMNUMBER_RECATEGORIZED": "Nº Habitaciones",
        "BATHNUMBER_RECATEGORIZED": "Nº Baños",
        "CADASTRALQUALITYID_RECATEGORIZED": "Calidad Catastral",
        "LATITUDE": "Latitud",
        "FLATLOCATIONID_RECATEGORIZED": "ID Ubicación"
    })
    return df_filtered

df = load_data()

# Función para interactuar con Mistral en Ollama
def chat_mistral(prompt):
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "mistral",
        "prompt": f"Responde de forma breve y directa. Basado en estos datos:\n{df.to_string(index=False)}\nPregunta: {prompt}\nRespuesta:",
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json().get("response", "No pude generar una respuesta.")
        else:
            return f"⚠️ Error en la API ({response.status_code}): {response.text}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Error: No se pudo conectar con Ollama. ¿Está ejecutándose?"

# Interfaz en Streamlit
st.title("🏡 Chatbot de Viviendas en Madrid (Mistral + Ollama)")
st.write("Pregunta sobre viviendas en Madrid y obtén información en tiempo real.")

# Entrada de usuario
pregunta = st.text_input("Escribe tu pregunta:")

if pregunta:
    respuesta = chat_mistral(pregunta)
    st.write("🤖 **Respuesta:**")
    st.write(respuesta)

# Mostrar datos de viviendas
st.write("📊 **Datos de viviendas disponibles**")
st.dataframe(df)

""" 🔍 Prueba preguntas como:

    ¿Cuál es el precio promedio de una vivienda en Chamartín?
    ¿En qué distrito hay más viviendas con terraza?
    ¿Qué calidad catastral tienen los pisos en Centro? """