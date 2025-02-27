import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import streamlit.components.v1 as components
import folium
from streamlit_folium import folium_static
import numpy as np
import os
import json
import pandas as pd
from mistralai import Mistral
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM


# Cargar el modelo de valoración
modelo = joblib.load('stacking_model.joblib')

# Cargar datos reales de viviendas en Madrid
@st.cache_resource
def load_data():
    file_path = "data_con_distritos.csv"
    df = pd.read_csv(file_path)
    df = df.sample(n=550)
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
    api_key = "bCOJ2FX4PF8aFclQYBt9T6voTAe1CFm5"
    model = "mistral-small-latest"

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": f"Responde de forma breve y directa. Basado en estos datos:\n{df.to_string(index=False)}\nPregunta: {prompt}\nRespuesta:"
            },
        ]
    )
    return chat_response.choices[0].message.content

def transformar_datos():
    return np.array([[ constructedarea, distance_to_city_center, roomnumber_recategorized, bathnumber_recategorized,distance_to_castellana]])


# Importar anuncios
ruta_archivo_anuncios = "anuncios.json"

# Verificar si el archivo existe y cargar los anuncios
if os.path.exists(ruta_archivo_anuncios):
    with open(ruta_archivo_anuncios, "r") as f:
        anuncios = json.load(f)


# Diccionario con coordenadas aproximadas de barrios en Madrid
coordenadas_barrios = {
    "Centro": (40.4168, -3.7038),
    "Arganzuela": (40.4002, -3.6957),
    "Puente de Vallecas": (40.3890, -3.6629),
    "Ciudad Lineal": (40.4456, -3.6516),
    "Salamanca": (40.4262, -3.6865),
    "Chamberí": (40.4344, -3.7038),
    "Moncloa - Aravaca": (40.4355, -3.7312),
    "Retiro": (40.4113, -3.6823),
    "Hortaleza": (40.4746, -3.6417),
    "Chamartín": (40.4629, -3.6763),
    "Tetuán": (40.4591, -3.6957),
    "Villa de Vallecas": (40.3713, -3.6010),
    "Fuencarral - El Pardo": (40.5120, -3.7422),
    "Latina": (40.4053, -3.7451),
    "Carabanchel": (40.3813, -3.7339),
    "Moratalaz": (40.4070, -3.6450),
    "Usera": (40.3826, -3.7098),
    "Villaverde": (40.3469, -3.7101),
    "Vicálvaro": (40.3986, -3.6016),
    "San Blas - Canillejas": (40.4396, -3.6161),
    "Barajas": (40.4736, -3.5803)
}

def obtener_coordenadas(barrio):
    """Devuelve las coordenadas del barrio si existe en el diccionario, de lo contrario devuelve el centro de Madrid."""
    return coordenadas_barrios.get(barrio, (40.4168, -3.7038))  # Centro de Madrid por defecto


# Estilos CSS y carga de Font Awesome
st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    
    <style>
    body {
        background-color: #fa8072;  /* Fondo salmón claro */
    }
    .top-buttons {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        padding: 10px;
    }
    .register-button, .login-button {
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        font-weight: bold;
        color: white;
        cursor: pointer;
    }
    .register-button {
        background-color: #fa8072;  /* Fondo salmón claro */
    }
    .login-button {
        background-color: #b2443a;  /* Fondo rojo oscuro */
    }
    .register-button:hover {
        background-color: #f28760;  /* Hover efecto */
    }
    .login-button:hover {
        background-color: #8b1d2c;  /* Hover efecto */
    }
    .title-icon {
        display: inline-block;
        width: 60px;
        height: 60px;
        background-color: #fa8072;  /* Fondo salmón claro en el centro del recuadro */
        border: 3px solid #b2443a;
        border-radius: 10px;
        text-align: center;
        line-height: 60px;
        font-size: 28px;
        color: white;
        margin-right: 10px;
    }
    .title-text {
        display: inline-block;
        vertical-align: middle;
        font-size: 36px;
        font-weight: bold;
        color: black;
    }
    .icon-box {
        display: inline-block;
        width: 40px;
        height: 40px;
        background-color: transparent;
        text-align: center;
        line-height: 40px;
        font-size: 22px;
        color: #b2443a;
        margin-right: 10px;
    }
    .property-info {
        font-size: 18px;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True
)

# Agregar los botones de "Register" y "Log In" en la parte superior derecha
st.markdown(
    """
    <div class="top-buttons">
        <button class="register-button">Register</button>
        <button class="login-button">Log In</button>
    </div>
    """, unsafe_allow_html=True
)



# Menú de navegación
selected = option_menu(
    menu_title=None,
    options=["Home", "Map", "Property Valuation", "AI Assistant", "Profile"],
    icons=["house", "map", "search", "robot", "person-circle"],
    menu_icon="menu-button-fill",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0", "background-color": "#fa8072"},  # Fondo salmón claro
        "nav-link": {
            "font-size": "18px",
            "color": "black",
            "font-weight": "normal",
            "text-align": "center",
        },
        "nav-link-selected": {
            "font-weight": "bold",
            "background-color": "#b2443a",  # Rojo oscuro cuando está seleccionada
            "color": "white",
            "text-align": "center",
        }
    }
)



if selected == "Home":
    st.markdown("""
    <div>
        <span class="title-icon"><i class="fas fa-home"></i></span>
        <span class="title-text">Featured Property Listings</span>
    </div>
    """, unsafe_allow_html=True)

    for index,anuncio in enumerate(anuncios):
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(anuncio["main_photo"], use_container_width=True)
            with col2:
                st.subheader(anuncio["title"])
               
                # Botón para seleccionar la ubicación y cambiar automáticamente a la pestaña "Map"

                if st.button('📍 View location of the property in the map' + '(' + f"{anuncio['location']}" +')', key=f"loc_{anuncio['location']}_{index}"):
                    st.session_state.selected_location = anuncio["location"]
                    st.rerun()  # Refrescar la app para cambiar de pestaña automáticamente

                # Detalles de la propiedad
                st.markdown(f"""
                <div class="property-info">
                    <div><span class="icon-box"><i class="fas fa-euro-sign"></i></span><strong>Price:</strong> {anuncio['price']}</div>
                    <div><span class="icon-box"><i class="fas fa-home"></i></span><strong>Area:</strong> {anuncio['area']}</div>
                    <div><span class="icon-box"><i class="fas fa-bed"></i></span><strong>Bedrooms:</strong> {anuncio['bedrooms']}</div>
                    <div><span class="icon-box"><i class="fas fa-bath"></i></span><strong>Bathrooms:</strong> {anuncio['bathrooms']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Si se ha seleccionado una ubicación, cambiar automáticamente a la pestaña "Map"
    if "selected_location" in st.session_state and st.session_state.selected_location:
        selected = "Map"
        st.rerun()  # Volver a cargar la app en la pestaña Map

elif selected == "Map":
    st.markdown("""
    <div>
        <span class="title-icon"><i class="fas fa-map-marked-alt"></i></span>
        <span class="title-text">Property Map</span>
    </div>
    """, unsafe_allow_html=True)

    st.write("Map of properties in Madrid")

    # Si hay una ubicación seleccionada en Home, mostrar el mapa en ese barrio
    if "selected_location" in st.session_state and st.session_state.selected_location:
        barrio = st.session_state.selected_location
        lat, lon = obtener_coordenadas(barrio)
        st.write(f"📍 Showing location: {barrio}")

        # Crear un mapa con Folium centrado en la ubicación seleccionada
        m = folium.Map(location=[lat, lon], zoom_start=14)
        folium.Marker([lat, lon], tooltip=barrio, icon=folium.Icon(color="red")).add_to(m)
        folium_static(m)

        # Botón para volver al mapa completo
        if st.button("🔄 Ver mapa completo de la ciudad"):
            del st.session_state.selected_location  # Eliminar la selección previa
            st.experimental_rerun()  # Refrescar la app para mostrar el mapa general

    else:
        # Mostrar el mapa interactivo general
        try:
            with open("interactive_map.html", "r") as file:
                map_html = file.read()
            st.components.v1.html(map_html, height=600, scrolling=True)
        except FileNotFoundError:
            st.error("El archivo del mapa interactivo no se encontró. Verifica la ruta.")

elif selected == "AI Assistant":
    st.markdown("""
    <div>
        <span class="title-icon"><i class="fas fa-robot"></i></span>
        <span class="title-text">AI Assistant</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("💬 Ask me anything about real estate properties!")

    # Entrada del usuario
    pregunta = st.text_input("Type your question here:")

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
    ¿Qué calidad catastral tienen los pisos en Centro? 
    """

elif selected == "Property Valuation":
    st.markdown("""
    <div>
        <span class="title-icon"><i class="fas fa-chart-line"></i></span>
        <span class="title-text">Property Valuation</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("### Enter the property details:")


    constructedarea = st.number_input("Square meters:", min_value=21, value=21, max_value=175)
    distance_to_city_center = st.number_input("Distance to city center (kms):", min_value=0, value=0, max_value = 50)
    roomnumber_recategorized = st.number_input("Number of rooms:",min_value = 1, max_value= 7,value= 1)
    bathnumber_recategorized =  st.number_input("Number of baths:", min_value = 1 ,max_value = 5, value = 1)
    distance_to_castellana = st.number_input("Distance of points of interest(kms)", min_value=0, value=0, max_value = 50)

    if st.button("Calcular Precio Estimado"):
        datos = transformar_datos()
        prediccion = modelo.predict(datos)[0]
        st.markdown(
            f"<div style='font-size: 24px; color: black; text-align: center;'>El valor estimado de la vivienda es: {prediccion:,.2f} €</div>",
            unsafe_allow_html=True
        )

elif selected == "Profile":
    st.markdown("""
    <div>
        <span class="title-icon"><i class="fas fa-user"></i></span>
        <span class="title-text">Profile</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("Welcome to your profile page. Here you can manage your information.")





