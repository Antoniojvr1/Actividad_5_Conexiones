import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Detector de Conexiones Sospechosas",
    page_icon="🛡️",
    layout="centered"
)

# Título y descripción
st.title("🛡️ Clasificador de Tráfico de Red")
st.markdown("""
Esta aplicación web utiliza un modelo de Machine Learning (**SVM con Kernel RBF**)
para clasificar en tiempo real si el estado de una conexión de red es **Normal** o **Sospechosa**
basándose en su nivel de tráfico y latencia.
""")

# Función para cargar el modelo de forma eficiente
@st.cache_resource
def cargar_modelo():
    return joblib.load('mejor_modelo_svm_rbf.joblib')

try:
    modelo = cargar_modelo()
    st.success("✅ Modelo predictivo cargado correctamente.")
except Exception as e:
    st.error("❌ No se pudo cargar el archivo 'mejor_modelo_svm_rbf.joblib'. Asegúrate de subirlo junto a este script.")
    st.stop()

# Formulario de entrada de datos
st.subheader("📊 Ingresa los parámetros de la conexión")

col1, col2 = st.columns(2)

with col1:
    trafico = st.number_input(
        "Tráfico de red (Mbps)", 
        min_value=0.0, 
        max_value=500.0, 
        value=50.0, 
        step=1.0,
        help="Velocidad de transferencia de datos en Megabits por segundo."
    )

with col2:
    latencia = st.number_input(
        "Latencia de red (ms)", 
        min_value=0.0, 
        max_value=500.0, 
        value=70.0, 
        step=1.0,
        help="Tiempo de respuesta de la conexión en milisegundos."
    )

# Botón para realizar la predicción
if st.button("🔮 Analizar Conexión", type="primary"):
    # Crear DataFrame para mantener la estructura y nombres de columnas con las que se entrenó
    datos_entrada = pd.DataFrame({
        "trafico_mbps": [trafico],
        "latencia_ms": [latencia]
    })
    
    # Realizar predicción
    prediccion = modelo.predict(datos_entrada)[0]
    
    st.write("--- ")
    st.subheader("Resultado del Análisis:")
    
    if prediccion == 1:
        st.error("🚨 **CONEXIÓN SOSPECHOSA (Clase 1)**")
        st.warning("Se recomienda verificar el origen de esta conexión debido a un comportamiento anómalo en el tráfico o latencia.")
    else:
        st.success("🟢 **CONEXIÓN NORMAL (Clase 0)**")
        st.info("La conexión presenta métricas estables dentro de los parámetros esperados.")
