import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from joblib import load
from sklearn.neighbors import NearestNeighbors
import kmapper as km
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from persim import plot_diagrams
import umap
import io
import base64
#from gtda.homology import VietorisRipsPersistence
#from gtda.diagrams import PersistenceEntropy
#from gtda.plotting import plot_diagram
import plotly.express as px
import pickle


#Modelo1
with open("scaler_modelo1.pkl", "rb") as f:
    scaler_modelo1 = pickle.load(f)
with open("projector_modelo1.pkl", "rb") as f:
    projector_modelo1 = pickle.load(f)
with open("X_projected_modelo1.pkl", "rb") as f:
    X_projected_modelo1 = pickle.load(f)
    
#Modelo2
with open("scaler_modelo2.pkl", "rb") as f:
    scaler_modelo2 = pickle.load(f)
with open("projector_modelo2.pkl", "rb") as f:
    projector_modelo2 = pickle.load(f)
with open("X_projected_modelo2.pkl", "rb") as f:
    X_projected_modelo2 = pickle.load(f)
    
df_nn = pd.read_csv("df_nn.csv")
X_projected = np.load("X_projected2.npy")
df_filtered = pd.read_csv("df_filtered.csv")
df_filtered_modelo = pd.read_csv("df_filtered_modelo.csv")
bengalas = pd.read_csv("bengalas.csv")
df_modelo1 = pd.read_csv("df_modelo1.csv")
df_modelo2 = pd.read_csv("df_modelo2.csv")

features = ["Recorrido", "Rendimiento_kmpl", "TON C02"]
scaler = StandardScaler()
df_norm = df_nn.copy()
df_norm[features] = scaler.fit_transform(df_nn[features])

#Función de predicción
def prediccion_regresion_mapper(nuevo_x, df_filtered, scaler, projector, X_projected, k=10):
    x_scaled = scaler.transform(nuevo_x)
    x_umap = projector.transform(x_scaled)

    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(X_projected)
    distancias, indices_vecinos = neigh.kneighbors(x_umap)

    distancias = distancias[0]
    indices_vecinos = indices_vecinos[0]
    mascara = distancias > 1e-6
    distancias_filtradas = distancias[mascara][:k]
    indices_filtrados = indices_vecinos[mascara][:k]

    pesos = 1 / (distancias_filtradas + 1e-6)
    rendimiento_vecinos = df_filtered.iloc[indices_filtrados]["TON C02"].values
    prediccion = np.average(rendimiento_vecinos, weights=pesos)

    return {
        "vecinos_usados": indices_filtrados.tolist(),
        "ton_co2_estimado": prediccion
    }

# Tabs principales
tabs = st.tabs([
    "Introducción",
    "Exploración",
    "UMAP y bengalas",
    "Predicción",
    "Conclusiones",
])

tab1, tab2, tab3, tab5, tab6= tabs
with tab1:
    st.title("Predicción de emisiones de CO₂ con Machine Learning y Topological Data Analysis (TDA).")
    st.info("Maritza Barrios | José Banda | Renata Garfias | Máximo Caballero | Isis Malfavón | Génesis Pereyra")
    st.subheader("Beneficios del Proyecto")
    st.markdown("""
    - **Detección temprana de anomalías:** antes de que generen impactos mayores.
    - **Identificación de áreas de oportunidad:** localizar comportamientos inusuales y errores sistemáticos que generen emisiones altas de CO₂.
    - **Planificación proactiva:** anticipación de demanda futura y estrategia para disminuir las emisiones de CO₂.
    """)

    # Crear 2 columnas
    col1, col2 = st.columns(2)

    # Imagen en la columna 1
    with col1:
        st.image("img1.jpeg", use_container_width=True)

    # Imagen en la columna 2
    with col2:
        st.image("img2.jpeg", use_container_width=True)


with tab2:
    st.header("Descripción de Variables")
    # Lista de variables
    variables = [
        {
            "nombre": "Recorrido",
            "descripcion": "Distancia total recorrida por el vehículo durante la transacción (en kilómetros)."
        },
        {
            "nombre": "Placa",
            "descripcion": "Matrícula única que identifica al vehículo (convertida a numérica/categórica)."
        },
        {
            "nombre": "Toneladas de CO₂",
            "descripcion": "Emisiones estimadas de dióxido de carbono generadas por el consumo de diésel en la transacción (en toneladas)."
        },
        {
            "nombre": "Rendimiento KMPL",
            "descripcion": "Rendimiento de combustible (Km por litro).\n *Variable creada*."
        }
    ]

    # Mostrar como tarjetas bonitas en 2 columnas
    for i in range(0, len(variables), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(variables):
                var = variables[i + j]
                with cols[j]:
                    st.subheader(f"🔸 {var['nombre']}")
                    st.markdown(var['descripcion'], unsafe_allow_html=True)
    
    st.subheader("Exploración del Dataset")
    st.dataframe(df_nn.tail(5))
    
    # 1️⃣ Recorrido vs TON C02
    fig1 = px.scatter(df_norm, 
                    x="Recorrido", 
                    y="TON C02", 
                    title="Recorrido vs TON C02")
    st.plotly_chart(fig1)
    
    # 2️⃣ Rendimiento_kmpl vs TON C02
    fig2 = px.scatter(df_norm, 
                    x="Rendimiento_kmpl", 
                    y="TON C02", 
                    title="Rendimiento vs TON C02")
    st.plotly_chart(fig2)
    
    # Scatter plot
    fig3 = px.scatter(df_nn, x="Placa", y="TON C02",
                    labels={"x": "Placa", "y": "Toneladas de CO₂"},
                    title="Relación de Placa vs TON C02")
    st.plotly_chart(fig3)
    
# Pestaña 3: UMAP
with tab3:
    st.title("Visualización del espacio UMAP")
    #st.write("Espacio proyectado de los datos con color por emisiones")
    st.plotly_chart(px.scatter(
        x=X_projected[:, 0], y=X_projected[:, 1],
        color=df_filtered_modelo["TON C02"],
        labels={"x": "UMAP 1", "y": "UMAP 2"}
    ))
    st.title("Visualización de Bengalas en UMAP")
    #st.write("Espacio proyectado de los datos con etiquetas manuales de Bengalas")

    # Creamos el scatter plot interactivo
    fig_bengalas = px.scatter(
        x=X_projected[:, 0],
        y=X_projected[:, 1],
        color=bengalas['Grupo'],
        color_discrete_map={
            'Resto': 'lightgray',
            'Bengala_1': 'red',
            'Bengala_2': 'green',
            'Bengala_3': 'blue',
            'Bengala_4': 'orange'
        },
        labels={"x": "UMAP 1", "y": "UMAP 2"}
    )

    st.plotly_chart(fig_bengalas)
    st.title("Segmentación de Bengalas")
    st.subheader("Creación de nuevo dataframe para predicción")
    #subset = bengalas[10:15,:]
    #df_subset = pd.DataFrame(subset, columns=bengalas.columns)
    bengalas_filtrado = bengalas.drop(columns=["Id Mercancía"])
    bengalas_filtrado = bengalas_filtrado[bengalas_filtrado['Recorrido'] > 0]
    st.dataframe(bengalas_filtrado.tail())
    st.info("Se implementaron técnicas topológicas adicionales como la entropía de persistencia y la homología; sin embargo, no se presentan, ya que no aportaron mejoras significativas a los resultados.")

# Pestaña 5: Predicción
with tab5:
    st.title("Predicción personalizada de emisiones")
    bengalas_top5 = bengalas_filtrado[bengalas_filtrado['Grupo'].isin(['Bengala_1', 'Bengala_2', 'Bengala_3', 'Bengala_4', 'Resto'])]
    bengalas_muestra = bengalas_top5.groupby('Grupo').sample(1, random_state=123).reset_index(drop=True)
    st.dataframe(bengalas_muestra)
    #bengalas.drop(columns="Id Mercancía")
    # Crear columnas de entrada
    st.subheader("Datos de entrada")

    col1, col2 = st.columns(2)

    with col1:
        recorrido = st.number_input("Recorrido (km)", min_value=0.0, value=400.0, step=1.0)
        placa = st.number_input("Placa (codificada)", min_value=0.0, value=592.0, step=1.0)
        rendimiento = st.number_input("Rendimiento km/L", min_value=0.0, value=10000.0, step=1.0)
        
        # NUEVA VARIABLE CATEGORICA (para modelo 2)
        categorias = ['Bengala_1', 'Bengala_2', 'Bengala_3', 'Bengala_4', 'Resto']
        nueva_variable = st.selectbox("Grupo (modelo 2)", categorias)

        # Creamos el mapping manual igual al LabelEncoder
        mapa_categorias = {
            'Bengala_1': 0,
            'Bengala_2': 1,
            'Bengala_3': 2,
            'Bengala_4': 3,
            'Resto': 4
        }
        categoria_codificada = mapa_categorias[nueva_variable]

    with col2:
        k1 = st.slider("Vecinos para M1 (k)", min_value=1, max_value=10, value=10)
        k2 = st.slider("Vecinos para M2 (k)", min_value=1, max_value=10, value=10)
    nuevo_df = pd.DataFrame([{
        "Recorrido": recorrido,
        "Placa": placa,
        "Rendimiento_kmpl": rendimiento,
        "Grupo": categoria_codificada
    }])

    # Botón único para predecir ambos modelos
    if st.button("Predecir ambos modelos"):
        nuevo_x = np.array([[recorrido, placa, rendimiento]])
        nuevo_x_m2 = nuevo_df[["Recorrido", "Placa", "Rendimiento_kmpl", "Grupo"]].values

        resultado_m1 = prediccion_regresion_mapper(nuevo_x, df_modelo1, scaler_modelo1, projector_modelo1, X_projected_modelo1, k1)
        resultado_m2 = prediccion_regresion_mapper(nuevo_x_m2, df_modelo2, scaler_modelo2, projector_modelo2, X_projected_modelo2, k2)

        # Mostrar resultados
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Modelo 1")
            st.metric("TON C02 Estimado", f"{resultado_m1['ton_co2_estimado']:.4f}")

        with col4:
            st.subheader("Modelo 2")
            st.metric("TON C02 Estimado", f"{resultado_m2['ton_co2_estimado']:.4f}")
    st.subheader("Visualización entre predicciones de C0₂ y datos originales")

with tab6:
    st.title("Conclusiones del Proyecto")
    st.markdown("""
    - Las emisiones de CO₂ varían significativamente según el rendimiento y tipo de unidad.
    - El modelo predictivo basado en UMAP y vecinos obtiene buen desempeño (R² ≈ 0.82).
    - La homología persistente permite observar comportamientos estructurales atípicos.
    - Las "bengalas" indican subgrupos relevantes con alto impacto ambiental.
    - Esta plataforma puede asistir decisiones operativas para reducir emisiones.
    """)
    st.subheader("📊 Comparativa de Modelos de Predicción")

    tabla_modelos = pd.DataFrame({
        "Modelo": [
            "XGBoost",
            "UMAP + K Vecinos",
            "Entropía Homología",
            "Homología + Serie Temporal",
            "Segmentación de Bengalas"
        ],
        "R²": [0.812, 0.8557, 0.8024, 0.8186, 0.8852],
        "MAE": [0.0500, 0.0398, 0.0493, 0.0442, 0.0378],
        "RMSE": [0.109, 0.0944, 0.1105, 0.1059, 0.0833]
    })

    styled_tabla = tabla_modelos.style.format({
        "R²": "{:.4f}",
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}"
    }).highlight_max(axis=0, color="green", subset=["R²"])\
      .highlight_min(axis=0, color="green", subset=["MAE", "RMSE"])

    st.dataframe(styled_tabla, use_container_width=True)

    #st.dataframe(tabla_modelos.style.format({"R²": "{:.2f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}))



