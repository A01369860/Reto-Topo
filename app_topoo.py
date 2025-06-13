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


# Cargar archivos entrenados
scaler = load("scaler.joblib")
projector = load("umap_projector.joblib")
X_projected = np.load("X_projected.npy")
df_filtered_modelo = pd.read_csv("df_filtered_modelo.csv")

# Cargar archivos de homología (opcional)
diagramas = np.load("tda_diagramas.npy", allow_pickle=True)
#df_avg2 = pd.read_csv("df_avg2.csv", parse_dates=['Fecha Transacción'])
# Leer el CSV de la serie de tiempo

df_avg = pd.read_csv("df_avg_serie_tiempo.csv")

# Convertir la columna de fechas al formato correcto
df_avg['Fecha Transacción'] = pd.to_datetime(df_avg['Fecha Transacción'], dayfirst=True, errors='coerce')
df_avg.set_index('Fecha Transacción', inplace=True)
df_avg.sort_index(inplace=True)

# Función de predicción

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
    "TDA y Homología",
    "Predicción",
    "Conclusiones",
])

tab1, tab2, tab3, tab4, tab5, tab6= tabs

# Pestaña 1: Exploración
with tab1:
    st.title("Beneficios del Proyecto")
with tab2:
    st.title("Exploración del Dataset")
    st.dataframe(df_filtered_modelo.sample(10))
    st.subheader("Distribución de emisiones (TON C02)")
    st.plotly_chart(px.histogram(df_filtered_modelo, x="TON C02", nbins=30))
    st.subheader("Relación entre rendimiento y emisiones")
    st.plotly_chart(px.scatter(df_filtered_modelo, x="Rendimiento_kmpl", y="TON C02", color="Placa"))
    st.subheader("Distribución de variables numéricas")
    var = st.selectbox("Selecciona una variable", df_filtered_modelo.select_dtypes(include=np.number).columns)
    st.plotly_chart(px.histogram(df_filtered_modelo, x=var, nbins=30))


# Pestaña 3: UMAP
with tab3:
    st.title("Visualización del espacio UMAP")
    st.write("Espacio proyectado de los datos con color por emisiones")
    st.plotly_chart(px.scatter(
        x=X_projected[:, 0], y=X_projected[:, 1],
        color=df_filtered_modelo["TON C02"],
        labels={"x": "UMAP 1", "y": "UMAP 2"},
        title="Espacio UMAP"
    ))

# Pestaña 4: TDA y Homología
with tab4:
    st.title("Análisis topológico (TDA)")

    st.subheader("1. Gráfico Mapper")
    try:
        with open("mapper_Isis.html", "r", encoding="utf-8") as f:
            html_string = f.read()
        #st.write(html_string)
        components.html(html_string, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("No se encontró el archivo 'mapper_Isis.html'.")

    st.subheader("2. Serie de tiempo topológica (imagen)")

    # Mostrar la imagen
    # Mostrar la imagen
    st.image("tiempo.png", caption="Evolución de variables topológicas", use_container_width=True)


    st.subheader("3. Análisis de grupos (bengalas)")


    #mostrar_bengalas = st.checkbox("Mostrar grupos identificados")
    #if mostrar_bengalas:
    #    df_bengalas = pd.read_csv("df_filtered_bengalas.csv")
    #    colores = df_bengalas['Grupo'].map({
    #        'Resto': 'lightgray', 'Bengala_1': 'red', 'Bengala_2': 'green',
    #        'Bengala_3': 'blue', 'Bengala_4': 'orange'
    #    })
    #    st.plotly_chart(px.scatter(
    #        x=X_projected[:, 0], y=X_projected[:, 1], color=colores,
    #        labels={"x": "UMAP 1", "y": "UMAP 2"}, title="Grupos en el espacio UMAP"
    #    ))

# Pestaña 5: Conclusiones
# Pestaña 2: Predicción
with tab5:
    st.title("Predicción personalizada de emisiones")
    recorrido = st.number_input("Recorrido (km)", min_value=0.0, value=100.0)
    placa = st.number_input("Placa (codificada)", min_value=0.0, value=5.0)
    rendimiento = st.number_input("Rendimiento km/L", min_value=0.1, value=2.0)
    k = st.slider("Vecinos para predicción (k)", min_value=1, max_value=10, value=10)

    if st.button("Predecir"):
        nuevo_x = np.array([[recorrido, placa, rendimiento]])
        resultado = prediccion_regresion_mapper(nuevo_x, df_filtered_modelo, scaler, projector, X_projected, k)
        st.metric("TON C02 Estimado", f"{resultado['ton_co2_estimado']:.4f}")

    st.subheader("Desempeño general del modelo")
    metricas = pd.DataFrame({
        'Métrica': ['R²', 'MAE', 'RMSE'],
        'Valor': [0.82, 0.049, 0.110]
    })
    st.table(metricas)
    
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

# Sidebar
st.sidebar.header("Resumen del Proyecto")
st.sidebar.info("""
Análisis de emisiones de CO₂ con Machine Learning y Topological Data Analysis (TDA).
""")
st.sidebar.markdown("Creado por: Maritza, José David, Andrea Renata, Máximo, Isis, Génesis\nJunio 2025")

