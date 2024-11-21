from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar el modelo entrenado
try:
    with open('mejor_modelo.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except (pickle.UnpicklingError, FileNotFoundError):
    import joblib
    model = joblib.load('mejor_modelo.pkl')

# Leer los datos
data = pd.read_csv('C:/Users/daniela/Music/Housing.csv')

# Convertir las variables categóricas 'yes'/'no' a valores binarios (1 y 0)
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['parking'] = data['parking'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 1, 'unfurnished': 0})

# Función para generar el histograma interactivo
def generar_histograma():
    fig = px.histogram(data, x='area', title='Histograma de Características de Área')
    fig.update_layout(title='Histograma de Características')
    return fig.to_html(full_html=False)

# Función para generar la matriz de correlación interactiva
def generar_matriz_correlacion():
    corr_matrix = data[['area', 'bedrooms', 'bathrooms']].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title="Correlación")
    ))
    fig.update_layout(title='Matriz de Correlación', title_x=0.5)
    return fig.to_html(full_html=False)

# Función para generar el gráfico de dispersión interactivo
def generar_dispersion():
    fig = px.scatter(data, x='area', y='price', title='Dispersión entre Área y Precio')
    fig.update_layout(title='Gráfico de Dispersión', title_x=0.5)
    return fig.to_html(full_html=False)

# Función para generar el boxplot interactivo
def generar_boxplot():
    fig = px.box(data, y='price', title='Distribución de los Precios')
    fig.update_layout(title='Boxplot de Precios', title_x=0.5)
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los valores de los inputs del formulario
        area = float(request.form['area'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        stories = float(request.form['stories'])
        mainroad = 1 if request.form['mainroad'] == 'yes' else 0
        guestroom = 1 if request.form['guestroom'] == 'yes' else 0
        basement = 1 if request.form['basement'] == 'yes' else 0
        hotwaterheating = 1 if request.form['hotwaterheating'] == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
        parking = 1 if request.form['parking'] == 'yes' else 0
        prefarea = 1 if request.form['prefarea'] == 'yes' else 0
        furnishingstatus = 1 if request.form['furnishingstatus'] == 'furnished' else 0

        # Organizar las características para la predicción
        features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])

        # Predecir el precio con el modelo
        prediction = model.predict(features)

        # Generar los gráficos
        histograma = generar_histograma()
        matriz = generar_matriz_correlacion()
        dispersion = generar_dispersion()
        boxplot = generar_boxplot()  # Generamos el gráfico de Boxplot

        # Pasar la predicción y los gráficos a la plantilla
        return render_template('results.html', precio=prediction[0], histograma=histograma, matriz=matriz, dispersion=dispersion, boxplot=boxplot)

    except Exception as e:
        return f"Error al predecir el precio: {e}"

if __name__ == '__main__':
    app.run(debug=True)
