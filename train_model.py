from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import joblib
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
best_model_path = os.path.join(BASE_DIR, 'mejor_modelo.pkl')

# Cargar los datos
data = pd.read_csv(r'C:\Users\daniela\Music\Housing.csv')

# Limpieza de datos
data.dropna(inplace=True)

# Convertir las columnas categóricas en valores numéricos usando LabelEncoder
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# Seleccionar todas las columnas, excepto la columna objetivo 'price'
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 
          'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = data['price']

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar el escalador
joblib.dump(scaler, scaler_path)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Modelo 1: Bosque Aleatorio
param_grid_rf = {
    'n_estimators': [20, 50],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
rf = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, y_train)

y_pred_rf = grid_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Bosque Aleatorio: MSE={mse_rf:.2f}, R2={r2_rf:.2f}")

# Modelo 2: K-Nearest Neighbors
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print(f"KNN: MSE={mse_knn:.2f}, R2={r2_knn:.2f}")

# Modelo 3: Support Vector Regressor
svc = SVR(kernel='linear')
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)
mse_svc = mean_squared_error(y_test, y_pred_svc)
r2_svc = r2_score(y_test, y_pred_svc)
print(f"SVC: MSE={mse_svc:.2f}, R2={r2_svc:.2f}")

# Comparar modelos
models = {
    'Bosque Aleatorio': (grid_rf.best_estimator_, mse_rf, r2_rf),
    'KNN': (knn, mse_knn, r2_knn),
    'SVC': (svc, mse_svc, r2_svc)
}

best_model = None
best_mse = float('inf')
best_model_name = ""

for model_name, (model, mse, _) in models.items():
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_model_name = model_name

# Guardar el mejor modelo
joblib.dump(best_model, best_model_path)
print(f"El mejor modelo es {best_model_name} con MSE={best_mse:.2f}")

# Evaluar el mejor modelo
print(f"\nEvaluando el mejor modelo: {best_model_name}")

# Obtener predicciones del mejor modelo
if best_model_name == 'Bosque Aleatorio':
    y_pred_best = grid_rf.best_estimator_.predict(X_test)
elif best_model_name == 'KNN':
    y_pred_best = knn.predict(X_test)
elif best_model_name == 'SVC':
    y_pred_best = svc.predict(X_test)

# Evaluación para regresión (MSE, R², MAE)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)

# Mostrar métricas de regresión
print(f"MSE: {mse_best:.2f}")
print(f"R²: {r2_best:.2f}")
print(f"MAE: {mae_best:.2f}")

# Mensaje final de proceso completado
print("Proceso completado exitosamente.")
