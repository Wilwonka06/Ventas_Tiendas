from fastapi import FastAPI
import pandas as pd
import joblib 

model = joblib.load('modelo_ventas_tiendas.pkl')
columnas = joblib.load('columnas_modelo.pkl')

app = FastAPI(title="Prediccion precio de Ventas")

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de ventas"}

@app.post("/predict/")
def predict(empleados: float, publicidad: float, ubicacion: float):
    x_new = pd.DataFrame([[empleados, publicidad, ubicacion]], columns=columnas)
    prediccion = model.predict(x_new)
    return {"prediccion": prediccion[0]}