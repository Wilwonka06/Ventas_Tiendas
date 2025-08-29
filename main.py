from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib 

model = joblib.load('modelo_ventas_tiendas.pkl')
columnas = joblib.load('columnas_modelo.pkl')

app = FastAPI(title="Prediccion precio de Ventas")

# Modelo de entrada
class InputData(BaseModel):
    empleados: float
    publicidad: float
    ubicacion: float
    
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de ventas"}

@app.post("/predict/")
def predict(data: InputData):
    x_new = pd.DataFrame([[data.empleados, data.publicidad, data.ubicacion]], columns=columnas)
    prediccion = model.predict(x_new)
    return {"prediccion": float(prediccion[0])}