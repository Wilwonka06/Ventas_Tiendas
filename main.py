import pandas as pd
from sklearn.linear_model import LogisticRegression

# Datos
datos= [
    {"asistencia": 80, "tareas": 3, "examen": 3.5, "aprueba": "si"},
    {"asistencia": 70, "tareas": 2, "examen": 4.0, "aprueba": "si"},
    {"asistencia": 60, "tareas": 1, "examen": 2.5, "aprueba": "no"},
    {"asistencia": 50, "tareas": 1, "examen": 1.5, "aprueba": "no"},
    {"asistencia": 80, "tareas": 0, "examen": 1.0, "aprueba": "no"},
    {"asistencia": 30, "tareas": 1, "examen": 3.4, "aprueba": "no"},
]

df = pd.DataFrame(datos)
df["prueba"]= df["aprueba"].map({"si":1, "no":0})

X = df[["asistencia", "tareas", "examen"]]  # variables independientes
Y = df["prueba"]  # variable dependiente

# Modelo
modelo = LogisticRegression()
modelo.fit(X, Y)

# Predicciones
predicciones = modelo.predict(X)

# Comparar esperado vs predicho
for i, fila in df.iterrows():
    esperado = "si" if fila["prueba"] == 1 else "no"
    predicho = "si" if predicciones[i] == 1 else "no"
    print(f"Estudiante {i+1}: Esperado: {esperado} | Predicho: {predicho}")
    
# Evaluar precisión
precision = modelo.score(X, Y)
print(f"Precisión del modelo: {precision}")

from sklearn.metrics import accuracy_score
precision2 = accuracy_score(Y, predicciones)
print(f"Precisión del modelo (accuracy_score): {precision2*100:.1f}%")

#implementacion 
nuevo_estudiante=[[75, 2, 3.0]]
prediccion= modelo.predict(nuevo_estudiante)[0]
print(f"El nuevo estudiante va a aprobar {'Aprueba' if prediccion == 1 else 'No aprueba'}")