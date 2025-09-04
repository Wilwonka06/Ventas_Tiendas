import pandas as pd
from sklearn.linear_model import LogisticRegression
from sympy import prod
import joblib 

df = pd.read_csv("falla_frenos_limpios.csv")

# Variables independientes (X)
X = df[[
    "kms_recorridos",
    "años_uso",
    "ultima_revision",
    "temperatura_frenos",
    "cambios_pastillas",
    "estilo_conduccion",
    "carga_promedio",
    "luz_alarma_freno"
]]

# Variable dependiente (Y)
Y = df["falla_frenos"]

#modelos
modelo = LogisticRegression()
modelo.fit(X, Y)

#prediccion
prediccion = modelo.predict(X)

#comparar esperado vs predicho
for i, fila in df.iterrows():
    esperado = "no falla" if fila["falla_frenos"] == 1 else "falla"
    predicho = "no falla" if prediccion[i] == 1 else "falla"
    print(f"Carro {i+1}: Esperado: {esperado} | Predicho: {predicho}")

# Evaluar precisión
precision = modelo.score(X, Y)
print(f"Precisión del modelo: {precision*100:.1f}%")

# Implementación con un nuevo carro
nuevo_carro=[[80000, 1, 2, 35, 1, 0, 200, 1]] 
prediccion_nuevo = modelo.predict(nuevo_carro)
pred = modelo.predict(nuevo_carro)
prob = modelo.predict_proba(nuevo_carro)

print("|------------Prediccion de Nuevo Carro------------|")
print("|                                                 |")
if pred[0] == 0:
    print("| Predicción de falla de frenos: NO falla 🚗✅    |")
else:
    print("| Predicción de falla de frenos: SÍ falla 🚨      |")
print("|                                                 |")
print("|-------------------------------------------------|")

# resultado = "no falla" if prediccion_nuevo[0] == 0 else "falla"
# print(f"Nuevo carro: Predicción de falla de frenos: {resultado}")
print("")
print("")
print("                   /\\                       ___   ")
print("                  /**\\                  .-'   `'.")
print("                 /****\\                /         \\")
print("                /******\\              |           |")
print("               /********\\              \\        .'")
print("              /**********\\              `-.__.-'   ")
print("                  ||")
print("                  ||")
print("======================================================")
print("       ______                ||              ")
print("      /|_||_\\`.__            ||            ")
print("     (   _    _ _\\           ||            ")
print("     =`-(_)--(_)-'           ||            ")
print("=======================================================")

# Guardar el modelo y las columnas
joblib.dump(modelo, 'modelo_falla_frenos.pkl')
joblib.dump(X.columns.tolist(), 'columnas_modelo_falla_frenos.pkl')



