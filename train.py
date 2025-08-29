import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv('ventas_tiendas_limpias.csv')

X = df[['empleados', 'publicidad', 'ubicacion']]
Y = df['ventas']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
print(X_train.head())
print(Y_train.head())

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("MSE:", mse)
print("R2:", r2)

# scores = cross_val_score(model, X, Y, cv=5)
# print("Cross-validation scores:", scores)

#Mostramos la ecuacion
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)
print(f"Ecuacion: ventas = {model.intercept_:.2f}, + {model.coef_[0]:.2f}, * empleados + {model.coef_[1]:.2f}, * publicidad + {model.coef_[2]:.2f}, * ubicacion")

joblib.dump(model, 'modelo_ventas_tiendas.pkl')
joblib.dump(X.columns.tolist(), 'columnas_modelo.pkl')
