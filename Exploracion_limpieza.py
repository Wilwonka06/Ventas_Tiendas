import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ventas_tiendas (4).csv')
print(df)
print(df.describe(include='all'))
#Exploramos los datos
# print(df.head(20))
print('------------------ Info ---------------')
print(df.info())
# print('-------------------Describe------------')
# print(df.describe())

#vamos a quitar los reguistros con valores nulos
df = df.dropna(how='any')
#verificamos que ya no hay valores nulos
print(df.isnull().sum())

# Mantener solo las filas donde ventas sea mayor o igual a 0
df = df[df["ventas"] >= 0]

#Asignamos 0 y 1 a la ubicacion
df["ubicacion"] = df["ubicacion"].replace({
    "rural": 2,
    "urbana": 1,
    "suburbana": 0
})


#borramos los datos duplicados
df_new = df.drop_duplicates()
df_new.info()

print(df_new.head(20))

# Guardar el DataFrame limpio en un nuevo archivo CSV
df_new.to_csv("ventas_tiendas_limpias.csv", index=False)


