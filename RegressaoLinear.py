from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

# carregar o dataframe

df = pd.read_csv('FuelConsumptionCo2.csv')

# exibir a estrutura do dataframe

#print(df.head())

# resumo do dataframe

#print(df.describe())

# selecionar apenas as features do motor e co2
motores = df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]

#print(motores.head())

# dividir o dataset em dados de treino e teste
motores_treino, motores_test, co2_treino, co2_test = train_test_split(motores, co2, test_size=0.2, random_state=42)

#print(motores_treino)

# exibir gráfico
plt.scatter(motores_treino, co2_treino, color='blue')
plt.xlabel("Motor")
plt.ylabel("Emissão CO²")
plt.show()
