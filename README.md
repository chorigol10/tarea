# tarea
import pandas as pd

boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df = pd.read_csv(boston_url)



import pandas as pd
import matplotlib.pyplot as plt

# A partir del código anterior

# Boxplot para "Median value of owner-occupied homes"
plt.figure(figsize=(8, 6))
boston_df['MEDV'].plot(kind='box')
plt.title('Boxplot del Valor Medio de las Viviendas Ocupadas por sus Propietarios')
plt.ylabel('MEDV')
plt.show()

# Gráfico de barras para el río Charles
plt.figure(figsize=(8, 6))
boston_df['CHAS'].value_counts().plot(kind='bar')
plt.title('Distribución de la Variable Río Charles')
plt.xlabel('CHAS')
plt.ylabel('Conteo')
plt.show()

# Boxplot para MEDV vs AGE
boston_df['AGE_group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['35 años o menos', '35-70 años', '70 años o más'])
plt.figure(figsize=(8, 6))
boston_df.boxplot(column='MEDV', by='AGE_group')
plt.title('Boxplot de MEDV vs Grupos de Edad')
plt.suptitle('')  # Elimina el título automático
plt.xlabel('Grupo de Edad')
plt.ylabel('MEDV')
plt.show()

# Gráfico de dispersión entre NOX y INDUS
plt.figure(figsize=(8, 6))
boston_df.plot.scatter(x='NOX', y='INDUS')
plt.title('Relación entre Concentraciones de Óxido Nítrico y Proporción de Acres de Negocios no Minoristas por Ciudad')
plt.xlabel('Concentraciones de Óxido Nítrico (NOX)')
plt.ylabel('Proporción de Acres de Negocios no Minoristas (INDUS)')
plt.show()

# Histograma para PTRATIO
plt.figure(figsize=(8, 6))
boston_df['PTRATIO'].plot(kind='hist', bins=30)
plt.title('Histograma de la Proporción de Alumnos por Maestro')
plt.xlabel('Proporción de Alumnos por Maestro (PTRATIO)')
plt.show()

# t-test

from scipy.stats import ttest_ind

charles_houses = boston_df[boston_df['CHAS'] == 1]['MEDV']
non_charles_houses = boston_df[boston_df['CHAS'] == 0]['MEDV']

t_stat, p_value = ttest_ind(charles_houses, non_charles_houses)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Rechazar la hipótesis nula")
else:
    print("No rechazar la hipótesis nula")

# ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

modelo = ols('MEDV ~ AGE', data=boston_df).fit()
anova_table = sm.stats.anova_lm(modelo, typ=2)

print(anova_table)

#Correlación de Pearson


# Correlación de Pearson

from scipy import stats

correlation, p_value = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])

print("Coeficiente de correlación:", correlation)
print("P-value:", p_value)


#Análisis de regresión

X = boston_df['DIS']
y = boston_df['MEDV']

X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()

print(modelo.summary())
