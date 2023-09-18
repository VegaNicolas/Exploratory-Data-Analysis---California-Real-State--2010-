import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm

# English version WIP

# Se define el path
df_train = pd.read_csv('train.csv')

# Primero mirammos los primeros 20 elementos
# print(df_train.head(20))

# Podemos ver que tiene 81 columnas, pero no las muestra todas, para verlas todas:
# print(df_train.columns)


# Podemos averiguar las medidas (x * y) de la distribución con:
# print(df_train.shape)  # (1460, 81)

# Si se quiere ver una columna determinada, ejemplo Id:
# print(df_train['Id'])  # It only shows the first and last 5 elements

# Pandas maneja una sintaxis de diccionario, por lo que si quiero acceder a los 10 primeros elementos de Id:
# print(df_train['Id'][:10])

# A su vez, se pueden seleccionar más de una columna, usando doble corchetes:
# print(df_train[['Id', 'SalePrice']])  # double brackets

# Medidas de estadística descriptiva
# print(df_train.describe()) # Count, M(x), s(x), Q1, Me(x), Q3, Min, Max for every column

# Para ver la medida resumen, en este caso la media aritmética, de una columna en particular:
# print(df_train['SalePrice'].mean())


"""
La idea del proyecto es determinar cuáles son las variables que más influyen en el precio de venta de una casa.
En este caso, se van a tener en cuenta cuatro variables que estarán divididas en dos grupos que servirán como hipótesis:
Variables de Construcción (Cualitativa ordinal / Cuantitativa discreta)
    -OverallQual: Calidad general
    -YearBuilt: Año de construcción

Variables de Espacio (Cuantitativas continuas)
    -TotalBsmtSF: Tamaño del sótano 
    -GrLivArea: Tamaño de la casa


"""

# Para empezar se debe analizar el SalePrice, puesto que es la variable respuesta

# print(df_train['SalePrice'].describe())

"""
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
"""

# Para graficarlo se utiliza el método distplot de seaborn (DEPRECATED)
# sns.histplot(df_train['SalePrice'])


"""Como se puede ver, la distribución de la variable SalePrice es asimétrica derecha o positiva
se puede predecir debido a que la media es mayor que la mediana (mean > median),
es decir, la mayoría de las casas se venden por debajo de la media.
A su vez también se puede ver que la distribución es puntiaguda o leptocúrtica,
lo que significa que hay una mayor concentración de los datos alrededor de la media."""

# Para ver la asimetría se utiliza el método skewness de scipy.stats
# print(df_train['SalePrice'].skew()) # 1.88 > 0 -> positiva

# En el caso de la kurtosis:
# print(df_train['SalePrice'].kurtosis())  # 6.54 > 0 -> leptocúrtica


# Relaciones Numéricas - OverallQual y BsmtSF / Determinar si hay relación con SalePrice

GrLivArea = 'GrLivArea'

# Concatenamos con SalePrice para hacer una tabla aparte para realizar una tabla de scatterplot

data = pd.concat([df_train['SalePrice'], df_train[GrLivArea]],
                 axis=1)  # axis=1 para concatenar por columnas  / Usar una variable no cambió nada

"""
      SalePrice  GrLivArea
0        208500       1710
1        181500       1262
2        223500       1786
3        140000       1717
4        250000       2198
...         ...        ...
1455     175000       1647
1456     210000       2073
1457     266500       2340
1458     142125       1078
1459     147500       1256
"""

# Con estos datos, se puede hacer el scatterplot

# x = var. indepte. / y = var. depen. / ylim = limites del eje y
# data.plot.scatter(x=GrLivArea, y='SalePrice', ylim=(0, 800000))

"""
Se puede apreciar a simple vista que existe una relación lineal directa entre GrLivArea y SalePrice.
Asi y todo, existen algunos valores atípicos (outliers)
A mayor espacio evitable, mayor precio de venta (Conclusión)
"""

# Repetimos el proceso para TotalBsmtSF


TotalBsmtSF = 'TotalBsmtSF'

# data_bsmt = pd.concat([df_train['SalePrice'], df_train[TotalBsmtSF]], axis=1)

# data_bsmt.plot.scatter(x=TotalBsmtSF, y='SalePrice', ylim=(0, 800000))
# print(data_bsmt)

"""
Si bien existe una relación lineal directa, no es tan evidente o tan fuerte como la de GrLivArea.
Al igual que la variable anterior, también existen valores atípicos.
"""

# Variables Categóricas - Relación entre SalePrice y OverallQual y YearBuilt

OverallQual = 'OverallQual'

# Para este tipo de variable se puede utilizar un diagrama de caja y brazos (boxplot)

"""data_qual = pd.concat([df_train['SalePrice'], df_train[OverallQual]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))  # Marco
fig = sns.boxplot(x=OverallQual, y='SalePrice', data=data_qual)
fig.axis(ymin=0, ymax=800000)"""

"""
Se confirma la relación lineal directa.
Existe una mayor dispersión relativa en los precios de las casas que tienen OverallQual más alta
debido a que el recorrido intercuartil es mayor que el resto (tamaño de las cajas), de hecho hay
casas del decil 10 que tienen un precio más bajo que una de decil 9. A su vez
parece haber una tendencia de que los valores atípicos por encima del límite superior interno
son más comunes que los que están por debajo del límite inferior interno.
Elementos a considerar:
Cuando OverallQual == 5, la cantidad de ocurrencias atípicas es máxima, tanto para abajo como
para arriba.
El 25% de las casas más baratas del decil 10, vale lo mismo (aproximadamente) o menos que el 50%
de las casas más baratas del decil 9.
"""

# En el caso de YearBuilt

YearBuilt = 'YearBuilt'

data_year = pd.concat([df_train['SalePrice'], df_train[YearBuilt]], axis=1)
"""f, ax = plt.subplots(figsize=(18, 8))
fig = sns.boxplot(x=YearBuilt, y='SalePrice', data=data_year)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)  # Para que años se vean en vertical"""


"""
No se puede evidenciar una asociación lineal muy fuerte puesto que hay
casas muy viejas (antigüedades) que se venden al precio de una propiedad
más actual. Si acotamos la distribución podemos decir que en los últimos
veinte años existen muchísimos más outliers que en los demás, asumiendo
que la variable precio está más dispersa en estos valores.
"""


"""
Basándonos en este simple análisis, podemos afirmar que
Variables Numéricas:
Precio de venta ('SalePrice') se relaciona muchísimo con Superficie Total (GrLiveArea)
de forma lineal directa, al igual que con Superficie del Sótano (BsmtSF) pero esta última
con menor intensidad.

Variables Categóricas:
Conforme a la Calidad (OverallQual) resulta mucho más ponderante a la hora de decidir
un precio de venta (SalePrice) que al año de construcción del inmueble (YearBuilt) debido
a la asociación lineal más fuerte de la primera.
"""

# Para confirmar la hipótesis, procedemos a realizar una matriz de correlación
# Tal que g1 = [[sum(xi-mx)^3]/n]/s^3 ó P = (mx - mo)/s -> P 3(mx - mex)/s

# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(df_train.corr(numeric_only=True), vmax=.8, square=True)
# numeric_only=True since pandas 2.0.0

"""
Llegamos a la conclusión, según la matriz de correlación que
La variable más fuertemente relacionada es OverallQual,
seguido de GrLiveArea y aquellas derivadas del Garage.
- Para ver los r^2:
"""
GarageArea = 'GarageArea'
GarageCars = 'GarageCars'

"""print(f"r2 GrLivArea:", round(df_train.corr(
    numeric_only=True)['SalePrice'][GrLivArea], 4))
print(f"r2 OverallQual:", round(df_train.corr(
    numeric_only=True)['SalePrice'][OverallQual], 4))
print(f"r2 GarageArea:", round(df_train.corr(
    numeric_only=True)['SalePrice'][GarageArea], 4))
print(f"r2 GarageCars:", round(df_train.corr(
    numeric_only=True)['SalePrice'][GarageCars], 4))"""

"""
r^2 GrLivArea: 0.7086 
r^2 OverallQual: 0.791
r^2 GarageArea: 0.6234
r^2 GarageCars: 0.6404

Respecto a las variables analizadas, TotalBsmtSF Y YearBuilt
"""

"""print(f"r2 TotalBsmtSF: ", round(df_train.corr(
    numeric_only=True)['SalePrice'][TotalBsmtSF], 4))
print(f"r2 YearBuilt:", round(df_train.corr(
    numeric_only=True)['SalePrice'][YearBuilt], 4))"""


"""
r2 TotalBsmtSF:  0.6136
r2 YearBuilt: 0.5229

Para lo cual se puede asumir una correlación cierta, pero con menor
intensidad que las variables antes mencionadas.
"""

# Matriz de correlación con números (Pearson o Momentos Centrados)
# Tal que P = (m(x)-mo(x))/s ó 3*(m(x) - me(x))/s  // s ∈ DS(x)
# Tal que G1 = [Σ(xi - m(x))^3/n]/s^3  //  s ∈ DS(x)

"""
Se van a seleccionar las 10 variables con un r2 más alto para asimilar
las conclusiones.
"""
"""k = 10
cols = df_train.corr(numeric_only=True).nlargest(k, 'SalePrice')[
    'SalePrice'].index  # k variables con más r2 de SalePrice"""

"""
Index(['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'],
      dtype='object')

Son las 10 variables más correlaciondas con SalePrice
"""

"""cm = np.corrcoef(df_train[cols].values.T)  # T -> transponer columnas

hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10, },
                 yticklabels=cols.values,
                 xticklabels=cols.values)"""

"""
-OverallQual, GrLiveArea y TotalBsmtSF están fuertemente correlaciondas entre sí.
-Lógicamente las variables derivadas del Garage están correlacionadas entre sí
(colinealidad, no podés cambiar una sin cambiar la otra), pero
también entre sí con 'SalePrice'
-1stFlrSF (Superficie Primer Piso) y TotalBsmSF van de la mano
-De las variables tomadas como Hipótesis, la que menos relación tiene con SalePrice
es YearBuilt.

"""
# Scatterplots entre SalePrice y variables corr.

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea',
        'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], height=2.5)  # Gráfica de pares o bidimensional

plt.show()
