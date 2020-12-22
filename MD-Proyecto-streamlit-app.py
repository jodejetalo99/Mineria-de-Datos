import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logo = Image.open("logoUNAM.png")
logo2 = Image.open("Proyecto/IIMAS.png")
hospital = Image.open("Proyecto/rft.jpg")

# Formato
st.image([logo,logo2],use_column_width=False, width=100)
#st.image(logo,use_column_width=False, width=100)
#st.image(logo2,use_column_width=False, width=100)
st.sidebar.image(hospital, width=300)
st.sidebar.info("Aplicación creada por Misael López Sánchez y José de Jesús Tapia López")
st.title("Universidad Nacional Autónoma de México")
st.header("Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas")
st.header("Minería de Datos")
st.header("Proyecto: Ciencia de Datos enfocada al estudio de proyectos financiados con recursos federales transferidos")
st.header("Enero del 2021")

datos_recursos = pd.read_csv("Proyecto/recursos-federales-transferidos.csv")

def graficaFrecuencia(variable):
    fig = plt.figure(figsize=(5,4))
    ax = sns.countplot(x=variable, data=datos_recursos, order = datos_recursos[variable].value_counts().index)
    for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+25),fontsize = 6, rotation = 25)
    ax.set_ylabel('Frecuencia',fontsize=10)
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel(variable,fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title(variable,fontsize=12)
    return fig

variable = st.selectbox("Variable", ["Alcaldía","Descripción del Flujo","DESC_CLASIFICACION",
                                     "Descripción del Tipo de Proyecto","Descripción de la Categoría del Proyecto",
                                     "DESC_ESTATUS","INFORMACION_CONTRATOS"])
st.pyplot(graficaFrecuencia(variable))

def diagramaDeCaja(variable_monto):
    fig = plt.figure(figsize=(5,4))
    ax = sns.boxplot(variable_monto,data=datos_recursos)
    #ax.set_ylabel('Frecuencia',fontsize=10)
    ax.set_xlabel('')
    ax.set_title("Diagrama de caja del "+ variable_monto,fontsize=12)
    return fig

variable_monto = st.selectbox("Variable Monto", ["Monto Aprobado","Monto Modificado","Monto Comprometido",
                                     "Monto Devengado","Monto Ejercido"])
st.pyplot(diagramaDeCaja(variable_monto))


### KNN
st.header("Predicción con KNN")
st.write("Dados la categoría, la alcaldía, la clasificación y el estatus del proyecto, predecir la información del contrato")
datos_recursos['ESTATUS'] = pd.factorize(datos_recursos['DESC_ESTATUS'])[0]
datos_recursos['CLAVE_INFORMACION_CONTRATOS'] = pd.factorize(datos_recursos['INFORMACION_CONTRATOS'])[0]

datos_recursos_knn = datos_recursos[['Categoría del Proyecto', 'Clave de la Alcaldía', 'CLASIFICACION','ESTATUS','CLAVE_INFORMACION_CONTRATOS']]
datos_recursos_knn = datos_recursos_knn.dropna()

# obtenemos las variables que vamos a usar para predecir
X = datos_recursos_knn[['Categoría del Proyecto', 'Clave de la Alcaldía', 'CLASIFICACION','ESTATUS']]
# variable predicha
y = datos_recursos_knn['CLAVE_INFORMACION_CONTRATOS']

# dividimos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


knn = KNeighborsClassifier(n_neighbors=53,weights='distance',algorithm='brute',metric='manhattan')
knn.fit(X_train,y_train)

dicc_categoria_proyecto = {'Otros Proyectos':0, 'Acción/Salud':1, 'Adquisición/Educación':2, 
                           'Proyecto de inversión/Transportes y vialidades':3, 'Agua y saneamiento':4,
                           'Urbanización':5, 'Cultura y turismo':6, 'Asistencia Social':7,
                           'Deporte':8, 'Seguridad':9, 'Vivienda':10, 'Comunicaciones':11}
dicc_clave_alcaldia = {'Gobierno de la Entidad':0.0,'Azcapotzalco':2.0, 'Coyoacán':3.0, 'Cuajimalpa de Morelos':4.0,
                       'Gustavo A. Madero':5.0, 'Iztacalco':6.0, 'Iztapalapa':7.0, 
                       'La Magdalena Contreras':8.0, 'Milpa Alta':9.0, 'Álvaro Obregón':10.0,
                       'Tláhuac':11.0, 'Tlalpan':12.0, 'Xochimilco':13.0, 'Benito Juárez':14.0,
                       'Cuauhtémoc':15.0, 'Miguel Hidalgo':16.0, 'Venustiano Carranza':17.0}
dicc_clasificacion = {'Otros proyectos':0.0, 'Acción/Salud': 1.0, 'Educación/Adquisición':2.0, 
                      'Proyecto de inversión/Transportes y vialidades':3.0, 'Agua y saneamiento':4.0,
                      'Urbanización':5.0, 'Cultura y Turismo':6.0, 'Asistencia Social':7.0,
                      'Deporte':8.0, 'Seguridad':9.0, 'Vivienda':10.0 , 'Comunicaciones':11.0,
                      'PRODIM':12.0}
dicc_estatus = {'En Ejecución':0, 'Terminado':1, 'Cancelado':2, 'Suspendido':3}

categoria_proyecto =  st.selectbox("Categoría del Proyecto", ['Otros Proyectos','Acción/Salud', 'Adquisición/Educación', 
                                                              'Proyecto de inversión/Transportes y vialidades', 
                                                              'Agua y saneamiento','Urbanización', 'Cultura y turismo', 
                                                              'Asistencia Social','Deporte', 'Seguridad', 'Vivienda',
                                                              'Comunicaciones'])

clave_alcaldia = st.selectbox("Alcaldía", ['Gobierno de la Entidad','Azcapotzalco', 'Coyoacán', 
                                           'Cuajimalpa de Morelos','Gustavo A. Madero', 'Iztacalco', 
                                           'Iztapalapa','La Magdalena Contreras', 'Milpa Alta', 'Álvaro Obregón',
                                           'Tláhuac', 'Tlalpan', 'Xochimilco', 'Benito Juárez',
                                           'Cuauhtémoc', 'Miguel Hidalgo', 'Venustiano Carranza'])

clasificacion = st.selectbox("Clasificación", ['Otros proyectos','Acción/Salud', 'Educación/Adquisición',
                                               'Proyecto de inversión/Transportes y vialidades','Agua y saneamiento',
                                               'Urbanización', 'Cultura y Turismo', 'Asistencia Social',
                                               'Deporte', 'Seguridad', 'Vivienda', 'Comunicaciones','PRODIM'])

estatus = st.selectbox("Estatus", ['En Ejecución','Terminado','Cancelado','Suspendido'])

dicc_clave_inf = {0:'Sin contratos',1:'Sí',2:'No'}



#Predicciones con KNN
prediccion_knn = knn.predict([[dicc_categoria_proyecto[categoria_proyecto],
                               dicc_clave_alcaldia[clave_alcaldia], 
                               dicc_clasificacion[clasificacion], dicc_estatus[estatus]]])

st.write("Por medio de una predicción con KNN, ¿Lo más seguro es que la información del contrato, con estas características, esté disponible? ¿O acaso la información del proyecto involucra que sea sin contratos?", dicc_clave_inf[int(prediccion_knn)])


# Regresión Logística
st.header("Predicción con Regresión Logística")
st.write("Dadas la categoría del proyecto y la información del contrato, predecir si el proyecto es terminado o cancelado")
datos_recursos_rl = datos_recursos.loc[datos_recursos['ESTATUS'].isin(['1','2'])]
datos_recursos_rl = datos_recursos_rl[['Categoría del Proyecto','ESTATUS','CLAVE_INFORMACION_CONTRATOS']]
datos_recursos_rl = datos_recursos_rl.dropna()

# obtenemos las variables que vamos a usar para predecir
X2 = datos_recursos_rl[['Categoría del Proyecto', 'CLAVE_INFORMACION_CONTRATOS']].values
# variable predicha
y2 = datos_recursos_rl['ESTATUS'].values

# aunque son binarias, instanciamos un codificador de etiquetas para que sea 0, 1 
# en lugar de 1,2

le = LabelEncoder()
y2 = le.fit_transform(y2)

# Los Terminados van a ser 0, los cancelados van a ser 1
le.transform([1,2])

# dividimos en entrenamiento y prueba
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size = 0.2, random_state = 42)

# pipeline como encapulador de pasos
# el valor de C por defecto es 1.0, y proximamente mostraremos que es bueno
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train2,y_train2)

dicc_categoria_proyecto2 = {'otros proyectos':0, 'acción/salud':1, 'adquisición/educación':2, 
                           'proyecto de inversión/transportes y vialidades':3, 'agua y saneamiento':4,
                           'urbanización':5, 'cultura y turismo':6, 'asistencia social':7,
                           'deporte':8, 'seguridad':9, 'vivienda':10, 'comunicaciones':11}


dicc_clave_inf2 = {'Sin contratos':0,'Sí':1,'No':2}

categoria_proyecto2 =  st.selectbox("Categoría del Proyecto", ['otros proyectos','acción/salud', 'adquisición/educación', 
                                                              'proyecto de inversión/transportes y vialidades', 
                                                              'agua y saneamiento','urbanización', 'cultura y turismo', 
                                                              'asistencia social','deporte', 'seguridad', 'vivienda',
                                                              'comunicaciones'])

clave_inf = st.selectbox("Información del contrato", ['Sin contratos','Sí','No'])

dicc_estatus2 = {0:'Terminado', 1:'Cancelado'}

#Predicciones con KNN
prediccion_logreg = logreg.predict([[dicc_categoria_proyecto2[categoria_proyecto2],
                               dicc_clave_inf2[clave_inf]]])

st.write("Por medio de una predicción con Regresión Logística, ¿Lo más seguro es que el proyecto sea Terminado o Cancelado?", dicc_estatus2[int(prediccion_logreg)])


