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

import geopandas as gpd
#import statsmodels.api as sm
import pymc3 as pm


st.set_option('deprecation.showPyplotGlobalUse', False)

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


def graficaFrecuencia(variable, limite_x=200):
    # Funcion principal para las gráficas.
    figure = plt.figure(figsize=(12,8))
    ax = sns.countplot(y=variable, data=datos_recursos, order = datos_recursos[variable].value_counts().index)
    for p in ax.patches:
      # Obtenemos la ubicación de X e Y de la etiqueta de p.
      x_value = p.get_width()
      y_value = p.get_y() + p.get_height() / 2
      # Numero de puntos entre la barra y la etiqueta.
      espacio = 1
      # Usamos el valor de X como etiqueta y número de formato sin decimales
      label = "{:.0f}".format(x_value)

      # Colocamos la frecuencia en cada barra
      plt.annotate(
          label,                      # frecuencia de cada barra como etiquetas
          (x_value, y_value),         # coordenadas en las que pondremos las etiquetas
          xytext=(espacio, 0),          # Desplazamos cada etiqueta horizontalmente por "espacio"
          textcoords="offset points", # como compensación en puntos
          va='center',                # centramos la etiqueta verticalmente
          ha='left',                  # alineación vertical
          fontsize=13)              # tamanio del texto  
    ax.set_xlabel('Frecuencia',fontsize=13.5)
    ax.tick_params(axis='x')
    ax.set_ylabel(variable,fontsize=13.5)
    ax.set_xlim(0,max(datos_recursos[variable].value_counts())+limite_x)
    ax.tick_params(axis='both', which='major', labelsize=13.5)
    return (figure)


variable = st.selectbox("Variable", ["Alcaldía","Descripción del Flujo","DESC_CLASIFICACION",
                                     "Descripción del Tipo de Proyecto","Descripción de la Categoría del Proyecto",
                                     "DESC_ESTATUS","INFORMACION_CONTRATOS"])
st.pyplot(graficaFrecuencia(variable))



def diagramaDeCaja(variable_monto):
    fig = plt.figure(figsize=(5,4))
    ax = sns.boxplot(x = variable_monto,data=datos_recursos)
    #ax.set_ylabel('Frecuencia',fontsize=10)
    ax.set_xlabel('')
    ax.set_title("Diagrama de caja del "+ variable_monto,fontsize=12)
    return fig

st.header("Diagrama de caja de los montos de los proyectos en la CDMX de 2013 a 2019")

variable_monto = st.selectbox("Variable Monto", ["Monto Aprobado","Monto Modificado","Monto Comprometido",
                                     "Monto Devengado","Monto Ejercido"])
st.pyplot(diagramaDeCaja(variable_monto))

datos_recursos2  = datos_recursos[datos_recursos['Ciclo'] >= 2019]

def diagramaDeCaja4T(variable_montos):
    variable_montos = variable_montos.title()
    fig = plt.figure(figsize=(5,4))
    
    ax = sns.boxplot(x = variable_montos,data=datos_recursos2)
    #ax.set_ylabel('Frecuencia',fontsize=10)
    ax.set_xlabel('')
    ax.set_title("Diagrama de caja del "+ variable_montos,fontsize=12)
    return fig

st.header("Diagrama de caja de los montos de los proyectos en la CDMX en la 4T")

variable_montos = st.selectbox("Variable Montos", ["Monto aprobado","Monto modificado","Monto comprometido",
                                     "Monto devengado","Monto ejercido"])
st.pyplot(diagramaDeCaja4T(variable_montos))

datos_recursos3 = datos_recursos[datos_recursos['Ciclo'] < 2019]

def diagramaDeCajaNO4T(variables_montos):
    variables_montos = variables_montos.title()
    fig = plt.figure(figsize=(5,4))
    
    ax = sns.boxplot(x = variables_montos,data=datos_recursos3)
    #ax.set_ylabel('Frecuencia',fontsize=10)
    ax.set_xlabel('')
    ax.set_title("Diagrama de caja del "+ variables_montos,fontsize=12)
    return fig

st.header("Diagrama de caja de los montos de los proyectos en la CDMX antes de la 4T")

variables_montos = st.selectbox("Variables Montos", ["monto aprobado","monto modificado","monto comprometido",
                                     "monto devengado","monto ejercido"])
st.pyplot(diagramaDeCajaNO4T(variables_montos))


## Proyectos realizados por Trimestre
st.header("Proyectos realizados por Trimestre")
st.write("Proyectos realizados en la CDMX desde 2013 a 2019") 


figure = plt.figure(figsize=(12,8))
ax = sns.countplot(x='Ciclo', hue='Periodo',data= datos_recursos)
for p in ax.patches:
    ax.annotate("{:.0f}".format(p.get_height()), (p.get_x(), p.get_height()+40), fontsize=13.5, rotation=80)
ax.set_xlabel('Ciclo',fontsize=13.5)
ax.tick_params(axis='x')
ax.set_ylabel('Frecuencia',fontsize=13.5)
ax.set_ylim(0,2600)
ax.tick_params(axis='both', which='major', labelsize=13.5)
plt.title("Proyectos realizados por trimestre en la CDMX de 2013 a 2019",fontsize=15)
st.pyplot(figure)


st.header("Cantidad de recursos por alcaldía")
st.write("Gráfica de el monto de recursos recibidos por alcaldía en la CDMX de 2013 a 2019")

figure = plt.figure(figsize=(12,8))
ax = datos_recursos.groupby('Alcaldía')['Monto Aprobado'].agg('sum').sort_values(ascending=False).plot.bar()
for p in ax.patches:
    ax.annotate("{:.0f}".format(p.get_height()), (p.get_x(), p.get_height()+1e8), fontsize=13.5, rotation=45)
ax.set_xlabel('Alcaldía',fontsize=13.5)
ax.set_ylim(0,5.5e10)
ax.set_xlim(-1,17.5)
ax.set_ylabel('Monto Aprobado de los proyectos',fontsize=13.5)
ax.tick_params(axis='both', which='major', labelsize=13.5)
plt.title("Montos recibidos de los proyectos por alcaldía en la CDMX de 2013 a 2019",fontsize=15)
st.pyplot(figure)


# Analisis de los proyectos realizados

st.header("Proyectos realizados con recursos públicos")
st.write("Uno de los objetivos principales del proyecto fue la investigación del número de proyectos realizados a lo largo del tiempo con los datos disponibles de nuestra base de datos, a continuación presentamos una tabla con la información de los proyectos realizados a lo largo del tiempo con los proyectos realizados por trimestre.")

proyect_trim = pd.pivot_table(data=datos_recursos[['Ciclo','Periodo','Folio de Proyecto de la SHCP']], index='Ciclo', columns='Periodo', aggfunc='count')

st.dataframe(proyect_trim)
    

## Analisis por medio de series de tiempo
st.header("Serie de tiempo de los proyectos realizados en la CDMX")
st.write("Serie de tismpo de los proyectos realizados por día inicializados y finalizados en la Ciudad de México que aparecen en la información de los registros historicos de la base de datos")

def grafica_st(estado):
    figure = plt.figure(figsize=(15,9))
    datos_recursos.groupby(estado)['Ciclo'].agg('count').plot.line().grid()
    plt.ylabel('Cantidad de proyectos',fontsize=13.5)
    plt.xlabel('Fecha de inicio del Proyecto',fontsize=13.5)
    plt.tick_params(axis='both', which='major', labelsize=13.5)
    plt.title("Proyectos iniciados en la CDMX",fontsize=15)
    return (figure)


estado_st = st.selectbox("Estado", [ 'Fecha de Inicio del Proyecto','Fecha de término del Proyecto'])

st.pyplot(grafica_st(estado_st))


### Separación de la información en varios años 
# Calculamos algunos parametros que nos serviran más adelante3

def datosRecursosAnio(anio):
  return (datos_recursos[datos_recursos['Ciclo'] == anio])


datos_2013 = datosRecursosAnio(2013)
datos_2014 = datosRecursosAnio(2014)
datos_2015 = datosRecursosAnio(2015)
datos_2016 = datosRecursosAnio(2016)
datos_2017 = datosRecursosAnio(2017)
datos_2018 = datosRecursosAnio(2018)
datos_2019 = datosRecursosAnio(2019)


### Montos anuales gastados por año
st.header("Montos Anuales")
st.write("Describimos los montos anuales gastados en la ciudad de méxico por año, mostramos los principales 4 partidas de montos dentro de la Base de datos los que son los siguientes:")

st.write("**Monto Aprobado**: Son las asignaciones presupuestarias anuales comprendidas en el Presupuesto de Egresos a nivel de clave presupuestaria en el caso de los ramos autónomos, administrativos y generales, y a nivel de los rubros de gasto que aparecen en las carátulas de flujo de efectivo para las entidades.") 

st.write("**Monto Modificado**: Momento contable del gasto que refleja la asignación presupuestaria que resulta de incorporar, en su caso, las adecuaciones presupuestarias al presupuesto aprobado.") 

st.write("**Monto Comprometido**: Momento contable que denota la aprobación por la autoridad competente de un acto administrativo, u otro instrumento jurídico que formaliza una relación jurídica con terceros para la adquisición de bienes y servicios o ejecución de obras. En el caso de las obras a ejecutarse o de bienes y servicios a recibirse durante varios ejercicios, el compromiso será registrado durante cada ejercicio.")

st.write("**Monto Pagado**: Momento contable que refleja la cancelación total o parcial de las obligaciones de pago que se concreta mediante el desembolso de efectivo o cualquier otro medio de pago.")




def montosAnuales(datos_anio, anio):

  print("Año: ", anio)

  montos = {'Tipos de Montos': ['Monto Aprobado','Monto Modificado',
                       'Monto Comprometido','Monto Devengado',
                       'Monto Ejercido'],
            'Monto Anual': [datos_anio['Monto Aprobado'].sum(), datos_anio['Monto Modificado'].sum(),
                            datos_anio['Monto Comprometido'].sum(), datos_anio['Monto Devengado'].sum(),
                            datos_anio['Monto Ejercido'].sum()]
        }

  df = pd.DataFrame(montos, columns = ['Tipos de Montos', 'Monto Anual'])
  return df

diccionario_anio = {'2013':datos_2013,'2014':datos_2014,
                    '2015':datos_2015,'2016':datos_2016,
                    '2017':datos_2017,'2018':datos_2018,
                    '2019':datos_2019}
agno_recursos = st.selectbox("Año fiscal", [ '2013','2014','2015','2016','2017','2018','2019'])

st.dataframe(montosAnuales(diccionario_anio[agno_recursos],agno_recursos))


# Agregamos el bloque de separación de los dataframes en inforamción de ant y act
datos_2013_ant = datos_2013[datos_2013['Ciclo'] > datos_2013['Ciclo del recurso']]
datos_2014_ant = datos_2014[datos_2014['Ciclo'] > datos_2014['Ciclo del recurso']]
datos_2015_ant = datos_2015[datos_2015['Ciclo'] > datos_2015['Ciclo del recurso']]
datos_2016_ant = datos_2016[datos_2016['Ciclo'] > datos_2016['Ciclo del recurso']]
datos_2017_ant = datos_2017[datos_2017['Ciclo'] > datos_2017['Ciclo del recurso']]
datos_2018_ant = datos_2018[datos_2018['Ciclo'] > datos_2018['Ciclo del recurso']]
datos_2019_ant = datos_2019[datos_2019['Ciclo'] > datos_2019['Ciclo del recurso']]

# Hacemos la separación de los registros de proyectos que corresponden al ejercicio fiscal vigente pues son recursos que estan en 
# regla en teoria y aún no se fiscalizan por completo, sino que se estan planeando. 
datos_2013_act = datos_2013[datos_2013['Ciclo'] == datos_2013['Ciclo del recurso']]
datos_2014_act = datos_2014[datos_2014['Ciclo'] == datos_2014['Ciclo del recurso']]
datos_2015_act = datos_2015[datos_2015['Ciclo'] == datos_2015['Ciclo del recurso']]
datos_2016_act = datos_2016[datos_2016['Ciclo'] == datos_2016['Ciclo del recurso']]
datos_2017_act = datos_2017[datos_2017['Ciclo'] == datos_2017['Ciclo del recurso']]
datos_2018_act = datos_2018[datos_2018['Ciclo'] == datos_2018['Ciclo del recurso']]
datos_2019_act = datos_2019[datos_2019['Ciclo'] == datos_2019['Ciclo del recurso']]

# AGregamos la presentación de aquellos registros que aparecen sospechosos.

st.header("Proyectos sospechosos de corrupción") 
st.write("En esta sección presentamos la información de aquellos proyectos que tienen la caracteristicas sospechosas de posible corrupción, esto debido a que son proyectos que tienen las caracteristicas de un avance fisico del 0% en el ciclo fiscal vigente y cuyo porcentaje de monto pagado es mayor al 50% lo que quiere decir que son proyectos que han estado siendo pagados pero que sus avances son nulos")



inconclusos_2013 = datos_2013_ant[(datos_2013_ant['Avance físico'] == 0) & (datos_2013_ant['porcentaje_pagado'] >= 50)]
inconclusos_2014 = datos_2014_ant[(datos_2014_ant['Avance físico'] == 0) & (datos_2014_ant['porcentaje_pagado'] >= 50)]
inconclusos_2015 = datos_2015_ant[(datos_2015_ant['Avance físico'] == 0) & (datos_2015_ant['porcentaje_pagado'] >= 50)]
inconclusos_2016 = datos_2016_ant[(datos_2016_ant['Avance físico'] == 0) & (datos_2016_ant['porcentaje_pagado'] >= 50)]
inconclusos_2017 = datos_2017_ant[(datos_2017_ant['Avance físico'] == 0) & (datos_2017_ant['porcentaje_pagado'] >= 50)]
inconclusos_2018 = datos_2018_ant[(datos_2018_ant['Avance físico'] == 0) & (datos_2018_ant['porcentaje_pagado'] >= 50)]
inconclusos_2019 = datos_2019_ant[(datos_2019_ant['Avance físico'] == 0) & (datos_2019_ant['porcentaje_pagado'] >= 50)]

dic_inconclusos = {'2013':inconclusos_2013, 
                   '2014':inconclusos_2014,
                   '2015':inconclusos_2015,
                   '2016':inconclusos_2016,
                   '2017':inconclusos_2017,
                   '2018':inconclusos_2018,
                   '2019':inconclusos_2019}

agno_recursos_dos = st.selectbox("Proyectos inconclusos", [ '2013','2014','2015','2016','2017','2018','2019'])

st.dataframe(dic_inconclusos[agno_recursos_dos])   

## Separación de los proyectos en actualies y viegentes

# Creamos un diccionario para mandar a llamar a los datos
datos_dict = {2013:datos_2013_act,
              2014:datos_2014_act,
              2015:datos_2015_act,
              2016:datos_2016_act,
              2017:datos_2017_act,
              2018:datos_2018_act,
              2019:datos_2019_act}


datos_dict_ant = {2013:datos_2013_ant,
                2014:datos_2014_ant,
                2015:datos_2015_ant,
                2016:datos_2016_ant,
                2017:datos_2017_ant,
                2018:datos_2018_ant,
                2019:datos_2019_ant}

option = st.selectbox("option", [ 'Fecha de Inicio del Proyecto', 'Fecha de término del Proyecto'])
agnio_fiscal = st.selectbox("Montos Ciclo fiscal", [ '2013','2014','2015','2016','2017','2018','2019'])

st.pyplot(datos_dict[int(agnio_fiscal)].groupby(option)['Monto Aprobado','Monto Comprometido','Monto Pagado'].agg('sum').plot.line(figsize=(15,9)).grid()) 






# Observamos el problema de los montos sobrepasados de la información.

st.header("Proyectos con montos sobrepasados")
st.write("Ponemos a continuación se muestran por ciclo fiscal aquellos registros que tienen montos pagados mayores a los montos aprobados o montos comprometidos, estos proyectos son principalmente alarmantes debido a que son proyectos en los que se pagaron más recursos de los que originalmente fueron presupuestados, estos proyectos levantan más la sospecha de corrupción pues no debieron sobrepasar los montos aprobados") 




def monto_sobrepasado(agnio_fiscal):
    '''
    @summary: Función que recibe como parámetro un data frame y lo que se devuelve son
              los registros con respecto  al año fiscal cuyos montos pagados superaron 
              ya sea el monto aprobado o superaron el monto comprometido
    '''
    registros_sup = agnio_fiscal[(agnio_fiscal['Monto Pagado']>agnio_fiscal['Monto Aprobado']) | (agnio_fiscal['Monto Pagado']>agnio_fiscal['Monto Comprometido'])]
    #print("De los {} registros de programas originales un total de {} son los que sobrepasaron ya sea sum monto aprobado o comprometido".format(len(agnio_fiscal), len(registros_sup)))

    # Ahora podemos calcular cosillas más interesante sobre como es que se desplegaron los recursos
    


    # Una vez que ya tengo detectados cuales son los problematicos creo una variable extra con la diferencia
    # monto pagado - monto aprobado -> esto nos diráá cuanto es que se pasaron de lo que presupuestaron originalmente.
    registros_sup['Monto_excedido'] = registros_sup['Monto Pagado'] - registros_sup['Monto Aprobado']
    return (registros_sup[['Nombre del  Proyecto','Descripción del Ramo', 'Descripción de la Categoría del Proyecto',
                           'Alcaldía', 'Descripción del recurso', 'Monto Aprobado', 'Monto Modificado',
                           'Monto Comprometido', 'Monto Pagado','Monto_excedido',
                           'porcentaje_pagado', 'DESC_ESTATUS']])

agno_recursos = st.selectbox("Montos excedidos", [ '2013','2014','2015','2016','2017','2018','2019'])

st.dataframe(monto_sobrepasado(datos_dict[int(agno_recursos)]))



# Descripcion de los montos sobrepasados que establecimos en el punto anterior. 

st.write("Estos programas que tienen montos sobrecargados son fácilmente identificables, a continuación mostramos cuales son:")

def descripcion_sobrecargos(data_montos):
    '''
    @Summary: En esta funcion realizaremos el proceso de descripcion de todas las tablas que encontramos
    anteriormente, con esto lo que buscamos es crear los grááficos para ver por año donde es que se estan gastando 
    los recurso y donde se esta gastando.
    Nota: Recordemos que estamos analizando todos aqueyos proyectos en los que el monto pagado fue mayor al presupuestado originalmente
    '''

    fig, ax = plt.subplots(1,2,figsize=(20,6) )
    sns.despine(left=True)
    #fig.subtitle("Distribucion de montos exedidos")

    g1 = sns.boxplot(ax=ax[0], x='Alcaldía', y='Monto_excedido', data=data_montos)
    ax[0].set_title("Distribución de montos por alaldía")
    ax[0].xaxis.set_visible(True)
    ax[0].set_xticklabels(labels=data_montos['Alcaldía'], rotation=70)
    #ax[0].tick_params(rotation=70)
    
    #ax[0].set_rotation(70)
    #plt.xticks(rotation=70)
    

    g2 = sns.boxplot(ax=ax[1] ,x='Descripción de la Categoría del Proyecto', y='Monto_excedido', data=data_montos)
    ax[1].set_title("Distribución de montos por Categoría")
    ax[1].xaxis.set_visible(True)
    ax[1].tick_params(rotation=70)
    
    plt.show()
    return(fig)

    #display(data_montos.groupby('Nombre del  Proyecto')[['Monto_excedido']].sum())

#descripcion_sobrecargos(monto_sobrepasado(datos_2015_act))

# @MLS21
## Me esta dando problemas esta parte del subplot, espero resolverlo pronto
#st.pyplot(descripcion_sobrecargos(monto_sobrepasado(datos_2015_act)))


st.dataframe(monto_sobrepasado(datos_dict[int(agno_recursos)]).groupby('Nombre del  Proyecto')[['Monto_excedido']].sum())


## Creamos la tablas para apreciación de los montos de diferentes recursos de acuerdoa  diferentes campos

list(datos_dict.keys())
list_categorias = []
list_alcaldias = []
list_proyectos = []


for k in list(datos_dict.keys()):
    df_categoria = monto_sobrepasado(datos_dict[k]).groupby('Descripción de la Categoría del Proyecto')[['Monto_excedido']].sum()
    df_alcaldia = monto_sobrepasado(datos_dict[k]).groupby('Alcaldía')[['Monto_excedido']].sum()
    df_proyectos = monto_sobrepasado(datos_dict[k]).groupby('Descripción del recurso')[['Monto_excedido']].sum()
    # agrupasmos en las listas
    list_categorias.append(df_categoria)
    list_alcaldias.append(df_alcaldia)
    list_proyectos.append(df_proyectos)


categoria_irr = pd.concat(list_categorias, axis=1)
alcaldia_irr = pd.concat(list_alcaldias, axis=1)
proyectos_irr = pd.concat(list_proyectos, axis=1)


alcaldia_irr.columns = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']
proyectos_irr.columns = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']
categoria_irr.columns = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']

# Agrupo los dataframes que acabo de crear en un solo diccionario
dict_frames = {'Descripción de la Categoría del Proyecto':categoria_irr,
               'Alcaldía':alcaldia_irr,
               'Descripción del recurso':proyectos_irr}


select_recurso = st.selectbox("Monto por categoría", ['Descripción de la Categoría del Proyecto','Alcaldía','Descripción del recurso'])

st.dataframe(dict_frames[select_recurso])









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


knn = KNeighborsClassifier(n_neighbors=44,weights='distance',algorithm='auto',metric='euclidean')
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
logreg = LogisticRegression(C=0.1,tol=1.0, fit_intercept=True, solver='saga', random_state = 42)
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


