import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io


# =========================
# FUNCIONES DE CARGA
# =========================
def cargar_datos(path):
    try:
        # Cargar archivo CSV con separador ";" , codificación UTF-8 y tomando como índice la primera columna
        df = pd.read_csv(path,sep=';', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# =========================
# FUNCIONES DE INSPECCIÓN , DESCRIPCIÓN y CORRECCION TYPO DATO
# =========================
def inspeccionYcorreccion(df):
    st.subheader("Información del DataSet sin tratamiento")
    col1, col2 = st.columns(2)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])

    # Mostrar ultimos valores de la tabla para verificar el indice de la tab
    st.subheader("Ultimos valores del DataFrame sin tratar")
    st.write(df.tail(5)) 
    st.markdown("Por este medio podemos notar que el indice de la tabla es correcto o hay diferencias con el esperado")

    # Mostrar información general sobre el DataFrame sin tratar
    buffer = io.StringIO()   # Crear un buffer para capturar la salida de info()
    df.info(buf=buffer)      # Captura la salida de info() en el buffer
    s = buffer.getvalue()
    st.text(s)
    
    # Conversión de tipos de datos string
    df['description'] = df['description'].astype("string")
    df['designation'] = df['designation'].astype("string")
    df['title'] = df['title'].astype("string")
    # Conversión de tipos de datos categóricos
    df['country'] = df['country'].astype('category')
    df['province'] = df['province'].astype('category')
    df['region_1'] = df['region_1'].astype('category')
    df['region_2'] = df['region_2'].astype('category')
    df['taster_name'] = df['taster_name'].astype('category')
    df['taster_twitter_handle'] = df['taster_twitter_handle'].astype('category')
    df['variety'] = df['variety'].astype('category')
    df['winery'] = df['winery'].astype('category')
    # Conversión de tipos de datos numéricos

    df['Unnamed: 0'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce')
    df['Unnamed: 0'] = df['Unnamed: 0'].fillna(0).astype('int64')  # Imputar NaN con 0 y convertir a int
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['points'] = pd.to_numeric(df['points'], errors='coerce')  # Asegura que sea numérico
    df['points'] = df['points'].fillna(0).astype('int64')  # Imputar NaN con 0 y convertir a int

    # Mostrar la tabla final con los tipos de datos modificados
    st.subheader("DataFrame actualizado")
    st.write(df.dtypes) 


    #------------------PRELIMINACION *********************
    st.write(f"Tamaño del DataFrame antes de eliminar las filas con 'Unnamed: 0' igual a cero: {len(df)}")
    # eliminar filas con valores no numericos en el indice de ahora en mas tendremos el dataset limpio de esas filas
    df= df[df['Unnamed: 0'] != 0].copy()
    st.write(f"Tamaño del DataFrame después de eliminar las filas con 'Unnamed: 0' igual a cero: {len(df)}")
    #-----------------PRELIMINACION *********************---------------------------
    return df

def explorar_estructura_general(df):
    st.subheader("Estructura general de los datos")

    resumen = [] # lista para almacenar los resúmenes de cada columna
    total_filas = len(df) # cuenta el total de filas del DataFrame

    for col in df.columns:
        n_unicos = df[col].nunique()  # cuenta los valores únicos
        pct_unicos = (n_unicos / total_filas) * 100

        n_nulos = df[col].isnull().sum() # cuenta los nulos
        pct_nulos = (n_nulos / total_filas) * 100

        if df[col].dtype in [np.int64, np.float64]: # Si es numérico cuenta cuantis ceros hay
            n_ceros = (df[col] == 0).sum()
        else:
            n_ceros = 0  # asigna 0 en columnas categóricas

        pct_ceros = (n_ceros / total_filas) * 100

        moda = df[col].mode() # calcula la moda
        moda_valor = moda.iloc[0] if not moda.empty else "Sin datos"

        resumen.append({      # todos los datos se guardan en un diccionario
            'Columna': col,
            'Valores únicos': n_unicos,
            '% únicos': round(pct_unicos, 2),  # redondea a 2 decimales el por % de valores únicos
            'Nulos': n_nulos,
            '% nulos': round(pct_nulos, 2),# redondea a 2 decimales el por % de valores nulos
            'Ceros': n_ceros,
            '% ceros': round(pct_ceros, 2),# redondea a 2 decimales el por % de valores ceros
            'Moda': moda_valor
        })

    resumen_df = pd.DataFrame(resumen) #Convierte la lista de diccionarios a un DataFrame para poder mostrarlo en tabla
    st.dataframe(resumen_df.style.background_gradient(cmap='Reds', subset=['% nulos']))

# =========================
# FUNCIONES DE ANÁLISIS EXPLORATORIO
# =========================
def resumen_estadistico(df):
    

    st.subheader("Resumen estadístico - Variables numéricas")
    st.write(df.describe())
    st.subheader("Resumen - Variables categóricas")
    st.write(df.select_dtypes(include='object').describe().T)
    

def graficos_exploratorios(df):
    st.subheader("Distribuciones y conteos")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if num_cols:
        st.write("Histogramas de variables numéricas")
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribución de {col}")
            st.pyplot(fig)

    if cat_cols:
        st.write("Conteo de variables categóricas (top 10 categorías)")
        for col in cat_cols[:5]:  # limita para evitar sobrecarga
            fig, ax = plt.subplots()
            df[col].value_counts().nlargest(10).plot(kind='bar', ax=ax)
            ax.set_title(f"Conteo de {col}")
            st.pyplot(fig)

# =========================
# FUNCIONES DE DATOS FALTANTES
# =========================
def mostrar_datos_faltantes(df):
    st.subheader("Datos faltantes")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.write(missing)
    st.markdown("**Estrategias comunes para tratar valores nulos:**")
    st.markdown("- Eliminar columnas o filas con demasiados nulos")
    st.markdown("- Imputar con la media, mediana o moda")
    st.markdown("- Usar modelos predictivos para imputación") 
    st.markdown("- Imputar a partir de una distribución (normal u otra)")

def imputar_media(df, columnas):
    st.markdown(f"**Se trató los siguientes faltantes {columnas} mediante el siguiente método: imputación por media**")
    for col in columnas:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)

def imputar_distribucion_normal(df, columnas):
    st.markdown(f"**Se trató los siguientes faltantes {columnas} mediante el siguiente método: imputación por distribución normal**")
    for col in columnas:
        if df[col].dtype in [np.float64, np.int64]:
            media = df[col].mean()
            std = df[col].std()
            n_missing = df[col].isnull().sum()
            valores_imputados = np.random.normal(loc=media, scale=std, size=n_missing)
            df.loc[df[col].isnull(), col] = valores_imputados

# =========================
# FUNCIONES DE ANÁLISIS DE OUTLIERS
# =========================
def detectar_outliers(df):
    st.subheader("Análisis de valores atípicos (outliers)")
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot de {col}")
        st.pyplot(fig)

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        st.write(f"{col}: {len(outliers)} outliers detectados")

    st.markdown("**Nota:** Los outliers pueden ser errores de medición o datos válidos pero extremos. Su tratamiento depende del contexto.")

def eliminar_outliers_iqr(df, columnas):
    st.markdown(f"**Se trató los siguientes outliers {columnas} mediante el siguiente método: eliminación por IQR**")
    for col in columnas:
        if df[col].dtype in [np.float64, np.int64]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            condicion = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
            df = df[condicion]
    return df

def winsorizar_outliers(df, columnas):
    st.markdown(f"**Se trató los siguientes outliers {columnas} mediante el siguiente método: winsorización (percentiles 1-99)**")
    for col in columnas:
        if df[col].dtype in [np.float64, np.int64]:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower, upper)

# =========================
# FUNCIÓN PRINCIPAL
# =========================
def main():
    st.title("Limpieza y Análisis Exploratorio de Datos ")
    st.subheader("Autor: Francisco Molinari")
    st.markdown("Este proyecto utiliza Streamlit para crear una interfaz interactiva que permite cargar, limpiar y analizar un conjunto de datos de vinos. El análisis incluye la inspección de datos, el tratamiento de valores faltantes y la detección y manejo de outliers.")
    
    # Ruta del archivo CSV
    path = "2winemag-data-130k-v2.csv" 

    # Cargar archivo
    df = cargar_datos(path)

    if df is not None:
        # Menú de navegación lateral
        opcion = st.sidebar.selectbox(
            "Selecciona una sección del análisis",
            (
                "Inspección y corrección",
                "Resumen estadístico y graficos",
                "Datos faltantes",
                "Outliers",
            )
        )
    
        # Mostrar según opción elegida
        if opcion == "Inspección y corrección":
            df=inspeccionYcorreccion(df)
            explorar_estructura_general(df)
        elif opcion == "Resumen estadístico y graficos":
            resumen_estadistico(df)
            graficos_exploratorios(df)
        elif opcion == "Datos faltantes":
            mostrar_datos_faltantes(df)
            imputar_media(df, ['price'])
            imputar_distribucion_normal(df, ['points'])
        elif opcion == "Outliers":
            detectar_outliers(df)
            df = eliminar_outliers_iqr(df, ['price'])
            winsorizar_outliers(df, ['price'])
            
if __name__ == "__main__":
    main()
