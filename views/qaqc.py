import streamlit as st
import pandas as pd
import numpy as np
from scrpt import scrptqc
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("ARD and HARD analysis")
# Subida de archivo CSV
uploaded_file = st.file_uploader("Choose a file", type="csv")
# Verificar si el archivo fue cargado
if uploaded_file is not None:
    # Leer el archivo CSV
    df1x = pd.read_csv(uploaded_file)
    #df = df1x[df1x['QC_Category'].str.startswith('y')]
    # Mostrar un resumen de las primeras filas del DataFrame
    st.write(df1x.head())
    # Definir parámetros con entradas interactivas para los usuarios
    #st.subheader("Parámetros de Entrada")
    # Parámetros que el usuario puede modificar
    unique_qc_types = df1x['QC_Category'].unique()
    unique_element = df1x['Element'].unique()
    #st.write(unique_qc_types)
    # Usar esos valores únicos en el selectbox
    with st.sidebar:
        st.subheader("Input Parameters")
        QC = st.multiselect('Value for QC Category', options=unique_qc_types, default=unique_qc_types)
        org = st.text_input("Value for Original", value="Orig_Au")
        dup = st.text_input("Value for Duplicate", value="Au_ppm")
        #Elem = st.text_input("Element", value="Au")
        Elem = st.multiselect('Value for Element', options=unique_element, default=unique_element)
        ldl = st.number_input("Detection limit (LDL)", format="%.3f")
        # Usar un slider en la barra lateral para ARD
        ARD = st.slider("Select ARD percentage", 
                            min_value=0.1, 
                            max_value=0.3, 
                            value=0.3, 
                            step=0.05, 
                            format="%.2f", 
                            key="ARD")
    # Usar un slider en la barra lateral para ARD
    # Encontrar el valor máximo entre las dos columnas
    df = df1x[df1x['QC_Category'].isin(QC)&df1x['Element'].isin(Elem)]
    max_value = max(df[org].max(), df[dup].max())
    # Redondear el valor máximo al siguiente múltiplo de 5
    rounded_max = 5 * round(max_value / 5)
    with st.sidebar:
        MAX = st.slider("Select Max value", 
                            min_value=0, 
                            max_value=rounded_max+10, 
                            value=rounded_max, 
                            step=1, 
                            format="%.2f", 
                            key="MAX")
    # Subtítulo para ARD Summary
    st.subheader("ARD Summary", divider="gray")
    # Llamar a la función que procesa los datos (asegúrate de que la función esté definida correctamente)
    # Nota: Asegúrate de que el módulo scrptqc esté correctamente importado y las funciones definidas
    if 'scrptqc' in globals():
        filtered_data_ard = scrptqc.filter_calcARD(df, QC, org, dup, Elem, ldl, ARD)
        st.write(filtered_data_ard)
    else:
        st.warning("The scrptqc.filter_calcARD function is not defined or imported.")

    # Subtítulo para HARD Summary
    st.subheader("HARD Summary", divider="gray")
    
    # Llamar a la función para HARD Summary
    if 'scrptqc' in globals():
        #df1 = df.copy()
        filtered_data_hard = scrptqc.filter_calcHARD(df, org, dup, Elem, ldl)
        st.write(filtered_data_hard)
    else:
        st.warning("The scrptqc.filter_calcHARD function is not defined or imported.")

    st.subheader("Final Chart", divider="gray")
    # Subtítulo para gráficos
    if  'scrptqc' in globals():
        # Leer los archivos CSV en DataFrames
        #df = pd.read_csv(uploaded_file)
        dataf = filtered_data_ard.copy()
        datafh = filtered_data_hard.copy()

        # Llamar a la función resumen con los parámetros y los datos
        scrptqc.resumen(dataf, datafh, maxx=MAX, org=org, dup=dup, Elem=Elem, ldl=ldl, ARD=ARD)

        # Mostrar mensaje de éxito
        st.success("Chart generated successfully.")
    else:
        st.warning("Please upload CSVs")
else:
    st.warning("Please upload correct CSVs")