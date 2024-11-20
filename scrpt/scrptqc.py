import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcol
from sklearn.linear_model import LinearRegression
import streamlit as st
# ## Parametros
def filter_calcARD(dataf, QC, org, dup, Elem, ldl, ARD):
    # Create a copy of the input DataFrame
    df = dataf.copy()
    # Drop rows with missing values in the specified columns
    df_cleaned = df.dropna(subset=[org, dup])
    # Select relevant columns
    selected_columns = df_cleaned.copy()
    #selected_columns = df_cleaned[['DataSet', 'Orig_SampleID', org, 'SampleID', dup]]
    # Define the limit based on the lower detection limit (LDL)
    #ldl
    LD = ldl * 10
    
    # Classify values based on the LDL threshold
    classified = (selected_columns
                  .assign(
                      org_status=lambda x: np.where(x[org] <= LD, 'Fail', 'Ok'),
                      dup_status=lambda x: np.where(x[dup] <= LD, 'Fail', 'Ok')
                  )
                  .assign(union=lambda x: x[['org_status', 'dup_status']].agg('-'.join, axis=1))
                 )
    # Filter out entries where both statuses are 'Fail'
    filtered_data = classified[~classified.union.isin(['Fail-Fail'])]
    # Calculate absolute and relative differences
    filtered_data = (filtered_data
                     .assign(
                         difABS=lambda x: abs(x[org] - x[dup]) / ((x[org] + x[dup]) / 2),
                         difREL=lambda x: (x[org] - x[dup]) / ((x[org] + x[dup]) / 2) * 100,
                         Success='n'
                     )
                     .assign(Success=lambda x: x.Success.mask(x.difABS < ARD, 'y'))
                     #.assign(NuEff=lambda x: x.Success.mask(x.difREL > NE, '
                    )
    # Calculate statistics
    n_total_data = len(df_cleaned)
    n_total = len(filtered_data)
    n_warnings = len(filtered_data[filtered_data['Success'] == 'y'])
    pctDups = (n_warnings / n_total * 100) if n_total > 0 else 0   
    # Mostrar estadísticas
    st.write(f"**QC Category:** {QC}")
    st.write(f"**Element    :** {Elem[0]}")
    st.write(f"**Count      :** {n_total_data}")
    st.write(f"**Count ARD  :** {n_total}")
    st.write(f"**% Dup ARD  :** {round(pctDups, 0)}")    
    return filtered_data #, message

def filter_calcHARD(df, org, dup, Elem,ldl):
    # Create a copy of the input DataFrame
    #df = datafh.copy()
    # Drop rows with missing values in the specified columns
    #df_cleaned = df.dropna(subset=[org, dup])
    # Select relevant columns
    selected_columns = df.copy()
    #selected_columns = df_cleaned[['DataSet', 'Orig_SampleID', org, 'SampleID', dup]]
    # Classify values based on the ldl threshold
    results = (selected_columns
               .assign(
                   org_status=lambda x: np.where(x[org] < ldl, 'Fail', 'Ok'),
                   dup_status=lambda x: np.where(x[dup] < ldl, 'Fail', 'Ok')
               )
               .assign(union=lambda x: x[['org_status', 'dup_status']].agg('-'.join, axis=1))
               )
    # Filter out entries where both statuses are 'Fail'
    filtered_resultsx = results[~results.union.isin(['Fail-Fail'])]

    # Calculate the absolute difference as a percentage
    filtered_results = filtered_resultsx.copy()
    filtered_results['difABS'] = 0.5 * (abs(filtered_results[org] - filtered_results[dup]) / 
                                          ((filtered_results[org] + filtered_results[dup]) / 2))

    # Assign QC and calculate percentile ranks
    filtered_results1 = filtered_results.copy()

    filtered_results1['succs'] = 'QC'
    filtered_results1['percentile'] = filtered_results1.groupby('succs')['difABS'].rank(pct=True)

    # Calculate minimum and 90th percentile of difABS
    percentile_stats = filtered_results1.groupby('succs').agg(
        min_val=('difABS', 'min'),
        percentile_90=('difABS', lambda x: x.quantile(0.9))
    )
    mean_percentile_90 = percentile_stats['percentile_90'].mean()
    n_total = len(filtered_results1)
    # Mostrar estadísticas
    st.write(f"**Count Dup HARD:** {n_total}")
    st.write(f"**HARD Percentil 90%:** {round(mean_percentile_90 * 100, 1)}")
    st.write(f"**LimitLower:** {ldl}")
    return filtered_results1

def resumen(filtered_data_ard, filtered_data_hard, maxx, org, dup, Elem, ldl, ARD):
    # Filtrar datos de ARD y HARD
    #df = filter_calcARD(dataf, QC, org, dup, Elem, ldl, ARD)
    df = filtered_data_ard.copy()
    df1x = df[df['Success'] == 'y']
    df2x = df[df['Success'] == 'n']
    # Cálculo de correlación y R^2
    pearson = df[org].corr(df[dup])
    r_squared = round((pearson**2), 2)
    # Preparar los datos para la regresión lineal
    X = df[org].values[:, np.newaxis]
    y = df[dup].values
    # Estadísticas generales
    n_total = len(df1x)
    n_warnings = len(df1x[df1x['Success'] == 'y'])
    pctDups = (n_warnings / n_total) * 100
    # Visualización
    plt.rcParams['axes.grid'] = True  # Activar grillas en todas las subgráficas
    fig = plt.figure(tight_layout=True, figsize=(12, 10))
    figGrid = gridspec.GridSpec(2, 2)
    # Subgráfico 1: Scatter plot (Regresión Lineal)
    skatter = fig.add_subplot(figGrid[0, 0])
    plot_scatter(skatter, df1x, df2x, org, dup, X, y, r_squared, ARD, Elem, maxx)
    # Subgráfico 2: Mapas (Frecuencia acumulada)
    mapS = fig.add_subplot(figGrid[0, 1])
    plot_map(df, ARD, mapS)
    # Subgráfico 3: MAPD y percentiles
    MapX = fig.add_subplot(figGrid[1, :])
    plot_mapd(filtered_data_hard, MapX)
    # Alinear etiquetas y mostrar figura
    fig.align_labels()
    #plt.savefig(QC+Elem+'.jpg', dpi=600,
    #        bbox_inches='tight', transparent=True)
    #plt.show()
    st.pyplot(fig)
    
def plot_scatter(ax, df1x, df2x, org, dup, X, y, r_squared, ARD, Elem, maxx):
    """Dibuja el gráfico de dispersión con la regresión lineal"""
    ax.plot(df1x[org], df1x[dup], 'ob', markersize=8, alpha=0.5, label='Inliers')
    ax.plot(df2x[org], df2x[dup], '^r', markersize=8, alpha=0.5, label='Outliers')

    model = LinearRegression()
    model.fit(X, y)
    ax.plot(X, model.predict(X), lw=1, color='k')

    # Líneas de referencia
    line = mlines.Line2D([0, 1], [0, 1], lw=2, color='red', label=f'Y=X $r^2$: {round((r_squared * 100),2)}%')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    line1 = mlines.Line2D([0, 1], [0, 1 - ARD], lw=1.5, color='g', label=f'+/-{ARD * 100}')
    transform1 = ax.transAxes
    line1.set_transform(transform1)
    ax.add_line(line1)

    line2 = mlines.Line2D([0, 1], [0, 1 + ARD], lw=1.5, color='g')
    transform1 = ax.transAxes
    line2.set_transform(transform1)
    ax.add_line(line2)

    ax.set_ylabel(f'{Elem[0]} Duplicate', fontsize=12)
    ax.set_xlabel(f'{Elem[0]} Original', fontsize=12)
    ax.set_ylim(ymin=0, ymax=maxx)
    ax.set_xlim(xmin=0, xmax=maxx)
    #ax.yaxis.grid()
    #ax.xaxis.grid()
    ax.legend(loc='lower right')

def plot_map(df, ARD, ax):
    """Dibuja el mapa de frecuencia acumulada"""
    df = df.sort_values("difREL")
    df1 = df[['difREL']].copy()
    xs = df1.count()
    df1.loc[:,'Order'] = np.arange(1, xs.values + 1)
    df1.loc[:,'FrexAcum'] = (df1['Order'] / xs.values) * 100

    ax.plot(df1['difREL'], df1['FrexAcum'], 'ob', markersize=8, alpha=0.5)
    ax.set_ylim(ymin=0, ymax=101)
    ax.set_xlim(xmin=-70, xmax=70)
    ax.axvline(x=ARD * 100, color='g', linestyle='-', linewidth=1)
    ax.axvline(x=-ARD * 100, color='g', linestyle='-', linewidth=1)
    ax.axvline(x=-ARD * 100 / 3, color='r', linestyle='-', linewidth=1)
    ax.axvline(x=ARD * 100 / 3, color='r', linestyle='-', linewidth=1)
    #ax = plt.axes()
    ax.yaxis.grid()
    ax.xaxis.grid()
    ax.set_ylabel('Cumulative Frequency (%)', fontsize=12)
    ax.set_xlabel('Relative Difference (%)', fontsize=12)

def plot_mapd(filtered_data_hard, ax):
    """Dibuja el gráfico de MAPD y percentiles"""
    df1 = filtered_data_hard.copy() #filter_calcHARD(df, org, dup, Elem, ldl)
    xxx = df1.groupby('succs').agg(min_val=('difABS', 'min'), percentile_90=('difABS', lambda x: x.quantile(0.9)))
    per = xxx['percentile_90'].mean()
    maxs = df1['percentile'].max()

    ax.plot(df1['percentile'], df1['difABS'], 'ob', markersize=8, alpha=0.5)
    ax.annotate(f'90th Percentile = {round(per * 100, 1)}%', xy=(0.15, maxs - 0.2), xycoords="data",
                va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
    
    ax.plot([0, 0.9], [per, per], linewidth=2, color="r")
    ax.plot([0.9, 0.9], [0, per], linewidth=2, color="r")
    #ax.xaxis.grid()
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=0, ymax=maxs)
    ax.set_ylabel('Half Absolute Relative Difference', fontsize=12)
    ax.set_xlabel('HARD Rank', fontsize=12)
