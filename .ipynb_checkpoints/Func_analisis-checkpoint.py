import pandas as pd
import math
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations




def retardAgg_tNat(df, vars, lags, frec):
    """
    Retardos agregados en tiempo natural.
    df = DataFrame	
    vars = Variable a desfasar en una lista.
    lags = Lista de ints para saber cuántos lags hacer.
    frec = Frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year).
    """
    df1 = df.copy()
    if 'date' in df1.index.names:  # Si 'date' está en el índice
        df1 = df1.reset_index(level='date')  # Restablecer 'date' como columna
    
    # Crear un índice basado en la frecuencia temporal
    df1[f'{frec}_start'] = df1['date'].dt.to_period(f'{frec}').dt.start_time

    # Agrupación según la frecuencia
    df_agg = df1.groupby(f'{frec}_start')[vars].sum().reset_index()

    # Crear un diccionario para almacenar las nuevas columnas
    lagged_columns = {}

    # Iterar sobre las variables y calcular acumulados
    for var in vars:
        for lag in lags:
            # Calcular acumulados
            col_name = f'{var}_sum_last{lag}{frec}'
            lagged_columns[col_name] = df_agg[var].rolling(window=lag).sum()

    # Convertir el diccionario a un DataFrame
    lagged_df = pd.concat(lagged_columns, axis=1)
    lagged_df[f'{frec}_start'] = df_agg[f'{frec}_start']  # Mantener el índice temporal

    # Unir los resultados al DataFrame original
    df1 = df1.merge(lagged_df, on=f'{frec}_start', how='left')

    return df1

def retardAvg_tNat(df, vars, lags, frec):
    """
    Retardos promedio en tiempo natural.
    df = DataFrame
    vars = Variable a desfasar en una lista
    lags = Lista de ints para saber cuántos lags hacer
    frec = frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year)
    """
    df1 = df.copy()
    if 'date' in df1.index.names:  # Si 'date' está en el índice
        df1 = df1.reset_index(level='date')  # Restablecer 'date' como columna
    
    # Crear un índice basado en la frecuencia temporal
    df1[f'{frec}_start'] = df1['date'].dt.to_period(f'{frec}').dt.start_time

    # Agrupación según la frecuencia para calcular promedios
    monthly_avg = df1.groupby(f'{frec}_start')[vars].mean().reset_index()

    # Crear un diccionario para almacenar las nuevas columnas
    lagged_columns = {}

    # Iterar sobre las variables y calcular promedios acumulados
    for var in vars:
        for lag in lags:
            # Calcular acumulados
            col_name = f'{var}_mean_last{lag}{frec}'
            lagged_columns[col_name] = monthly_avg[var].rolling(window=lag).mean()

    # Convertir el diccionario a un DataFrame
    lagged_df = pd.concat(lagged_columns, axis=1)
    lagged_df[f'{frec}_start'] = monthly_avg[f'{frec}_start']  # Mantener el índice temporal

    # Unir los resultados al DataFrame original
    df1 = df1.merge(lagged_df, on=f'{frec}_start', how='left')

    return df1


def retardAgg_tDin(df,vars,lags):
	"""
	Retardos agregados en tiempo dinámico
	df = DataFrame
	vars = Variable a desfasar en una lista
	lags = Lista de ints para saber cuántos lags hacer
	frec = frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year)
	"""
	df1 = df.copy()
	#if 'date' in df1.index.names:  # Si 'date' está en el índice
		#df1 = df1.reset_index(level='date')  # Restablecer 'date' como columna
	for var in vars:
		for lag in lags:
            # Crear acumulados dinámicos basados en la frecuencia especificada
			df1[f'{var}_sum_last{lag}'] = df1[var].rolling(
                window=f'{lag}',  # Ventana de tiempo dinámica
                min_periods=1         # Asegurar acumulados incluso con pocos datos
            ).sum()
		
	return df1

def retardAvg_tDin(df,vars,lags):
	"""
	Retardos prmedios en tiempo dinámico
	df = DataFrame
	vars = Variable a desfasar en una lista
	lags = Lista de ints para saber cuántos lags hacer
	frec = frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year)
	"""
	df1 = df.copy()
	#if 'date' in df1.index.names:  # Si 'date' está en el índice
		#df1 = df1.reset_index()  # Restablecer 'date' como columna
	for var in vars:
		for lag in lags:
            # Crear acumulados dinámicos basados en la frecuencia especificada
			df1[f'{var}_mean_last{lag}'] = df1[var].rolling(
                window=f'{lag}',  # Ventana de tiempo dinámica
                min_periods=1         # Asegurar acumulados incluso con pocos datos
            ).mean()
		
	return df1

def apply_pca(df, var, frec, var_threshold=95):
    """
    Aplica PCA a las columnas generadas por retardAvg_tNat y retardAgg_tNat
    para una frecuencia específica, añadiendo el porcentaje de varianza explicada
    hasta alcanzar el umbral especificado.
    
    df: DataFrame procesado.
    var: Variable base usada en las funciones de retardos (como 'total_precipitation').
    frec: Frecuencia temporal ('D', 'M', 'Y').
    var_threshold: Umbral de varianza explicada acumulada (%) para decidir el número de componentes principales.
    
    Retorna:
        - Un DataFrame con las componentes principales.
    """
    # Filtrar las columnas relevantes para la frecuencia y la variable
    cols_to_pca = [
        col for col in df.columns 
        if var in col and f'{frec}' in col and ('_sum_' in col or '_mean_' in col)
    ]
    
    if not cols_to_pca:
        print(f"No hay columnas para aplicar PCA con frecuencia {frec} para la variable '{var}'.")
        return None
    
    # Normalizar las columnas antes de PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[cols_to_pca].fillna(0))  # Reemplazar NaN con 0 para evitar problemas
    
    # Aplicar PCA sin límite de componentes
    pca = PCA()
    pca_components = pca.fit_transform(data_scaled)
    
    # Calcular la varianza explicada acumulada
    explained_variance_ratio = pca.explained_variance_ratio_ * 100  # Convertir a porcentaje
    cumulative_explained_variance = explained_variance_ratio.cumsum()  # Cálculo acumulado
    
    # Determinar el número de componentes que alcanzan el umbral de varianza explicada
    n_components = (cumulative_explained_variance <= var_threshold).sum() + 1
    
    # Aplicar PCA nuevamente con el número de componentes necesario
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data_scaled)
    
    # Crear nombres para las componentes principales, incluyendo la variable y frecuencia
    pca_columns = [
        f'{var}_PCA_{frec}_comp{i+1}' for i in range(n_components)
    ]
    
    # Crear un nuevo DataFrame con las componentes principales
    pca_df = pd.DataFrame(pca_components, columns=pca_columns)
    
    # Agregar la clave temporal al nuevo DataFrame si existe en el original
    if 'date' in df.columns:
        pca_df['date'] = df['date'].values
    elif 'date' in df.index.names:
        
        pca_df['date'] = df.index.get_level_values('date')
    
    # Imprimir resultados de la varianza explicada
    print(f"PCA aplicado para variable '{var}' y frecuencia '{frec}'.")
    print(f"Varianza explicada acumulada para los {n_components} componentes: {cumulative_explained_variance[n_components-1]}%")
    
    return pca_df

def process_pca_for_variables(df, variables, frecuencias, var_threshold=90):
    """
    Aplica PCA para cada combinación de variable y frecuencia temporal, 
    y combina los resultados en un único DataFrame.
    
    Args:
        df (pd.DataFrame): El DataFrame original.
        variables (list): Lista de variables base para aplicar PCA.
        frecuencias (list): Lista de frecuencias temporales ('D', 'M', 'Y').
        var_threshold (float): Umbral mínimo de varianza explicada acumulada para las componentes principales.
    
    Returns:
        pd.DataFrame: DataFrame con las componentes principales combinadas.
    """
    # Crear un diccionario para almacenar los resultados de PCA
    pca_results = {}
    
    # Aplicar PCA para cada variable y frecuencia
    for var in variables:
        for frec in frecuencias:
            pca_df = apply_pca(df, var=var, frec=frec, var_threshold=var_threshold)
            if pca_df is not None:
                # Almacenar en el diccionario
                pca_results[f'{var}_{frec}'] = pca_df

    # Combinar los resultados usando 'merge' basado en la columna 'date'
    final_pca_df = pd.DataFrame()
    for key, pca_df in pca_results.items():
        if final_pca_df.empty:
            final_pca_df = pca_df  # Inicializamos con el primer DataFrame
        else:
            # Unir por 'date', con 'how="outer"' para mantener todas las fechas
            final_pca_df = pd.merge(final_pca_df, pca_df, on='date', how='outer')
    
    return final_pca_df







def regresion(X, y, const=1):
    """
    Perform OLS regression.

    Parameters:
    X : DataFrame or ndarray
        Independent variables.
    y : Series or ndarray
        Dependent variable.
    const : int, optional
        Whether to include a constant (default is 1, include constant).

    Returns:
    results : RegressionResults
        Results of the regression.
    """
    
    if const == 1:
        X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results






def regression_analysis(df, features, target):
    """
    Realiza un análisis de regresión lineal sobre el DataFrame proporcionado.

    Args:
    df (pd.DataFrame): DataFrame que contiene las características y la variable dependiente.
    features (list): Lista de strings con los nombres de las características (features).
    target (str): Nombre de la variable dependiente.

    Returns:
    pd.DataFrame: DataFrame con los resultados del análisis de regresión.
    """
    
    # Normalizar los datos
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Variable dependiente
    y = df_scaled[target]
    results = []

    # Iterar sobre el número de características a usar
    for num_features in range(1, len(features) + 1):
        for feature_combination in combinations(features, num_features):
            # Opción 1: Sin constante
            X = df_scaled[list(feature_combination)]
            
            # Ajustar el modelo de regresión lineal sin constante usando statsmodels
            model = sm.OLS(y, X).fit()
            
            # Calcular R^2
            r2 = model.rsquared
            
            # Calcular el número condicional
            number_condition = np.linalg.cond(X)
            
            # Guardar los resultados sin constante
            results.append({
                'Features': feature_combination,
                'R^2': r2,
                'Conditional Number': number_condition,
                'Constant Included': False
            })
            
            # Opción 2: Con constante
            X_const = sm.add_constant(X)  # Añadir constante usando statsmodels
            
            # Ajustar el modelo de regresión lineal con constante
            model_const = sm.OLS(y, X_const).fit()
            
            # Calcular R^2 con constante
            r2_const = model_const.rsquared
            
            # Calcular el número condicional con constante
            number_condition_const = np.linalg.cond(X_const)
            
            # Guardar los resultados con constante
            results.append({
                'Features': feature_combination,
                'R^2': r2_const,
                'Conditional Number': number_condition_const,
                'Constant Included': True
            })

    # Convertir los resultados a DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df
