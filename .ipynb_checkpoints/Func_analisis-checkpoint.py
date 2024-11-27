import pandas as pd
import math
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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
                window=f'{lag}{frec}',  # Ventana de tiempo dinámica
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
                window=f'{lag}{frec}',  # Ventana de tiempo dinámica
                min_periods=1         # Asegurar acumulados incluso con pocos datos
            ).mean()
		
	return df1


def randomForest_mostImp(df, target):
    
    X = df.drop(columns=target)  # Variables predictoras
    y = df[target]  # Variable objetivo
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo de Random Forest (ajustes de velocidad)
    rf = RandomForestRegressor (
        n_estimators=100,  # Reducir el número de árboles
        max_depth=10,  # Limitar la profundidad de los árboles
        random_state=42, 
        n_jobs=-1  # Utilizar todos los núcleos del procesador
    )
    rf.fit(X_train, y_train)
    
    # Predicciones
    y_pred = rf.predict(X_test)
    
    # Obtener las importancias de las características
    importances = rf.feature_importances_
    
    # Crear un DataFrame para visualizar las importancias
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    
    # Ordenar las importancias de mayor a menor
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Métricas de calidad del ajuste
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Imprimir las métricas
    print(f"R^2: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return importance_df








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
                'Constant Included': False,
                'modelo': model
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
                'Constant Included': True,
                'modelo': model_const
            })

    # Convertir los resultados a DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df
