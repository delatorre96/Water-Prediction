import pandas as pd
import math
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from itertools import combinations
from Func_extr import create_df
import pickle
import os
import logging
logging.basicConfig(level=logging.INFO)
from multiprocessing import Pool

######################################## FASE 0: CALCULOS INICIALES ########################################
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
            #lagged_columns[col_name] = df_agg[var].rolling(window=lag, closed='left').sum()
            lagged_columns[col_name] = df_agg[var].shift(lag)

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
            lagged_columns[col_name] = monthly_avg[var].shift(lag)

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
                window=lag,  # Ventana de tiempo dinámica
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
                window=lag,  # Ventana de tiempo dinámica
                min_periods=1         # Asegurar acumulados incluso con pocos datos
            ).mean()
		
	return df1


def create_dict_retardosMax (retardosMax):
    retardos_dict = {}

    for day in range(retardosMax + 1):
        retardos_dict[day] = {}
        dias_iniciales = [1,2,3,4,5,6,7,15,20]
        dias_retardos = [i+day for i in dias_iniciales]
        retardos_dict[day]['D'] = dias_retardos
        meses_iniciales = [1,2,3,4,6,8,9,10,11,12]
        mes = min(day // 30, len(meses_iniciales))
        meses_retardos = [i+mes for i in meses_iniciales]
        retardos_dict[day]['M'] = meses_retardos
        años_iniciales = [1,2,3,4,5]
        año = min(day // 365, len(años_iniciales)) 
        años_retardos = [i+año for i in años_iniciales]
        retardos_dict[day]['Y'] = años_retardos

    return retardos_dict


def calculate_df_retardos(df,retardosMax):
    '''
    
    '''
    variables = ['total_precipitation', 'skin_temperature', 'evaporation', 'runoff',
               'snowfall', 'high_vegetation_cover', 'low_vegetation_cover',
               'soil_water']
    retardos_dict = create_dict_retardosMax(retardosMax)
    day_lags = retardos_dict[retardosMax]['D']
    month_lags = retardos_dict[retardosMax]['M']
    year_lags = retardos_dict[retardosMax]['Y']
    df = retardAvg_tNat(df, vars = variables, lags = day_lags, frec = 'D')
    df = retardAvg_tNat(df, vars = variables, lags = month_lags, frec = 'M')
    df = retardAvg_tNat(df, vars = variables, lags = year_lags, frec = 'Y')
    df = retardAgg_tNat(df, vars = variables, lags = day_lags, frec = 'D')
    df = retardAgg_tNat(df, vars = variables, lags = month_lags, frec = 'M')
    df = retardAgg_tNat(df, vars = variables, lags = year_lags, frec = 'Y')
    df_lags = df.sort_values('date')
    df_lags = df_lags.set_index('date')
    df_lags = df_lags.select_dtypes(exclude=['datetime64']) #para eliminar columnas residuales de creación de retardos, previamente se establece la fecha como date para no eliminar dicha columna con este método
    df_lags = df_lags.reset_index()
    return df_lags
    
def calculate_df_retardos_Pixel (retardosMax, pixel):
    df = create_df(pixel)        
    df = calculate_df_retardos(df,retardosMax)
    df['pixel'] = pixel
    if 'quantity_hm3_rios' in df.columns:
        df = df.drop(['quantity_hm3','quantity_hm3_embalses'], axis = 1)
        df['quantity_hm3'] = df['quantity_hm3_rios']
        df = df.drop(['quantity_hm3_rios'], axis = 1)
    return df  
    
def apply_pca(df, var, frec, var_threshold=95, imprimir = False):
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
        #print(f"No hay columnas para aplicar PCA con frecuencia {frec} para la variable '{var}'.")
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
    if imprimir:
        print(f"PCA aplicado para variable '{var}' y frecuencia '{frec}'.")
        print(f"Varianza explicada acumulada para los {n_components} componentes: {cumulative_explained_variance[n_components-1]}%")
    
    return pca_df
    
############################################## FASE 1: Crear el data frame para entrenar  ##############################################
def create_df_retardos(retardosMax):
    df_filename = 'df_total_retardos.pkl'
    if os.path.exists(df_filename):
        os.remove(df_filename)
    logging.info(f"Creando retardos para cada pixel...")
    
    # Inicializar un DataFrame vacío para los resultados
    combined_df = pd.DataFrame()

    # Procesar cada píxel y acumular los resultados en `combined_df`
    for pixel in range(176, 301):
        # Generar el DataFrame para el píxel
        df = calculate_df_retardos_Pixel(retardosMax, pixel)
        
        # Verificar si el DataFrame no está vacío
        if not df.empty:
            # Concatenar el nuevo DataFrame con el existente
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            # Eliminar el DataFrame del píxel de la memoria
            del df
    
    # Guardar el DataFrame combinado en el archivo Pickle
    with open(df_filename, 'wb') as f:
        pickle.dump(combined_df, f)
    
    logging.info(f"Retardos calculados con éxito en {df_filename}")
    
    return df_filename

# def calculate_df_retardos_Pixel_wrapper(args):
#     retardosMax, pixel = args
#     return calculate_df_retardos_Pixel(retardosMax, pixel)

# def create_df_retardos(retardosMax):
#     df_filename = 'df_total_retardos.pkl'
#     logging.info(f"Creando retardos para cada pixel...")

#     # Inicializar un DataFrame vacío para los resultados
#     combined_df = pd.DataFrame()

#     # Si el archivo Pickle existe, cargar el DataFrame existente
#     if os.path.exists(df_filename):
#         with open(df_filename, 'rb') as f:
#             combined_df = pickle.load(f)

#     # Procesar los píxeles (ejemplo: 176 a 301)
#     pixels = range(176, 301)

#     # Usar multiprocessing para procesar píxeles en paralelo
#     with Pool() as pool:
#         # Mapear la función de cálculo para cada píxel
#         results = pool.map(calculate_df_retardos_Pixel_wrapper, [(retardosMax, pixel) for pixel in pixels])

#     # Concatenar todos los DataFrames devueltos por los procesos
#     for df in results:
#         if not df.empty:
#             combined_df = pd.concat([combined_df, df], ignore_index=True)

#     # Guardar el DataFrame combinado en el archivo Pickle
#     with open(df_filename, 'wb') as f:
#         pickle.dump(combined_df, f)

#     logging.info(f"Retardos calculados con éxito en {df_filename}")
#     return df_filename



def process_pca_for_variables_pickle(df_filename, var_threshold=80):
    """
    Aplica PCA para cada combinación de variable y frecuencia temporal, 
    y combina los resultados en un único archivo Pickle con todas las combinaciones.
    
    Args:
        df_filename (str): Nombre del archivo Pickle que contiene el DataFrame original.
        var_threshold (float): Umbral mínimo de varianza explicada acumulada para las componentes principales.
    
    Returns:
        str: Nombre del archivo Pickle con los resultados de PCA.
        pd.Series: Fechas originales.
        pd.Series: Pixeles originales.
        pd.Series: Valores de 'quantity_hm3'.
    """
    logging.info("Haciendo PCA...")
    
    # Cargar el DataFrame desde el archivo Pickle
    with open(df_filename, 'rb') as f:
        df = pickle.load(f)
    
    # Definir las variables a procesar
    variables = ['total_precipitation', 'skin_temperature', 'evaporation', 'runoff',
           'snowfall', 'high_vegetation_cover', 'low_vegetation_cover',
           'soil_water']
    fechas = df['date']
    pixeles = df['pixel']
    quantity_hm3 = df['quantity_hm3']
    frecuencias = ['D', 'M', 'Y']
    
    # Inicializar el diccionario para resultados PCA
    pca_results = {}
    
    # Aplicar PCA para cada combinación de variable y frecuencia
    for var in variables:
        for frec in frecuencias:
            key = f'{var}_{frec}'
            # Verificar si esta clave ya fue procesada
            if key not in pca_results:
                pca_df = apply_pca(df, var=var, frec=frec, var_threshold=var_threshold)
                pca_results[key] = pca_df
                #logging.info(f"PCA completado para {key}")
    
    # Guardar los resultados combinados en un archivo Pickle
    pca_filename = 'pca_results.pkl'
    if os.path.exists(pca_filename):
        os.remove(pca_filename)
    with open(pca_filename, 'wb') as f:
        pickle.dump(pca_results, f)
    
    logging.info(f"PCA calculado con éxito en {pca_filename}")
    return pca_filename, fechas, pixeles, quantity_hm3
# def process_pca_for_variables_pickle(df_filename, var_threshold=80):
#     """
#     Aplica PCA para cada combinación de variable y frecuencia temporal en paralelo,
#     y combina los resultados en un único archivo Pickle con todas las combinaciones.
#     """
#     logging.info("Haciendo PCA...")
    
#     # Cargar el DataFrame desde el archivo Pickle
#     with open(df_filename, 'rb') as f:
#         df = pickle.load(f)
    
#     # Definir las variables a procesar
#     variables = [i for i in df.columns if i not in ['pixel', 'quantity_hm3', 'date']]
#     frecuencias = ['D', 'M', 'Y']
#     fechas = df['date']
#     pixeles = df['pixel']
#     quantity_hm3 = df['quantity_hm3']
    
#     # Inicializar el diccionario para resultados PCA
#     pca_filename = 'pca_results.pkl'
#     if os.path.exists(pca_filename):
#         # Si el archivo Pickle existe, cargar los resultados existentes
#         with open(pca_filename, 'rb') as f:
#             pca_results = pickle.load(f)
#     else:
#         # Si no existe, inicializar como un diccionario vacío
#         pca_results = {}

#     # Crear todas las combinaciones de variables y frecuencias
#     combinations = [(df, var, frec, var_threshold) for var in variables for frec in frecuencias if f'{var}_{frec}' not in pca_results]

#     # Procesar en paralelo las combinaciones
#     with Pool() as pool:
#         results = pool.map(process_pca_wrapper, combinations)

#     # Agregar los resultados al diccionario
#     for key, pca_df in results:
#         pca_results[key] = pca_df

#     # Guardar los resultados combinados en un archivo Pickle
#     with open(pca_filename, 'wb') as f:
#         pickle.dump(pca_results, f)

#     logging.info(f"PCA calculado con éxito en {pca_filename}")
#     return pca_filename, fechas, pixeles, quantity_hm3


# def process_pca_wrapper(args):
#     """
#     Wrapper para procesar PCA en paralelo.
    
#     Args:
#         args (tuple): Contiene (df, var, frec, var_threshold).
    
#     Returns:
#         tuple: Clave de la combinación y DataFrame resultante de PCA.
#     """
#     df, var, frec, var_threshold = args
#     key = f'{var}_{frec}'
#     pca_df = apply_pca(df, var=var, frec=frec, var_threshold=var_threshold)
#     return key, pca_df

def create_df_PCA (retardosMax):
    df_filename = create_df_retardos(retardosMax)
    pca_filename, fechas, pixeles, quantity_hm3 = process_pca_for_variables_pickle(df_filename)
    with open(pca_filename, 'rb') as f:
        pca_results = pickle.load(f)
    df_pca = pd.concat(pca_results.values(),axis=1)
    df_pca = df_pca.drop(['date'],axis = 1)
    new_columns = pd.DataFrame({'date' : fechas,'quantity_hm3': quantity_hm3, 'pixel': pixeles})
    df_pca = pd.concat([df_pca, new_columns], axis=1)
    #df_pca['date'] = fechas
    #df_pca['quantity_hm3'] = quantity_hm3
    #df_pca['pixel'] = pixeles
    df_pca = df_pca.set_index('date')
    with open('df_pca.pkl', 'wb') as f:
        pickle.dump(df_pca, f) 
    
    return df_pca
    
############################################## FASE 2: Entrenamiento de modelos de predicción  ##############################################
def entrenarModelo(retardosMax, modelo):
    df = create_df_PCA (retardosMax)
    logging.info(f"Entrenando modelo con {modelo}...")
    if modelo == 'gradient_boosting':
        xgb_model = gradient_boosting(df)
        return xgb_model
    elif modelo == 'randomForest':
        randomForest = randomForest(df)
        return randomForest
def gradient_boosting(df):
    X = df.drop(['quantity_hm3'], axis=1)  
    y = df['quantity_hm3']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    xgb_model = XGBRegressor(
        n_estimators=50,
        max_depth=20,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(train_r2)
    print(test_r2)
    logging.info(f"Modelo gradient boosting entrenado con éxito")
    return xgb_model

def randomForest (df):
    X = df.drop(['quantity_hm3'], axis=1)  # Excluir 'date' si no se usa explícitamente
    y = df['quantity_hm3']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # Modelo de Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,          # Cantida de árboles para capturar patrones
        max_depth=30,            # Límite de profundidad
        min_samples_split=10,       # Divisiones 
        min_samples_leaf=5,        # Permite hojas pequeñas
        random_state=42,           # Reproducibilidad
        bootstrap=True            # 
    )
    
    # Entrenar en todos los datos
    rf.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logging.info(f"Modelo gradient boosting entrenado con éxito")
    print(train_r2)
    print(test_r2)
    
    return rf

############################################## FASE 3: Crear data frame para la predicción  ##############################################

def create_df_retardos_prediccion(retardosMax):
    '''
    Para validar modelo haciendo una predicción ante todos los pixels.
    Imprescindible columna pixel 
    df es el data frame de la extraccion incial
    componentes pasarlo con la funcion extract_n_components(df.columns) siendo df el data frame con el PCA pasado
    '''
    logging.info(f"Creando retardos de la predicción...")
    df_filename = 'df_total_retardos.pkl'
    with open(df_filename, 'rb') as f:
        df = pickle.load(f)
    df2 = df.reset_index()
    pixeles = df['pixel'].unique()
    
    df_preds_list = []
    for pixel in pixeles:
        df_pixel = df2[df2['pixel'] == pixel]
        df_pixel = df_pixel.drop('pixel',axis = 1)
        fecha_inicial  = df_pixel['date'].max()
        fecha_final = fecha_inicial+ pd.Timedelta(days=retardosMax) 
        df_date_range = pd.DataFrame(pd.date_range(start=fecha_inicial, end=fecha_final, freq='D'), columns=['date'])
        df_pixel = pd.merge(df_date_range, df_pixel, on='date', how='outer').sort_values('date')
        df_pixel = create_df_retardos(df_pixel,retardosMax)
        df_pixel['pixel'] = pixel
        df_preds_list.append(df_pixel)       
    del df2, df
    df_preds = pd.concat(df_preds_list)
    logging.info(f"Retardos de la predicción calculados con éxito")
    return df_preds
    
def create_df_PCA_prediccion(retardosMax):
    logging.info(f"Calculando PCA de los retardos...")
    with open('df_pca.pkl', 'rb') as f:
        df_pca = pickle.load(f)
    componentes = extract_n_components(df_pca.columns)
    del df_pca
    df_preds = create_df_retardos_prediccion(retardosMax)
    fecha_inicial  = df_preds['date'].max()
    fecha_final = df_preds+ pd.Timedelta(days=retardosMax) 
    pixels_completo = df_preds['pixel']
    df_preds.drop('pixel', axis = 1, inplace =True)
    df_preds.set_index('date', inplace = True)
    df_pca_pred = process_pca_custom_components(df = df_preds, variables = variables, frecuencias  = ['D', 'M', 'Y'], n_components_dict = componentes, imprimir=False)
    df_pca_pred['pixel'] = pixels_completo
    df_pca_pred = df_pca_pred.reset_index()
    df_pca_pred = pca_df_pixel[pca_df_pixel['date'] >= fecha_inicial].set_index('date')
    logging.info(f"Data frame para la predicción creado con éxito")
    return df_pca_pred


def process_pca_custom_components(df, variables, frecuencias, n_components_dict, imprimir=False):
    """
    Aplica PCA para cada combinación de variable y frecuencia temporal, con un número de componentes especificado.
    
    Args:
        df (pd.DataFrame): El DataFrame original.
        variables (list): Lista de variables base para aplicar PCA.
        frecuencias (list): Lista de frecuencias temporales ('D', 'M', 'Y').
        n_components_dict (dict): Diccionario donde las claves son combinaciones 'var_frec' y los valores el número de componentes deseados.
                                  Ejemplo: {'total_precipitation_D': 3, 'evaporation_M': 2}
        imprimir (bool): Si True, imprime detalles sobre el PCA aplicado.
    
    Returns:
        pd.DataFrame: DataFrame con las componentes principales combinadas.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import pandas as pd

    # Crear un diccionario para almacenar los resultados de PCA
    pca_results = []

    for var in variables:
        for frec in frecuencias:
            # Identificar las columnas relevantes para la variable y frecuencia
            cols_to_pca = [
                col for col in df.columns 
                if var in col and f'{frec}' in col and ('_sum_' in col or '_mean_' in col)
            ]

            if not cols_to_pca:
                if imprimir:
                    print(f"No hay columnas para PCA en '{var}_{frec}'.")
                continue

            # Normalizar los datos
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df[cols_to_pca].fillna(0))

            # Determinar la cantidad de componentes para esta variable y frecuencia
            key = f'{var}_{frec}'
            n_components = n_components_dict.get(key, None)

            if n_components is None:
                if imprimir:
                    print(f"Cantidad de componentes no especificada para '{key}', saltando.")
                continue

            # Aplicar PCA
            pca = PCA(n_components=n_components)
            pca_components = pca.fit_transform(data_scaled)

            # Crear nombres para las componentes principales
            pca_columns = [f'{var}_PCA_{frec}_comp{i+1}' for i in range(n_components)]

            # Crear DataFrame con las componentes principales
            pca_df = pd.DataFrame(pca_components, columns=pca_columns)

            # Agregar la clave temporal al nuevo DataFrame si existe en el original
            if 'date' in df.columns:
                pca_df['date'] = df['date'].values
            elif 'date' in df.index.names:
                pca_df['date'] = df.index.get_level_values('date')

            # Guardar resultados
            pca_results.append(pca_df)

            # Imprimir información opcional
            if imprimir:
                explained_variance = pca.explained_variance_ratio_.sum() * 100
                print(f"PCA para '{key}': {n_components} componentes principales seleccionadas. Varianza explicada: {explained_variance:.2f}%.")

    # Combinar los resultados en un único DataFrame

    final_pca_df = pd.concat(pca_results, axis = 1)

    return final_pca_df

def extract_n_components(columns):
    """
    Convierte una lista de columnas con formato PCA en un diccionario
    que contiene la cantidad máxima de componentes por variable y frecuencia.

    Parameters:
        columns (list): Lista de nombres de columnas en formato '<variable>_PCA_<frecuencia>_comp<número>'.

    Returns:
        dict: Diccionario con la cantidad máxima de componentes por variable y frecuencia.
    """
    n_components_dict = {}

    for col in columns:
        parts = col.split('_PCA_')
        variable = parts[0]
        frequency, component = parts[1].split('_comp')
        key = f"{variable}_{frequency}"
        n_components_dict[key] = max(n_components_dict.get(key, 0), int(component))

    return n_components_dict

############################################## FASE 3: Predicción y validación ##############################################

def hacerPrediccion(retardosMax, modelo):
    logging.info(f"Se procede a calcular la predicción cantidad de agua...")
    df = create_df_PCA_prediccion(retardosMax)
    predicciones = modelo(df)
    df_predict = df.reset_index()[['date','pixel']]
    df_predict['quantity_hm3_pred'] = predicciones
    logging.info(f"Predicciones de cantidad de agua calculadas con éxito")
    return df_predict

def datos_validacion_general():
    url = "https://aps.chj.es/down/CSV/F2796_Rios_y_Canales_ROEA.zip"
    logging.info(f"Extrayendo datos de la web {url} para validar as predicciones")
    import requests
    import zipfile
    import io
    
    with open('df_pca.pkl', 'rb') as f:
        df_pca = pickle.load(f)
    fecha_inicial = df_pca['date'].max()
    del df_pca
    # Paso 1: Descargar el archivo ZIP
    response = requests.get(url)
    response.raise_for_status()  # Verificar si la descarga fue exitosa
    
    # Paso 2: Cargar el contenido del ZIP en memoria
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    
    
    # Paso 4: Leer el archivo "F2796_D2_Serie día.csv"
    target_file = "F2796_D2_Serie día.csv"
    if target_file in zip_file.namelist():
        with zip_file.open(target_file) as file:
            # Leer el archivo CSV directamente como DataFrame
            df_rios_canales = pd.read_csv(file, sep=";", encoding="latin1")  # Ajusta el separador y la codificación si es necesario
    
    else:
        print(f"El archivo '{target_file}' no se encuentra en el ZIP.")
    df_rios_canales = df_rios_canales.rename(columns = {'Cód. CHJ' : 'id_station','Fecha' : 'date','Cantidad (hm³)' : 'quantity_hm3'})
    df_rios_canales = df_rios_canales[['id_station', 'date','quantity_hm3']]
    df_rios_canales['date'] = pd.to_datetime(df_rios_canales['date'], format='%d-%m-%Y %H:%M:%S')
    df_rios_canales = df_rios_canales.dropna()
    df_rios_canales['quantity_hm3'] = df_rios_canales['quantity_hm3'].str.replace(',','.').astype('float')
    id_stations_list = []
    for pixel in range(176,301):
        id_stations = extract_infos (pixel)
        id_stations['pixel'] = pixel
        id_stations_list.append(id_stations)     
    id_stations_df = pd.concat(id_stations_list)
    df_rios_canales_valid = df_rios_canales[df_rios_canales['date'] > fecha_inicial]
    id_stations_df = id_stations_df[['id_station_rios_canales','pixel']].drop_duplicates()
    id_stations_df= id_stations_df.rename(columns = {'id_station_rios_canales' : 'id_station'})
    df_rios_canales_valid = pd.merge(df_rios_canales_valid, id_stations_df, on = 'id_station')
    return df_rios_canales_valid

def predecirYvalidar(retardosMax, modelo):
    model = entrenarModelo(retardosMax, modelo)
    logging.info(f"Validando resultados...")
    df_predict = hacerPrediccion(retardosMax, model)
    df_valid = datos_validacion_general()
    df = pd.merge(df_predict,df_valid, on =['date','pixel'], how = 'inner').groupby('date').drop('pixel', axis = 1)
    plt.figure(figsize=(10, 6))
    plt.plot(df['quantity_hm3'], label = 'Real')
    plt.plot(df['quantity_hm3_pred'], label = 'Predicción')
    plt.legend()
    plt.grid()
    plt.show()
    mae = np.mean(abs(df_validacion['quantity_hm3'] - df_validacion['quantity_hm3_pred']))
    print(mae)






