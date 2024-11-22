import sqlite3
import pandas as pd
import folium
import math
import matplotlib.pyplot as plt
import numpy as np

def mapRep_coper(pixel):
	conn = sqlite3.connect('BBDD/aguaCHJucar.db')

	cursor = conn.cursor()
	query = f'''
	    SELECT 
	        loc_copernicus.latitude AS latitude_copernicus,
	        loc_copernicus.longitude AS longitude_copernicus,
	        loc_embalses.latitude AS latitude_embalses,
	        loc_embalses.longitude AS longitude_embalses,
	        loc_aemet.latitude AS latitude_aemet,
	        loc_aemet.longitude AS longitude_aemet,
	        loc_rios_canales.latitude AS latitude_rios_canales,
	        loc_rios_canales.longitude AS longitude_rios_canales,
	        -- Información adicional
	        aemet_info.nombre AS nombre_aemet,
	        aemet_info.location_id AS location_id_aemet,
	        rios_canales_info.EstacióndeAforo AS estacion_aforo_rios_canales,
	        rios_canales_info.location_id AS location_id_rios,
	        -- Usar COALESCE para manejar los valores NULL
	        COALESCE(embalses_info.Embalse, 'No hay embalse') AS embalse,
	        COALESCE(embalses_info.location_id, 'No hay embalse') AS location_id_embalse
	    FROM df_pixeles_cercanos p
	    -- Relacionar las localizaciones de la tabla df_pixeles_cercanos con las latitudes y longitudes
	    INNER JOIN locations_id loc_copernicus ON loc_copernicus.location_id = p.location_id_copernicus
	    LEFT JOIN locations_id loc_embalses ON loc_embalses.location_id = p.location_id_embalses
	    INNER JOIN locations_id loc_aemet ON loc_aemet.location_id = p.location_id_aemet
	    INNER JOIN locations_id loc_rios_canales ON loc_rios_canales.location_id = p.location_id_rios_canales
	    -- Relacionar con tablas de información adicional
	    INNER JOIN df_aemet_info aemet_info ON aemet_info.location_id = loc_aemet.location_id
	    INNER JOIN df_rios_canales_info rios_canales_info ON rios_canales_info.location_id = loc_rios_canales.location_id
	    LEFT JOIN df_embalses_info embalses_info ON embalses_info.location_id = loc_embalses.location_id
	    -- Filtrar por location_id_copernicus = {pixel}
	    WHERE p.location_id_copernicus = {pixel};
	'''


	# Ejecutar la consulta
	cursor.execute(query)
	conn.close()

	df_loc = pd.read_sql_query(query, conn)


	# Coordenadas base para centrar el mapa
	map_center = [39.0, -0.5]  # Aproximadamente en la cuenca del Júcar

	# Crear el mapa base
	m = folium.Map(location=map_center, zoom_start=9.5)

	# Añadir el cuadrado de la zona de Copernicus (representando un área más grande)
	for index, row in df_loc.iterrows():
	    # Longitud y latitud del centro de Copernicus
	    lat_copernicus = row['latitude_copernicus']
	    lon_copernicus = row['longitude_copernicus']

	    # Definir un lado del cuadrado basado en 0.5 grados
	    size_lat = 0.5  # 0.5 grados de latitud
	    size_lon = 0.5 * 111.32 * math.cos(math.radians(lat_copernicus)) / 111.32  # Ajustado por la latitud (en grados)

	    additional_lon = 5 / (111.32 * math.cos(math.radians(lat_copernicus))) 
	    # Crear las coordenadas de las esquinas del cuadrado
	    bounds = [
	        [lat_copernicus - size_lat / 2, lon_copernicus - size_lon / 2],  # Esquina inferior izquierda
	        [lat_copernicus + size_lat / 2, lon_copernicus + size_lon / 2 + additional_lon]   # Esquina superior derecha
	    ]
	    
	    # Añadir el cuadrado de Copernicus al mapa
	    folium.Rectangle(
	        bounds=bounds,
	        color='blue',
	        weight=2,
	        fill=True,
	        fill_color='blue',
	        fill_opacity=0,
	        popup='Copernicus Area'
	    ).add_to(m)

	    # Añadir marcador para Aemet
	    folium.Marker(
	        location=[row['latitude_aemet'], row['longitude_aemet']],
	        popup=f"Aemet: {row['nombre_aemet']}; estacion {row['location_id_aemet']}",
	        icon=folium.Icon(color='red')
	    ).add_to(m)

	    # Añadir marcador para Ríos Canales
	    folium.Marker(
	        location=[row['latitude_rios_canales'], row['longitude_rios_canales']],
	        popup= f"Ríos Canales: {row['estacion_aforo_rios_canales']}; estacion {row['location_id_rios']}",
	        icon=folium.Icon(color='purple')
	    ).add_to(m)
	    # Añadir marcador para embalses
	    if df_loc['embalse'][0] != 'No hay embalse':
	        folium.Marker(
	            location=[row['latitude_embalses'], row['longitude_embalses']],
	            popup=f"Embalse: {row['embalse']}; estacion {row['location_id_embalse']}",
	            icon=folium.Icon(color='orange')
	        ).add_to(m)

    return m


def dataExtract(pixel):
	conn = sqlite3.connect('BBDD/aguaCHJucar.db')

	cursor = conn.cursor()
	query = f'''
	        SELECT 
	            d.date,
	            c.total_precipitation,
	            c.skin_temperature,
	            c.evaporation,
	            c.runoff,
	            c.snowfall,
	            c.soil_water_l1,
	            c.soil_water_l2,
	            c.soil_water_l3,
	            c.soil_water_l4,
	            c.high_vegetation_cover,
	            c.low_vegetation_cover 
	            FROM df_copernicus c JOIN df_date d ON d.date_id = c.date_id  WHERE c.location_id = {pixel};
	        '''
	# Ejecutar la consulta
	cursor.execute(query)


	df_c = pd.read_sql_query(query, conn)
	query = f'''
	        SELECT 
	            d.date,
	            aem.tmed,
	            aem.prec,
	            aem.location_id
	            FROM df_aemet aem 
	            JOIN df_date d ON d.date_id = aem.date_id 
	            JOIN df_pixeles_cercanos pc ON pc.location_id_aemet = aem.location_id 
	            WHERE pc.location_id_copernicus =  {pixel};
	        '''
	# Ejecutar la consulta
	cursor.execute(query)


	df_aemet = pd.read_sql_query(query, conn)
	query = f'''
	        SELECT 
	            d.date,
	            rios.quantity_hm3,
	            rios.location_id
	            FROM df_rios_canales rios 
	            JOIN df_date d ON d.date_id = rios.date_id 
	            JOIN df_pixeles_cercanos pc ON pc.location_id_rios_canales = rios.location_id 
	            WHERE pc.location_id_copernicus = {pixel};
	        '''
	# Ejecutar la consulta
	cursor.execute(query)


	df_rios = pd.read_sql_query(query, conn)

	query = f'''
	        SELECT 
	            d.date,
	            emb.quantity_hm3,
	            emb.location_id
	            FROM df_embalses emb 
	            JOIN df_date d ON d.date_id = emb.date_id 
	            JOIN df_pixeles_cercanos pc ON pc.location_id_embalses = emb.location_id 
	            WHERE pc.location_id_copernicus = {pixel};
	        '''
	# Ejecutar la consulta
	cursor.execute(query)


	df_embalses = pd.read_sql_query(query, conn)
	conn.close()

	df_embalses['date'] = pd.to_datetime(df_embalses['date'])
	df_rios['date'] = pd.to_datetime(df_rios['date'])
	df_aemet['date'] = pd.to_datetime(df_aemet['date'])
	df_c['date'] = pd.to_datetime(df_c['date'])
	df_c['soil_water'] = df_c['soil_water_l1'] + df_c['soil_water_l2'] + df_c['soil_water_l3'] + df_c['soil_water_l4']
	df_c = df_c.drop(['soil_water_l1', 'soil_water_l2', 'soil_water_l3','soil_water_l4'], axis = 1)

	df_x = pd.merge(df_c,df_aemet, on='date', how='inner',suffixes=('_copernicus', '_aemet'))

	return df_embalses, df_rios, df_aemet, df_x


def retardAgg_tNat(df,vars,lags, frec):
	"""
	Retardos agregados en tiempo natural
	df = DataFrame
	vars = Variable a desfasar en una lista
	lags = Lista de ints para saber cuántos lags hacer
	frec = frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year)
	"""
	if 'date' in df.index.names:  # Si 'date' está en el índice
    	df = df.reset_index(level='date')  # Restablecer 'date' como columna
	# Crear un índice basado en meses para agrupar por mes
	df[f'{frec}_start'] = df['date'].dt.to_period(f'{frec}').dt.start_time

	# Agrupación mensual para calcular los valores agregados por mes
	monthly_agg = df.groupby(f'{frec}_start')[vars].sum().reset_index()

	# Iterar sobre las variables y calcular acumulados
	for var in vars:
	    for lag in lags:        
	        # Crear acumulados de los últimos N meses
	        monthly_agg[f'{var}_sum_last{lag}{frec}'] = monthly_agg[var].rolling(window=lag).sum()
	# Unir los resultados al DataFrame diario
	df = df.merge(monthly_agg.drop(vars, axis = 1), left_on=f'{frec}_start', right_on=f'{frec}_start', how='left')

	return df

def retardAvg_tNat(df,vars,lags, frec):
	"""
	Retardos promedio en tiempo natural
	df = DataFrame
	vars = Variable a desfasar en una lista
	lags = Lista de ints para saber cuántos lags hacer
	frec = frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year)
	"""
	if 'date' in df.index.names:  # Si 'date' está en el índice
    	df = df.reset_index(level='date')  # Restablecer 'date' como columna
	# Crear un índice basado en meses para agrupar por mes
	df[f'{frec}_start'] = df['date'].dt.to_period(f'{frec}').dt.start_time

	# Agrupación mensual para calcular los valores promedios por mes
	monthly_avg = df.groupby(f'{frec}_start')[vars].mean().reset_index()

	# Iterar sobre las variables y calcular acumulados
	for var in vars:
	    for lag in lags:        
	        # Crear acumulados de los últimos N meses
	        monthly_avg[f'{var}_mean_last{lag}{frec}'] = monthly_avg[var].rolling(window=lag).mean()
	# Unir los resultados al DataFrame diario
	df = df.merge(monthly_avg.drop(vars, axis = 1), left_on=f'{frec}_start', right_on=f'{frec}_start', how='left')

	return df

def retardAgg_tDin(df,vars,lags, frec):
	"""
	Retardos agregados en tiempo dinámico
	df = DataFrame
	vars = Variable a desfasar en una lista
	lags = Lista de ints para saber cuántos lags hacer
	frec = frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year)
	"""
	if 'date' in df.index.names:  # Si 'date' está en el índice
    	df = df.reset_index(level='date')  # Restablecer 'date' como columna
	for var in vars:
        for lag in lags:
            # Crear acumulados dinámicos basados en la frecuencia especificada
            df[f'{var}_sum_last{lag}{frec}'] = df[var].rolling(
                window=f'{lag}{frec}',  # Ventana de tiempo dinámica
                min_periods=1,         # Asegurar acumulados incluso con pocos datos
                on='date'              # Basado en la columna de fecha
            ).sum()
		
	return df

def retardAvg_tDin(df,vars,lags, frec):
	"""
	Retardos prmedios en tiempo dinámico
	df = DataFrame
	vars = Variable a desfasar en una lista
	lags = Lista de ints para saber cuántos lags hacer
	frec = frecuencia temporal de los lags, D = Día, M = Mes, Y = Año(year)
	"""
	if 'date' in df.index.names:  # Si 'date' está en el índice
    	df = df.reset_index(level='date')  # Restablecer 'date' como columna
	for var in vars:
        for lag in lags:
            # Crear acumulados dinámicos basados en la frecuencia especificada
            df[f'{var}_mean_last{lag}{frec}'] = df[var].rolling(
                window=f'{lag}{frec}',  # Ventana de tiempo dinámica
                min_periods=1,         # Asegurar acumulados incluso con pocos datos
                on='date'              # Basado en la columna de fecha
            ).mean()
		
	return df


# Función para calcular la distancia Haversine
def haversine(lat1, lon1, lat2, lon2):
    # Convertir de grados a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Diferencias de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula Haversine
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radio de la Tierra en kilómetros
    R = 6371.0

    # Calcular la distancia
    distance = R * c
    return distance

def pixels_colindantes(pixel, maps = False)
	conn = sqlite3.connect('BBDD/aguaCHJucar.db')

	cursor = conn.cursor()
	query = f'''
	    SELECT 
            p.location_id_copernicus, 
            l1.latitude AS latitude_copernicus, 
            l1.longitude AS longitude_copernicus
        FROM df_pixeles_cercanos p
        LEFT JOIN locations_id l1 ON p.location_id_copernicus = l1.location_id
	'''


	# Ejecutar la consulta
	cursor.execute(query)


	# Obtener el DataFrame con las coordenadas
	df_loc = pd.read_sql_query(query, conn)

	conn.close()

	# Eliminar filas duplicadas en las columnas relevantes (location_id_copernicus, latitude_copernicus, longitude_copernicus)
	df_loc_unique = df_loc[['location_id_copernicus', 'latitude_copernicus', 'longitude_copernicus']].drop_duplicates()

	# Obtener las coordenadas del location_id_copernicus == 189
	target_location = df_loc_unique[df_loc_unique['location_id_copernicus'] == pixel]
	target_lat = target_location['latitude_copernicus'].values[0]
	target_lon = target_location['longitude_copernicus'].values[0]

	# Calcular la distancia a todos los demás location_id_copernicus
	df_loc_unique[f'distance_to_{pixel}'] = df_loc_unique.apply(
	    lambda row: haversine(target_lat, target_lon, row['latitude_copernicus'], row['longitude_copernicus']),
	    axis=1
	)

	# Filtrar los location_id_copernicus colindantes 
	colindantes = df_loc_unique[df_loc_unique[f'distance_to_{pixel}'] < 80]

	if maps == True:

		m = folium.Map(location=[target_lat, target_lon], zoom_start=12)

		# Añadir el marcador para el `location_id_copernicus == pixel`
		folium.Marker(
		    location=[target_lat, target_lon],
		    popup=f'location_id_copernicus == {pixel}',
		    icon=folium.Icon(color='red')
		).add_to(m)

		# Añadir marcadores para los puntos colindantes
		for _, row in colindantes.iterrows():
		    folium.Marker(
		        location=[row['latitude_copernicus'], row['longitude_copernicus']],
		        popup=f'location_id_copernicus == {row["location_id_copernicus"]}, Distancia: {row[f"distance_to_{pixel}"]:.2f} km',
		        icon=folium.Icon(color='blue')
		    ).add_to(m)
		    
		    # Añadir una línea entre `location_id_copernicus == 189` y el punto colindante
		    folium.PolyLine(
		        locations=[(target_lat, target_lon), (row['latitude_copernicus'], row['longitude_copernicus'])],
		        color='green',
		        weight=2.5,
		        opacity=1
		    ).add_to(m)

		return m, colindantes

	else:
		colindantes

def extraccion_colindantes(colindantes):

	locations_colindantes  = colindantes['location_id_copernicus'].to_list()
	locations_str = ', '.join(map(str, locations_colindantes))
	conn = sqlite3.connect('BBDD/aguaCHJucar.db')

	cursor = conn.cursor()
	query = f'''
	        SELECT 
	            d.date,
	            c.*
	            FROM df_copernicus c JOIN df_date d ON d.date_id = c.date_id  WHERE c.location_id IN ({locations_str});
	        '''
	# Ejecutar la consulta
	cursor.execute(query)


	df_colindantes_coper = pd.read_sql_query(query, conn)
	conn.close()

	return df_colindantes_coper


#######################
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




