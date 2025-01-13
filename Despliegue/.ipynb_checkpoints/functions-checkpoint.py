import sqlite3
import pandas as pd
import xarray as xr
import cdsapi
import os
import zipfile
import sqlite3
from datetime import datetime, timedelta
from calendar import monthrange
import requests
import zipfile
import io
from Func_extr import extract_infos
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
import time
from tqdm import tqdm

##### EXTRACCIÓN DE COPERMICUS

def preprocesarDataFrame(df):
    df = df[['valid_time', 'latitude', 'longitude', 'tp', 'skt', 'e', 'ro', 'sf',
         'swvl1', 'swvl2', 'swvl3', 'swvl4', 'cvh', 'cvl', 'tvh', 'tvl']]
    df = df.rename(columns={
        'valid_time': 'date',
        'tp': 'total_precipitation',
        'skt': 'skin_temperature',
        'e': 'evaporation',
        'ro': 'runoff',
        'sf': 'snowfall',
        'swvl1': 'soil_water_l1',
        'swvl2': 'soil_water_l2',
        'swvl3': 'soil_water_l3',
        'swvl4': 'soil_water_l4',
        'cvh': 'high_vegetation_cover',
        'cvl': 'low_vegetation_cover',
        'tvh': 'type_high_vegetation',
        'tvl': 'type_low_vegetation'
    })
    # Convertir la columna 'date' a formato de fecha
    df['date'] = pd.to_datetime(df['date']).dt.date
    # Realizar las agregaciones
    agg_funcs = {
        'total_precipitation': 'sum',
        'skin_temperature': 'mean',
        'evaporation': 'sum',
        'runoff': 'sum',
        'snowfall': 'sum',
        'soil_water_l1': 'sum',
        'soil_water_l2': 'sum',
        'soil_water_l3': 'sum',
        'soil_water_l4': 'sum',
        'high_vegetation_cover': 'mean',
        'low_vegetation_cover': 'mean',
        'type_high_vegetation': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'type_low_vegetation': lambda x: x.mode()[0] if not x.mode().empty else np.nan
    }
    df = df.groupby(['latitude', 'longitude', 'date']).agg(agg_funcs).reset_index()
    return df


def downloadMesCopernicus(days, year, month):
    dataset = "reanalysis-era5-single-levels"
    request = {
        'product_type': ['ensemble_mean'],
        'variable': [
            'total_precipitation', 'skin_temperature', 'evaporation', 
            'runoff', 'snowfall', 'volumetric_soil_water_layer_1', 
            'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 
            'volumetric_soil_water_layer_4', 'high_vegetation_cover', 
            'low_vegetation_cover', 'type_of_high_vegetation', 'type_of_low_vegetation'
        ],
        'year': year,
        'month': month,
        'day': days,
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
        'area': [40.5, -2.2, 38.1, 0.5]  # Cuenca hidrográfica del Júcar +/-.
    }
    
    client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",  # URL de producción
    key="3ca60b29-89a8-4614-88a3-2b4113bfc5f8"
)
    
    try:
        file_name = client.retrieve(dataset, request).download()
        return file_name
    except requests.exceptions.HTTPError as e:
        print(f"Error en la descarga: {e}")
        return None
    
def procesarZip(zip_file_name):
    dataFrames = []
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        # Listar los archivos en el ZIP
        archivos = zip_ref.namelist()
        for archivo_nc in archivos: 
            # Extraer el archivo 'instant'
            zip_ref.extract(archivo_nc)
            print(f"Archivo '{archivo_nc}' extraído.")
            ds = xr.open_dataset(archivo_nc)
            df = ds.to_dataframe().reset_index()
            dataFrames.append (df)
            ds.close()
            os.remove(archivo_nc)
    df1 = dataFrames[1]
    df1 = df1.drop(['latitude', 'longitude', 'valid_time'], axis = 1)
    df = pd.concat([dataFrames[0],df1], axis = 1)
    df = preprocesarDataFrame(df)
    fechaMin = df['date'].min()
    fechaMax = df['date'].max()
    print(f'Datos de coeprnicus con fechas de {fechaMin} a {fechaMax} descargadas correctamente')
    os.remove(zip_file_name)
    return df

def extraccionCopernicus (days,year, month):
    
    zip_file_name = downloadMesCopernicus(days,year, month)
    print(f"Archivo descargado: {zip_file_name}")
    if zip_file_name is None:
        return None
    else:
        df = procesarZip(zip_file_name)
    return df

def extraerUltimasFechasCopernicus():
    conn = sqlite3.connect('aguaCHJucar.db')
    
    cursor = conn.cursor()
    
    query = f'''
        SELECT 
        c.date_id, d.date
        FROM df_copernicus c JOIN df_date d ON  c.date_id = d.date_id;
    '''
    # Ejecutar la consulta
    cursor.execute(query)
    df_date = pd.read_sql_query(query, conn)
    conn.close()
    df_date['date'] = pd.to_datetime(df_date['date']).dt.date
    ultima_fecha = df_date['date'].max()
    return ultima_fecha
    
def extraerUltimasFechasRios():
    conn = sqlite3.connect('aguaCHJucar.db')
    
    cursor = conn.cursor()
    
    query = f'''
        SELECT 
        c.date_id, d.date
        FROM df_rios_canales c JOIN df_date d ON  c.date_id = d.date_id;
    '''
    # Ejecutar la consulta
    cursor.execute(query)
    df_date = pd.read_sql_query(query, conn)
    conn.close()
    df_date['date'] = pd.to_datetime(df_date['date']).dt.date
    ultima_fecha = df_date['date'].max()
    return ultima_fecha    
def generar_diferencias_mes_a_mes(fecha1 ,fecha2):
    """
    Genera una lista de días, meses y años entre dos fechas mes a mes,
    considerando la fecha inicial desde su día específico y el día final.

    Parámetros:
    - fecha_actual: Fecha inicial en formato datetime.date.
    - fecha_futura: Fecha final en formato datetime.date.

    Retorno:
    - Lista de tuplas (días, año, mes), donde:
      - días: lista de strings con los días del mes ('15', '16', ..., '30')
      - año: año correspondiente
      - mes: mes correspondiente
    """
    diferencias = []
    fecha1 = fecha1  + timedelta(days=1)
    fecha_inicio = fecha1  

    while fecha_inicio <= fecha2:
        # Obtener año y mes actuales
        year = fecha_inicio.year
        month = fecha_inicio.month

        # Si estamos en el primer mes, iniciar desde el día específico de la fecha actual
        if fecha_inicio == fecha1:
            start_day = fecha_inicio.day
        else:
            start_day = 1
        
        # Para el último mes (fecha_futura), restringir al día final
        if fecha_inicio.month == fecha2.month:
            end_day = fecha2.day
        else:
            # Si no es el último mes, el mes completo
            end_day = monthrange(year, month)[1]

        # Calcular los días del mes actual, considerando start_day y end_day
        days = [f"{day:02d}" for day in range(start_day, end_day + 1)]
        
        # Agregar a la lista de diferencias
        diferencias.append((days, year, month))

        # Avanzar al siguiente mes
        if month == 12:
            fecha_inicio = fecha_inicio.replace(year=year + 1, month=1, day=1)
        else:
            fecha_inicio = fecha_inicio.replace(month=month + 1, day=1)
    
    return diferencias

def fechasActualizarCopernicus():
    fecha1 = extraerUltimasFechasCopernicus()
    fecha2 = datetime.now().date()
    fechas_new = generar_diferencias_mes_a_mes(fecha1 ,fecha2)
    return fechas_new

def actualizarTablaCopernicus():
    fechas_new = fechasActualizarCopernicus()
    dataFrames = []
    for i in fechas_new:
        days = i[0]
        month = i[2]
        year = i[1]
        df = extraccionCopernicus (days,year, month)
        if df is None:
            pass
        else:
            dataFrames.append(df)
    df = pd.concat(dataFrames)
    return df

##### EXTRACCIÓN DE LA CUENCA DEL JUCAR
def extraccionRiosCHJ():
    url = "https://aps.chj.es/down/CSV/F2796_Rios_y_Canales_ROEA.zip"
    fecha_inicial = pd.to_datetime(extraerUltimasFechasRios())
    response = requests.get(url)
    response.raise_for_status()  # Verificar si la descarga fue exitosa
    
    # Paso 2: Cargar el contenido del ZIP en memoria
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
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
        id_stations['location_id'] = pixel
        id_stations_list.append(id_stations)     
    id_stations_df = pd.concat(id_stations_list)
    df_rios_canales = df_rios_canales[df_rios_canales['date'] > fecha_inicial]
    id_stations_df = id_stations_df[['id_station_rios_canales','location_id_rios']].drop_duplicates()
    id_stations_df= id_stations_df.rename(columns = {'id_station_rios_canales' : 'id_station', 'location_id_rios': 'location_id'})
    df_rios_canales = pd.merge(df_rios_canales, id_stations_df, on = 'id_station')
    return df_rios_canales

# def extraccionEmbalsesCHJ():
#     url = "https://aps.chj.es/down/CSV/F2797_Embalses_ROEA.zip"
#     fecha_inicial = pd.to_datetime(extraerUltimasFechasCopernicus())
#     response = requests.get(url)
#     response.raise_for_status()  # Verificar si la descarga fue exitosa
    
#     # Paso 2: Cargar el contenido del ZIP en memoria
#     zip_file = zipfile.ZipFile(io.BytesIO(response.content))
#     target_file = "F2797_D1_Serie mes.xlsx"
#     if target_file in zip_file.namelist():
#         with zip_file.open(target_file) as file:
#             # Leer el archivo CSV directamente como DataFrame
#             df_embalses = pd.read_excel(file)  # Ajusta el separador y la codificación si es necesario
    
#     else:
#         print(f"El archivo '{target_file}' no se encuentra en el ZIP.")
#     df_embalses = df_embalses.rename(columns = {'Cód. Embalse' : 'id_station','Fecha' : 'date','Volumen (hm³)' : 'quantity_hm3'})
#     df_embalses = df_embalses[['id_station', 'date','quantity_hm3']]
#     df_embalses['date'] = pd.to_datetime(df_rios_canales['date'], format='%d-%m-%Y %H:%M:%S')
#     df_embalses = df_rios_canales.dropna()
#     df_embalses['quantity_hm3'] = df_embalses['quantity_hm3'].str.replace(',','.').astype('float')
#     id_stations_list = []
#     for pixel in range(176,301):
#         id_stations = extract_infos (pixel)
#         id_stations['location_id'] = pixel
#         id_stations_list.append(id_stations)     
#     id_stations_df = pd.concat(id_stations_list)
#     df_embalses = df_embalses[df_embalses['date'] > fecha_inicial]
#     id_stations_df = id_stations_df[['id_station_rios_canales','location_id_embalse']].drop_duplicates()
#     id_stations_df= id_stations_df.rename(columns = {'id_station_embalse' : 'id_station'})
#     df_embalses = pd.merge(df_embalses, id_stations_df, on = 'id_station')
#     return df_embalses

def preparacionIngesta(df_copernicus, df_rios):
    #tabla fechas
    df_copernicus['date'] = pd.to_datetime(df_copernicus['date'])
    dates1 = df_copernicus['date'].dropna().drop_duplicates().reset_index(drop = True)
    df_date1 = pd.DataFrame({'date': dates1})
    df_date1['date_id'] = df_date1['date'].dt.strftime('%Y%m%d').astype(int)
    
    df_rios['date'] = pd.to_datetime(df_rios['date'])
    dates2 = df_rios['date'].dropna().drop_duplicates().reset_index(drop = True)
    df_date2 = pd.DataFrame({'date': dates2})
    df_date2['date_id'] = df_date2['date'].dt.strftime('%Y%m%d').astype(int)
    df_date = pd.concat([df_date1, df_date2])
    df_date = df_date.drop_duplicates().reset_index(drop = True)
    
    #tabla df_copernicus
    conn = sqlite3.connect('aguaCHJucar.db')
    
    cursor = conn.cursor()
    query = f'''
        SELECT * FROM locations_id WHERE Type == 'Copernicus'
            ;
    '''
    
    df_loc = pd.read_sql_query(query, conn)
    cursor.execute(query)
    conn.close()
    
    df_copernicus = pd.merge(df_copernicus, df_loc[['latitude', 'longitude','location_id']], on = ['latitude', 'longitude'], how = 'inner')
    
    df_copernicus = pd.merge(df_copernicus, df_date, on = ['date'], how = 'left')
    df_copernicus = df_copernicus.drop(['latitude', 'longitude', 'date'], axis = 1)
    
    #tabla ríos
    df_rios = pd.merge(df_rios, df_date, on = 'date', how = 'left')
    df_rios = df_rios[['quantity_hm3','location_id','date_id']]
    return df_copernicus, df_rios, df_date

def ingesta(df_copernicus, df_rios, df_date):
    # Conexión a la base de datos
    connection = sqlite3.connect("aguaCHJucar.db")
    cursor = connection.cursor()
    
    try:
        df_date["date"] = df_date["date"].astype(str)
        # Inserciones para df_copernicus
        for row in df_copernicus.itertuples(index=False):
            try:
                cursor.execute("""
                INSERT OR IGNORE INTO df_copernicus (
                    total_precipitation,
                    skin_temperature, evaporation, runoff, snowfall,
                    soil_water_l1, soil_water_l2, soil_water_l3, soil_water_l4,
                    high_vegetation_cover, low_vegetation_cover,
                    type_high_vegetation, type_low_vegetation,location_id, date_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row)
            except sqlite3.Error as e:
                print(f"Error al insertar en df_copernicus: {e}, Fila: {row}")
    
        # Inserciones para df_rios_canales
        for row in df_rios.itertuples(index=False):
            try:
                cursor.execute("""
                INSERT OR IGNORE INTO df_rios_canales (
                    quantity_hm3, location_id, date_id
                ) VALUES (?, ?, ?)
                """, row)
            except sqlite3.Error as e:
                print(f"Error al insertar en df_rios_canales: {e}, Fila: {row}")
    
        # Inserciones para df_date
        for row in df_date.itertuples(index=False):
            try:
                cursor.execute("""
                INSERT OR IGNORE INTO df_date (
                    date, date_id
                ) VALUES (?, ?)
                """, row)
            except sqlite3.Error as e:
                print(f"Error al insertar en df_date: {e}, Fila: {row}")
    
        # Confirmar todas las inserciones
        connection.commit()
        print("Inserciones realizadas con éxito.")
    
    except sqlite3.Error as e:
        # Manejo de errores con rollback en caso de fallo
        connection.rollback()
        print(f"Error general: {e}")
    
    finally:
        # Cerrar la conexión
        connection.close()

def actualizacionBBDD():
    df_copernicus = actualizarTablaCopernicus()
    df_rios = extraccionRiosCHJ()
    df_copernicus, df_rios, df_date = preparacionIngesta(df_copernicus, df_rios)
    ingesta(df_copernicus, df_rios, df_date)

def extraer_datosBBDD():
    logging.info(f"Extrayendo datos de la Base de datos...")
    conn = sqlite3.connect("aguaCHJucar.db")
    cursor = conn.cursor()
    
    # Inspecciona la tabla df_copernicus
    query = """SELECT 
            d.date,
            quantity_hm3,
            pc.location_id_copernicus AS location_id
            FROM df_rios_canales rios 
            JOIN df_date d ON d.date_id = rios.date_id 
            JOIN df_pixeles_cercanos pc ON pc.location_id_rios_canales = rios.location_id;
            """
    cursor.execute(query)
    df_rios_canales = pd.read_sql_query(query, conn)
    logging.info(f"Datos de ríos extraídos")
    # Inspecciona la tabla df_copernicus
    query = """SELECT 
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
                c.low_vegetation_cover,
                c.location_id
                FROM df_copernicus c JOIN df_date d ON d.date_id = c.date_id;"""
    cursor.execute(query)
    df_coper = pd.read_sql_query(query, conn)
    conn.close()
    logging.info(f"Datos climáticos de copernicus extraídos")
    df_coper['date'] = pd.to_datetime(df_coper['date'])
    df_rios_canales['date'] = pd.to_datetime(df_rios_canales['date'])
    df_rios_canales = df_rios_canales.groupby(['date', 'location_id']).sum().reset_index()
    df = pd.merge(df_coper, df_rios_canales, on = ['date', 'location_id'], how = 'inner')
    df['soil_water'] = df['soil_water_l1'] + df['soil_water_l2'] + df['soil_water_l3'] + df['soil_water_l4']
    df = df.drop(['soil_water_l1', 'soil_water_l2', 'soil_water_l3','soil_water_l4'], axis = 1)
    
    return df


##################################CALCULOS
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

def calculoRetardos (df, retardosMax):
    logging.info(f"Calculando retardos diarios, mensuales y anuales para {retardosMax} días...")
    combined_df = []
    for pixel in df['location_id'].unique():
        df_pixel = df[df['location_id'] == pixel]
        df_pixel = calculate_df_retardos(df = df_pixel,retardosMax = 50)
        combined_df.append(df_pixel)
    df_retard = pd.concat(combined_df)
    df_retard.reset_index(drop = True, inplace = True)
    return df_retard
def apply_pca(df, var, frec, var_threshold=80, imprimir = False):
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
    
    
    # Imprimir resultados de la varianza explicada
    if imprimir:
        print(f"PCA aplicado para variable '{var}' y frecuencia '{frec}'.")
        print(f"Varianza explicada acumulada para los {n_components} componentes: {cumulative_explained_variance[n_components-1]}%")
    
    return pca_df

def process_pca_for_variables(df, var_threshold=80):
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
    logging.info(f"Elaborando análisis de componentes principales (PCA)...")
    dates = df['date']
    location_id = df['location_id']
    quantity_hm3 = df['quantity_hm3']
    # Definir las variables a procesar
    variables = ['total_precipitation', 'skin_temperature', 'evaporation', 'runoff',
           'snowfall', 'high_vegetation_cover', 'low_vegetation_cover',
           'soil_water']
    frecuencias = ['D', 'M', 'Y']
    
    # Inicializar el diccionario para resultados PCA
    pca_results = []
    
    # Aplicar PCA para cada combinación de variable y frecuencia
    for var in variables:
        for frec in frecuencias:
            pca_df = apply_pca(df, var=var, frec=frec, var_threshold=var_threshold)
            pca_results.append(pca_df)
    df_final = pd.concat(pca_results, axis = 1).reset_index(drop = True)
    df_final['date'] = dates
    df_final['location_id'] = location_id
    df_final['quantity_hm3'] = quantity_hm3
    logging.info(f"Aanálisis de componentes principales (PCA) realizado con éxito")
    return df_final
    
# def gradient_boosting(df):
#     X = df.drop(['quantity_hm3'], axis=1)  
#     y = df['quantity_hm3']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
#     xgb_model = XGBRegressor(
#         n_estimators=50,
#         max_depth=20,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )
    
#     xgb_model.fit(X_train, y_train)
    
#     # Predicciones
#     y_train_pred = xgb_model.predict(X_train)
#     y_test_pred = xgb_model.predict(X_test)
    
#     train_r2 = r2_score(y_train, y_train_pred)
#     test_r2 = r2_score(y_test, y_test_pred)
    
#     print(train_r2)
#     print(test_r2)
#     #logging.info(f"Modelo gradient boosting entrenado con éxito")
#     return xgb_model
    
def randomForest(df):
    logging.info(f"Se inicia el entrenamiento del modelo de random forest")
    X = df.drop(['quantity_hm3'], axis=1)  # Excluir 'date' si no se usa explícitamente
    y = df['quantity_hm3']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo de Random Forest con warm_start=True para entrenamiento incremental
    rf = RandomForestRegressor(
        n_estimators=1,  # Inicializamos con 1 árbol
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        bootstrap=True,
        warm_start=True  # Permite agregar más árboles de forma incremental
    )
    
    # Entrenar árbol por árbol y mostrar progreso
    n_estimators = 100  # Cantidad de árboles deseada
    for i in tqdm(range(1, n_estimators + 1), desc="Entrenando Random Forest"):
        rf.set_params(n_estimators=i)  # Incrementamos el número de árboles
        rf.fit(X_train, y_train)  # Entrenamos con más árboles
    
    # Predicciones
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logging.info(f"Modelo random forest entrenado con éxito a un R^2 {train_r2} en train y un {test_r2} en test")
    
    return rf
    
def create_df_retardos_prediccion(df,retardosMax):
    '''
    Para validar modelo haciendo una predicción ante todos los pixels.
    Imprescindible columna pixel 
    df es el data frame de la extraccion incial
    componentes pasarlo con la funcion extract_n_components(df.columns) siendo df el data frame con el PCA pasado
    '''
    
    locations = df['location_id'].unique()
    
    df_preds_list = []
    for loc in locations:
        df_pixel = df[df['location_id'] == loc].copy()
        df_pixel.drop(['location_id'],axis = 1, inplace = True)
        fecha_inicial  = df_pixel['date'].max()
        fecha_final = fecha_inicial+ pd.Timedelta(days=retardosMax) 
        df_date_range = pd.DataFrame(pd.date_range(start=fecha_inicial, end=fecha_final, freq='D'), columns=['date'])
        df_pixel = pd.merge(df_date_range, df_pixel, on='date', how='outer').sort_values('date')
        df_pixel = calculate_df_retardos(df = df_pixel,retardosMax = retardosMax)
        df_pixel['location_id'] = loc
        df_preds_list.append(df_pixel)       
    df_preds = pd.concat(df_preds_list)
    df_preds = df_preds.reset_index(drop = True)
    #df_preds = df_preds.dropna(subset=['location_id'])
    # logging.info(f"Retardos de la predicción calculados con éxito")
    return df_preds


def process_pca_custom_components (df, variables, frecuencias, n_components_dict, imprimir=False):
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
    logging.info(f"Se procesa el PCA en base al entrenamiento...")
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
    logging.info(f"Se extraen los componentes principales usados en el entrenamiento")
    n_components_dict = {}

    for col in columns:
        parts = col.split('_PCA_')
        variable = parts[0]
        frequency, component = parts[1].split('_comp')
        key = f"{variable}_{frequency}"
        n_components_dict[key] = max(n_components_dict.get(key, 0), int(component))

    return n_components_dict
    
def create_df_PCA_prediccion(fecha_inicial,df_preds,df_pca ,retardosMax):
    logging.info(f"Análisis de componentes principales basados en los componentes usados en el entrenamiento")
    columnas_pca = [i for i in df_pca.columns if i not in  ['date', 'location_id', 'quantity_hm3']]
    componentes = extract_n_components(columnas_pca)
    location_id = df_preds['location_id']
    dates = df_preds['date']
    df_preds.set_index('date', inplace = True)
    variables = ['total_precipitation', 'skin_temperature', 'evaporation', 'runoff',
           'snowfall', 'high_vegetation_cover', 'low_vegetation_cover',
           'soil_water']
    frecuencias = ['D', 'M', 'Y']
    df_preds.drop('location_id', axis = 1, inplace = True)
    df_pca_preds = process_pca_custom_components(df = df_preds, variables = variables, frecuencias  = frecuencias, n_components_dict = componentes, imprimir=False)
    df_pca_preds.drop(['date'],axis = 1, inplace = True)
    df_pca_preds['date'] = dates
    df_pca_preds['location_id'] = location_id
    df_pca_preds = df_pca_preds[df_pca_preds['date'] >= fecha_inicial].set_index('date')
    logging.info(f"Data frame para la predicción creado con éxito")
    return df_pca_preds

def crearDataFrameTrain(retardosMax):
    df = extraer_datosBBDD()
    df_retard = calculoRetardos (df = df, retardosMax = retardosMax)
    df_pca = process_pca_for_variables(df = df_retard, var_threshold=80)
    return df, df_pca

def crearDataFramePred(df,df_pca, retardosMax):
    fecha_inicial = df['date'].max()
    df_preds = create_df_retardos_prediccion(df = df,retardosMax = retardosMax)
    df_pca_preds = create_df_PCA_prediccion(fecha_inicial = fecha_inicial,df_preds = df_preds, df_pca = df_pca ,retardosMax = retardosMax)
    return df_pca_preds

def comprobarActualizarBBDD(retardosMax):
    fecha_actual = datetime.now().date()
    ultimaFechaCoper = extraerUltimasFechasCopernicus()
    ultimaFechaRios = extraerUltimasFechasRios()
    if (fecha_actual - min(ultimaFechaCoper,ultimaFechaRios)).days > retardosMax:
        return True
    else:
        return False

def extract_Loc (pixel):
    conn = sqlite3.connect('aguaCHJucar.db')

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
            -- Usar COALESCE para manejar los valores NULL
            COALESCE(embalses_info.Embalse, 'No hay embalse') AS embalse,
            COALESCE(embalses_info.location_id, 'No hay embalse') AS location_id_embalse,
            COALESCE(rios_canales_info.EstacióndeAforo, 'No hay rio') AS estacion_aforo_rios_canales,
            COALESCE(rios_canales_info.location_id, 'No hay rio') AS location_id_rios,
            COALESCE(aemet_info.nombre, 'No hay aemet') AS nombre_aemet,
            COALESCE(aemet_info.location_id, 'No hay aemet') AS location_id_aemet
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

    df_loc = pd.read_sql_query(query, conn)
    # Ejecutar la consulta
    cursor.execute(query)
    conn.close()
    return df_loc


def crear_tablaInfos(prediccion):
    conn = sqlite3.connect('aguaCHJucar.db')
    
    cursor = conn.cursor()
    query = f'''
        SELECT
            pc.location_id_copernicus, pc.location_id_rios_canales, info.*
            FROM df_pixeles_cercanos pc
            JOIN locations_id l on pc.location_id_rios_canales = l.location_id
            JOIN df_rios_canales_info info on info.location_id = l.location_id ;
    '''
    
    df_infoRios = pd.read_sql_query(query, conn)
    # Ejecutar la consulta
    cursor.execute(query)
    conn.close()
    
    df_infoRios = df_infoRios[['location_id_copernicus', 'Municipio','SistemadeExplotación']].drop_duplicates().reset_index(drop  =True).rename(columns = {'location_id_copernicus' : 'Pixel'})
    
    info_sistemasExplot = df_infoRios.groupby("Pixel").agg({
        "Municipio": lambda x: ', '.join(x),
        "SistemadeExplotación": lambda x: ', '.join(sorted(set(val for val in x if val is not None)))  # Eliminar duplicados y ordenar
    }).reset_index()
    ########################
    df_loc_list = []
    for pixel in prediccion['location_id'].unique():
        df_loc_i = extract_Loc(pixel)
        df_loc_i['Pixel'] = pixel
        df_loc_i = df_loc_i.rename(columns = {'estacion_aforo_rios_canales' : 'Río','latitude_copernicus':'Latitud', 'longitude_copernicus' : 'Longitud'})
        df_loc_list.append(df_loc_i)
    df_locs = pd.concat(df_loc_list)
    
    df_rios = df_locs[['Latitud','Longitud','Pixel', 'Río']].drop_duplicates()
    result_rios = df_rios.groupby(['Pixel', 'Latitud', 'Longitud'])['Río'] \
               .apply(lambda x: ', '.join(x)).reset_index()
    
    # Renombrar la columna de salida para mayor claridad
    result_rios.rename(columns={'Río': 'Ríos'}, inplace=True)
    df_embalses = df_locs[['Pixel','embalse']]
    # Filtrar los valores "No hay embalse"
    result_embalses = df_embalses[df_embalses['embalse'] != "No hay embalse"]
    # Agrupar por 'Pixel', eliminar duplicados en 'embalse' antes de unir
    result_embalses = result_embalses.groupby('Pixel')['embalse'] \
        .apply(lambda x: ', '.join(x.drop_duplicates())).reset_index()
    
    # Renombrar la columna para mayor claridad
    result_embalses.rename(columns={'embalse': 'Embalses'}, inplace=True)
    
    df_rios = df_locs[['Pixel', 'Río']].drop_duplicates()
    contar_rios = df_rios.groupby('Pixel').size().reset_index(name='Cantidad de ríos')
    
    # Agregar píxeles con 0 embalses
    all_pixels = df_rios['Pixel'].unique()
    contar_rios = contar_rios.set_index('Pixel').reindex(all_pixels, fill_value=0).reset_index()
    
    df_embalses = df_locs[['Pixel', 'embalse']].drop_duplicates()
    contar_embalses = df_embalses[df_embalses['embalse'] != "No hay embalse"].groupby('Pixel').size().reset_index(name='Cantidad de emblases')
    
    # Agregar píxeles con 0 embalses
    all_pixels = df_embalses['Pixel'].unique()
    
    df_pixelesInfo = pd.merge(result_rios,contar_rios, on ='Pixel', how = 'left')
    df_pixelesInfo = pd.merge(df_pixelesInfo, result_embalses, on = 'Pixel', how = 'left')
    df_pixelesInfo = pd.merge(df_pixelesInfo, contar_embalses, on = 'Pixel', how = 'left')
    df_pixelesInfo = pd.merge(df_pixelesInfo, info_sistemasExplot, on = 'Pixel', how = 'left').fillna(0).rename(columns = {'Municipio' : 'Municipio cercano'})
    return df_pixelesInfo

def descargarHistoricoRios():
    conn = sqlite3.connect('aguaCHJucar.db')
    
    cursor = conn.cursor()
    ids = tuple(prediccion['location_id'].unique())
    query = f'''
        SELECT 
        d.date,
        pc.location_id_copernicus AS location_id,
        rios.quantity_hm3
        FROM df_rios_canales rios 
        JOIN df_date d ON d.date_id = rios.date_id 
        JOIN df_pixeles_cercanos pc ON pc.location_id_rios_canales = rios.location_id
        WHERE pc.location_id_copernicus IN {ids};
    '''
    # Ejecutar la consulta
    cursor.execute(query)
    df_rios = pd.read_sql_query(query, conn)
    df_rios['date'] = pd.to_datetime(df_rios['date'])
    df_rios = df_rios.sort_values(['date','location_id']).reset_index(drop = True)
    return df_rios


def flujoTrabajo(retardosMax):
    tiempo_inicial = time.time()
    logging.info(f"Se procede a calcular la predicción para {retardosMax} días. \n Para información adicional sobre cómo se realiza ver: https://github.com/delatorre96/Water-Prediction")
    if comprobarActualizarBBDD(retardosMax = retardosMax):
        logging.info(f"\nFASE 0: Actualización de la base de datos\nComo los días a predecir son mayores que la diferencia entre lafecha actual y la última fecha de la base dedatos, se procede a actualizar la base de datos.")
        actualizacionBBDD()
    else:
        pass
    logging.info(f"\nFASE 1: Extracción y preprocesado de los datos existentes en la actual base de datos")
    df, df_pca = crearDataFrameTrain(retardosMax = retardosMax)
    df_pca.set_index(['date', 'location_id'], inplace = True)
    logging.info(f"\nFASE 2: Entrenamiento del modelo de predicción")
    rtf = randomForest (df = df_pca)
    # xgb_model = gradient_boosting(df = df_pca)
    logging.info(f"\nFASE 3: Cálculo de la predicción")
    df_pca_preds = crearDataFramePred(df = df,df_pca = df_pca, retardosMax = retardosMax)
    df_pca_preds.reset_index(inplace = True)
    df_pca_preds.set_index(['date', 'location_id'], inplace = True)
    prediccion = rtf.predict(df_pca_preds)
    tiempo_final = time.time()
    duracion = (tiempo_final - tiempo_inicial)/60
    logging.info(f"Predicción para {retardosMax} días realizada em {duracion} minutos")
    logging.info(f"\nFASE 4: Presentación de resultados y salida de excel y csv. ")    
    df_prediccionFinal = df_pca_preds.reset_index()[['date', 'location_id']]
    df_prediccionFinal['prediccion'] = prediccion
    fecha_actual = datetime.now().date()
    prediccion = prediccion.rename(columns = {'prediccion':'quantity_hm3'}).sort_values(['location_id','date']).reset_index(drop = True)
    prediccionCHJ_promedio = prediccion.groupby('date').mean().reset_index()[['date','quantity_hm3']]
    prediccionCHJ_agregado= prediccion.groupby('date').sum().reset_index()[['date','quantity_hm3']]
    
    df_rios_historico = descargarHistoricoRios()
    df_rios_historico = df_rios_historico[df_rios_historico['date'].dt.year >= 1960].reset_index(drop = True)
    historicoCHJ_promedio = df_rios_historico.groupby('date').mean().reset_index()[['date','quantity_hm3']]
    historicoCHJ_agregado = df_rios_historico.groupby('date').sum().reset_index()[['date','quantity_hm3']]
    
    Info = crear_tablaInfos(prediccion)
    dataframes = {
        'Predicción CHJ caudal promedio': prediccionCHJ_promedio,
        'Predicción CHJ caudal agregado': prediccionCHJ_agregado,
        'Histórico CHJ caudal promedio': historicoCHJ_promedio,
        'Histórico CHJ caudal agregado': historicoCHJ_agregado,
        'Prediccion por pixel' : prediccion,
        'Info': df_pixelesInfo}
    output_path = f"Resultados_{fecha_actual}.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    df_rios_historico.to_csv(f'datos_rios_historico_{fecha_actual}.csv')
    logging.info(f"""
    Excel escrito correctamente como {output_path}. 
    El contenido de este excel tiene 6 páginas: 
    -Predicción CHJ caudal promedio
    -Predicción CHJ caudal promedio
    -Predicción CHJ caudal agregado
    -Histórico CHJ caudal promedio
    -Histórico CHJ caudal agregado
    -Info: Información hidrográfica adicional relacionada con cada área geográfica en donde aparecen los ríos, embalses y sistemas de explotación vinculados a cada pixel junto con sus coordenadas.
     Adicionalmente se escribe un csv con el histórico del caudal de ríos utilizado para la predicción. Esto es un csv en donde se incluyen los datos de cada observación temporal desde 1960 para la cantidad de agua total que hay en cada pixel.""")
  
    