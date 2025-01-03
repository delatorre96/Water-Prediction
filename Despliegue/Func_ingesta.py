import sqlite3
import pandas as pd
import xarray as xr
import cdsapi
import os
import zipfile
import sqlite3
from datetime import datetime, timedelta
from calendar import monthrange
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


def downloadMesCopernicus (days,year, month):
    dataset = "reanalysis-era5-single-levels"
    request = {
    'product_type': ['ensemble_mean'],
    'variable': ['total_precipitation', 'skin_temperature', 'evaporation', 'runoff', 'snowfall', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'high_vegetation_cover', 'low_vegetation_cover', 'type_of_high_vegetation', 'type_of_low_vegetation'],
    'year': [str(year)],
    'month': [str(month)],
    'day': days,
    'time': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
    'data_format': 'netcdf',
    'download_format': 'unarchived',
    'area': [40.5425, -2.255, 38.1739, 0.5665] ##Cuenca hidrográfica del Jucar +/-
    }
    client = cdsapi.Client()
    file_name = client.retrieve(dataset, request).download()

    return file_name
    
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
    fecha2 = fecha_actual = datetime.now().date()
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
        dataFrames.append(df)
    df = pd.concat(dataFrames)
    return df

