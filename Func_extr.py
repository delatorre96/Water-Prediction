import sqlite3
import pandas as pd
import folium
import math
import matplotlib.pyplot as plt
import numpy as np


def extract_Loc (pixel):
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

def extract_infos(pixel):
    conn = sqlite3.connect('BBDD/aguaCHJucar.db')

    cursor = conn.cursor()
    query = f'''
        SELECT 
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

    df_info = pd.read_sql_query(query, conn)
    # Ejecutar la consulta
    cursor.execute(query)
    conn.close()
    
    return df_info


def dataExtract(pixel, x = True):
    conn = sqlite3.connect('BBDD/aguaCHJucar.db')
    
    cursor = conn.cursor()
    
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
    df_rios['date'] = pd.to_datetime(df_rios['date'])
    
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
    
    df_embalses['date'] = pd.to_datetime(df_embalses['date'])
    
    if x:
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
        
        df_aemet['date'] = pd.to_datetime(df_aemet['date'])
        df_aemet = df_aemet.groupby('date').mean().reset_index()
        df_c['date'] = pd.to_datetime(df_c['date'])
        df_c['soil_water'] = df_c['soil_water_l1'] + df_c['soil_water_l2'] + df_c['soil_water_l3'] + df_c['soil_water_l4']
        df_c = df_c.drop(['soil_water_l1', 'soil_water_l2', 'soil_water_l3','soil_water_l4'], axis = 1)
        df_x = pd.merge(df_c,df_aemet, on='date', how='inner',suffixes=('_copernicus', '_aemet'))
        df_embalses['date'] = pd.to_datetime(df_embalses['date'])
        df_rios['date'] = pd.to_datetime(df_rios['date'])
        conn.close()
        return  df_embalses, df_rios, df_x
    else:
        conn.close()
        return  df_embalses, df_rios

def create_df(pixel):
    df_embalses, df_rios, df_x = dataExtract(pixel, x = True)
    df_embalses.drop('location_id',axis = 1,inplace = True)
    df_rios.drop('location_id',axis = 1,inplace = True)
    df_x.drop('location_id',axis = 1,inplace = True)
    if df_embalses.empty:
        df_rios = df_rios.groupby('date').mean().reset_index()
        df = pd.merge(df_x, df_rios, on = 'date', how = 'inner')
    else:
        df_rios = df_rios.groupby('date').mean().reset_index()
        df_embalses = df_embalses.groupby('date').mean().reset_index()
        df = pd.merge(df_x, df_rios, on = 'date', how = 'inner')
        df = pd.merge(df, df_embalses, on = 'date', how = 'inner')
    df.set_index('date', inplace =True)
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

def pixels_colindantes(pixel):
    conn = sqlite3.connect('BBDD/aguaCHJucar.db')

    query = f'''
        SELECT 
            p.location_id_copernicus, 
            l1.latitude AS latitude_copernicus, 
            l1.longitude AS longitude_copernicus
        FROM df_pixeles_cercanos p
        LEFT JOIN locations_id l1 ON p.location_id_copernicus = l1.location_id
    '''

    # Obtener el DataFrame con las coordenadas
    df_loc = pd.read_sql_query(query, conn)
    conn.close()

    # Eliminar filas duplicadas en las columnas relevantes
    df_loc_unique = df_loc[['location_id_copernicus', 'latitude_copernicus', 'longitude_copernicus']].drop_duplicates()

    # Obtener las coordenadas del location_id_copernicus objetivo
    target_location = df_loc_unique[df_loc_unique['location_id_copernicus'] == pixel]
    target_lat = target_location['latitude_copernicus'].values[0]
    target_lon = target_location['longitude_copernicus'].values[0]

    # Calcular la distancia a todos los demás location_id_copernicus
    df_loc_unique[f'distance_to_{pixel}'] = df_loc_unique.apply(
        lambda row: haversine(
            (target_lat, target_lon), 
            (row['latitude_copernicus'], row['longitude_copernicus'])
        ),
        axis=1
    )

    # Filtrar los location_id_copernicus colindantes
    colindantes = df_loc_unique[df_loc_unique[f'distance_to_{pixel}'] < 80]


    return colindantes


def extraccion_colindantes(colindantes):
    
    locations_colindantes  = colindantes['location_id_copernicus'].to_list()
    df_total = pd.DataFrame()
    for i in locations_colindantes:
        df_c, df_embalses, df_rios, df_aemet, df_x = dataExtract(i)
        df_embalses = df_embalses.groupby('date').mean()[['quantity_hm3']]
        df_embalses = df_embalses.rename(columns = {'quantity_hm3' : 'quantity_hm3_embalses'})
        df_embalses.reset_index(inplace = True)
        df_rios = df_rios.groupby('date').mean()[['quantity_hm3']]
        df_rios = df_rios.rename(columns = {'quantity_hm3' : 'quantity_hm3_rios'})
        df_rios.reset_index(inplace = True)
        df_x = df_x.groupby('date').mean()
        df_x.reset_index(inplace = True)
        df_merge = pd.merge(df_embalses, df_rios, on ='date', how = 'outer')
        df_merge = df_merge.fillna(0)
        df_merge['quantity_hm3_total'] = df_merge['quantity_hm3_embalses'] + df_merge['quantity_hm3_rios']
        df_merge = pd.merge(df_merge, df_x, on = 'date', how = 'inner')
        df_merge['pixel_colin'] = i
        df_total = pd.concat([df_merge,df_total])
    return df_total

