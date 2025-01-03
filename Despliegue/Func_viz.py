import pandas as pd
import folium
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from Func_extr import extract_Loc, pixels_colindantes, extract_infos, dataExtract
from math import cos, radians
from  statsmodels.tsa.seasonal import seasonal_decompose
from numpy import round

def mapRep_coper(pixel):
    """
    Genera un mapa interactivo que muestra información geoespacial relevante para un píxel objetivo.

    Parámetros:
    -----------
    pixel : int
        Identificador del píxel objetivo.

    Funcionalidad:
    --------------
    - Centra el mapa en las coordenadas del píxel objetivo.
    - Dibuja un rectángulo que representa el área de Copernicus asociada al píxel.
    - Añade marcadores para estaciones de Aemet, ríos/canales y embalses, si están disponibles.

    Salidas:
    --------
    Devuelve un objeto `folium.Map` con los elementos geoespaciales representados.
    """
    df_loc = extract_Loc (pixel)
    lat = round(df_loc['latitude_copernicus'].to_list()[0], 1)
    lon = round(df_loc['longitude_copernicus'].to_list()[0], 1)
    # Coordenadas base para centrar el mapa
    
    m = folium.Map(location=[lat, lon], zoom_start=10)

    
    # Añadir el cuadrado de la zona de Copernicus (representando un área más grande)
    for index, row in df_loc.iterrows():
        # Longitud y latitud del centro de Copernicus
        lat_copernicus = row['latitude_copernicus']
        lon_copernicus = row['longitude_copernicus']
    
        # Definir un lado del cuadrado basado en 0.5 grados
        size_lat = 0.5  # 0.5 grados de latitud
        size_lon = 0.5 * 111.32 * cos(radians(lat_copernicus)) / 111.32  # Ajustado por la latitud (en grados)
    
        additional_lon = 5 / (111.32 * cos(radians(lat_copernicus))) 
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
        if df_loc['nombre_aemet'][0] != 'No hay aemet':
            folium.Marker(
                location=[row['latitude_aemet'], row['longitude_aemet']],
                popup=f"Aemet: {row['nombre_aemet']}; estacion {row['location_id_aemet']}",
                icon=folium.Icon(color='red')
            ).add_to(m)
    
        # Añadir marcador para Ríos Canales
        if df_loc['estacion_aforo_rios_canales'][0] != 'No hay rio':
            folium.Marker(
                location=[row['latitude_rios_canales'], row['longitude_rios_canales']],
                popup=f"Ríos Canales: {row['estacion_aforo_rios_canales']}; estacion {row['location_id_rios']}",
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


def mapa_pixels_colindantes(pixel):
    """
    Crea un mapa interactivo con marcadores para un píxel objetivo y sus píxeles colindantes.

    Parámetros:
    -----------
    pixel : int
        Identificador del píxel objetivo.

    Funcionalidad:
    --------------
    - Ubica el píxel objetivo en un mapa con un marcador rojo.
    - Añade marcadores azules para los píxeles colindantes, mostrando su distancia al píxel objetivo.
    - Traza líneas verdes entre el píxel objetivo y cada píxel colindante.

    Salidas:
    --------
    Devuelve un objeto de tipo `folium.Map` con el mapa interactivo.
    """
    colindantes = pixels_colindantes(pixel)
    m = folium.Map(location=[target_lat, target_lon], zoom_start=12)
    
    # Añadir el marcador para el pixel objetivo
    folium.Marker(
        location=[target_lat, target_lon],
        popup=f'location_id_copernicus == {pixel}',
        icon=folium.Icon(color='red')
    ).add_to(m)
    
    # Añadir marcadores y líneas para los puntos colindantes
    for _, row in colindantes.iterrows():
        folium.Marker(
            location=[row['latitude_copernicus'], row['longitude_copernicus']],
            popup=f'location_id_copernicus == {row["location_id_copernicus"]}, Distancia: {row[f"distance_to_{pixel}"]:.2f} km',
            icon=folium.Icon(color='blue')
        ).add_to(m)
        
        # Línea entre el pixel objetivo y los puntos colindantes
        folium.PolyLine(
            locations=[
                (target_lat, target_lon), 
                (row['latitude_copernicus'], row['longitude_copernicus'])
            ],
            color='green',
            weight=2.5,
            opacity=1
        ).add_to(m)

        return m

def hist_evAgua (pixel, soloAgregados = False):
    """
    Genera gráficos de las series temporales de caudales y volúmenes de agua
    para ríos y embalses asociados a un píxel, incluyendo sus tendencias.

    Parámetros:
    -----------
    pixel : int
        Identificador del píxel para el que se extraerán los datos.
    soloAgregados : bool, opcional
        Si es True, solo genera gráficos agregados (total por ríos y embalses).
        Por defecto es False, incluyendo gráficos para cada río y embalse.

    Funcionalidad:
    --------------
    - Extrae datos de caudales y volúmenes de agua.
    - Grafica las series temporales totales y sus tendencias para ríos y embalses.
    - Si `soloAgregados` es False, también grafica datos individuales para
      cada río y embalse.

    Salidas:
    --------
    No retorna valores. Genera múltiples gráficos de las series temporales.

    """
    df_info = extract_infos(pixel)
    df_embalses, df_rios = dataExtract(pixel, x = False)
    if df_info['location_id_rios'][0] != 'No hay rio':
        df_rios_total = df_rios.groupby('date').mean()['quantity_hm3']
        tendencia =  seasonal_decompose(df_rios_total.dropna(), model='additive', period=200).trend
        plt.figure(figsize=(12, 8))
        plt.plot(df_rios_total)
        plt.plot(tendencia)
        plt.title('TOTAL RÍOS')
        plt.show()
        df_rios = pd.merge(df_rios, df_info, left_on = 'location_id', right_on = 'location_id_rios', how = 'left')  
        if soloAgregados == False:
            for rio in df_rios['location_id'].unique():
                serie_rio = df_rios[df_rios['location_id'] == rio][['date','quantity_hm3']].groupby('date').mean()
                tendencia =  seasonal_decompose(serie_rio['quantity_hm3'].dropna(), model='additive', period=200).trend
                nombre_rio = df_rios[df_rios['location_id'] == rio]['estacion_aforo_rios_canales'].to_list()[0]
                plt.figure(figsize=(12, 8))
                plt.plot(serie_rio)
                plt.plot(tendencia)
                plt.title(f'{nombre_rio}')        
                plt.show()
    if df_info['location_id_embalse'][0] != 'No hay embalse':
        df_embalses = pd.merge(df_embalses, df_info, left_on = 'location_id', right_on = 'location_id_embalse', how = 'left')
        df_embalses_total = df_embalses.groupby('date').mean()['quantity_hm3']
        tendencia =  seasonal_decompose(df_embalses_total['quantity_hm3'].dropna(), model='additive', period=200).trend
        plt.figure(figsize=(12, 8))
        plt.plot(df_embalses_total)
        plt.plot(tendencia)
        plt.title('TOTAL EMBALSES')
        plt.show()
        if soloAgregados == False:
            for embalse in df_embalses['location_id'].unique():
                serie_embalse = df_embalses[df_embalses['location_id'] == embalse][['date','quantity_hm3']].groupby('date').mean()
                tendencia =  seasonal_decompose(serie_embalse.dropna(), model='additive', period=200).trend
                nombre_embalse = df_embalses[df_embalses['location_id_embalse'] == embalse]['embalse'].to_list()[0]
                plt.figure(figsize=(12, 8))
                plt.plot(serie_embalse)
                plt.plot(tendencia)
                plt.title(f'{nombre_embalse}')
                plt.show()

def autocorr_serie (serie, col, lags_list = [200,30,10]):
    """
    Calcula y grafica la función de autocorrelación (ACF) de una serie temporal
    para frecuencias diarias, mensuales y anuales.

    Parámetros:
    -----------
    serie : pd.DataFrame
        DataFrame que contiene la serie temporal. Debe incluir una columna 'date'
        con las fechas y otra columna con los valores de la serie.
    col : str
        Nombre de la columna de la serie temporal que se analizará.
    lags_list : list, opcional
        Lista de tres enteros que especifican el número de retardos (lags) para
        cada frecuencia temporal:
        - lags_list[0]: Número de retardos para la autocorrelación diaria.
        - lags_list[1]: Número de retardos para la autocorrelación mensual.
        - lags_list[2]: Número de retardos para la autocorrelación anual.
        Valor por defecto: [200, 30, 10].

    Funcionalidad:
    --------------
    1. Convierte la columna 'date' a formato datetime y la establece como índice.
    2. Calcula y grafica la autocorrelación diaria utilizando la columna `col`
       con el número de retardos especificado en `lags_list[0]`.
    3. Resamplea la serie a frecuencia mensual (suma por defecto) y grafica la
       autocorrelación mensual utilizando `lags_list[1]`.
    4. Resamplea la serie a frecuencia anual (suma por defecto) y grafica la
       autocorrelación anual utilizando `lags_list[2]`.

    Salidas:
    --------
    No retorna valores. Genera tres gráficos de autocorrelación (diaria,
    mensual y anual).
    """
    # Asegúrate de que la columna 'date' es un datetime y está como índice
    serie['date'] = pd.to_datetime(serie['date'])
    serie = serie.set_index('date')
    
    # Autocorrelación diaria
    plot_acf(serie[col], lags=lags_list[0])
    plt.title('Función de Autocorrelación (ACF) de la serie diaria')
    plt.xlabel('Retardo (días)')
    plt.ylabel('Autocorrelación')
    plt.show()
    
    # Autocorrelación mensual
    monthly_series = serie.resample('M').sum()  # Cambia a .sum() si quieres acumulados
    plot_acf(monthly_series[col], lags=lags_list[1])
    plt.title('Función de Autocorrelación (ACF) de la serie mensual')
    plt.xlabel('Retardo (meses)')
    plt.ylabel('Autocorrelación')
    plt.show()
    
    # Autocorrelación anual
    annualy_series = serie.resample('Y').sum()  # Cambia a .sum() si quieres acumulados
    plot_acf(annualy_series[col], lags=lags_list[2])
    plt.title('Función de Autocorrelación (ACF) de la serie anual')
    plt.xlabel('Retardo (años)')
    plt.ylabel('Autocorrelación')
    plt.show()

    

