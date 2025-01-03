'''
AGUAS SUBTERRANEAS
https://aps.chj.es/down/html/descargas.html

* Unidades de gestión
    * PHJ 2022/2027
        * Masas de agua subterránea Datos Cartografía WMS


* Medidas de cantidad
    * Aguas subterráneas
        * Piezómetros Datos Cartografía WMS
'''


### Piezómetros
df_aguas_sub_id = pd.read_excel('Cuenca hidrográfica Juca SI/F71_agua sub piezometrico/F71_M10_Todas.xlsx')
df_aguas_sub1 = pd.read_excel('Cuenca hidrográfica Juca SI/F71_agua sub piezometrico/F71_D7_Medida en campo_1.xlsx')
df_aguas_sub2 = pd.read_excel('Cuenca hidrográfica Juca SI/F71_agua sub piezometrico/F71_D7_Medida en campo_2.xlsx')
## Unidades de gestión
df_aguas_ud_gestion = pd.read_excel('Cuenca hidrográfica Juca SI/F2626_PHJ_2022_2027_Masas_de_agua_subterranea/F2626_M0_PHJ_2022_2027_Masas_de_agua_subterranea.xlsx')

##Calculamos masas de agua
df_pozos_id = df_aguas_sub_id[['Cód. Punto', 'Nombre','Municipio','Masa Subterránea PHJ22 Hidrog.','UTM ETRS89 X','UTM ETRS89 Y']]
df_pozos_id = df_aguas_sub_id.rename(columns = {'Masa Subterránea PHJ22 Hidrog.' : 'Masa Subterránea'})
df_acuiferos = df_aguas_ud_gestion[['Cód. Masa','Masa Subterránea','Sup. (km²)','Tipo']]
df_aguas_subterraneas = pd.merge(df_pozos_id, df_acuiferos, on = 'Masa Subterránea', how = 'inner')
df_aguas_subterraneas_locs = df_aguas_subterraneas[['Cód. Masa','Masa Subterránea',	'UTM ETRS89 X',	'UTM ETRS89 Y',	'Sup. (km²)']]
df_aguas_subterraneas_locs = df_aguas_subterraneas_locs.groupby(['Cód. Masa','Masa Subterránea']).mean().reset_index()
#transformar coordenadas
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)

# Aplicar la transformación a latitud y longitud
df_aguas_subterraneas_locs['longitude'], df_aguas_subterraneas_locs['latitude'] = transformer.transform(df_aguas_subterraneas_locs['UTM ETRS89 X'].values, df_aguas_subterraneas_locs['UTM ETRS89 Y'].values)
df_aguas_subterraneas_locs = df_aguas_subterraneas_locs.drop(['UTM ETRS89 X',	'UTM ETRS89 Y'], axis = 1)
df_aguas_subterraneas_locs.head()

#enlazamos el cod punto que referencia el pozo con el id de la masa subterranea para enlazarlo después con las series
df_masaSub_idPozo = pd.merge(df_aguas_subterraneas_locs, df_pozos_id[['Cód. Punto','Masa Subterránea']] , on = ['Masa Subterránea'])
series_nivel_aguaSub = pd.concat([df_aguas_sub1,df_aguas_sub2])[['Cód. Punto','Fecha','Med. Valid. Estática (m)']]
series_nivel_aguaSub['Fecha'] = series_nivel_aguaSub['Fecha'].dt.date
series_nivel_aguaSub['Fecha'] = pd.to_datetime(series_nivel_aguaSub['Fecha'])
df_acuiferos_series = pd.merge(series_nivel_aguaSub, df_masaSub_idPozo[['Cód. Masa','Cód. Punto','Sup. (km²)']],on = 'Cód. Punto').drop('Cód. Punto', axis = 1).dropna().groupby(['Cód. Masa','Fecha']).mean().reset_index()
#incluimos las coordenadas al data set
df_acuiferos_series = pd.merge(df_aguas_subterraneas_locs[['Cód. Masa', 'longitude', 'latitude']],df_acuiferos_series, on = 'Cód. Masa', how = 'right')
df_acuiferos_series['cantidadAgua'] = df_acuiferos_series['Med. Valid. Estática (m)'] * df_acuiferos_series['Sup. (km²)']/1000000

df_aguas_subterraneas_locs.to_csv('df_acuiferos_series_id.csv',index = False)
df_acuiferos_series.to_csv('df_acuiferos_series.csv',index = False)

'''
ALBUFERA
'''
df_albufera = pd.read_excel('Cuenca hidrográfica Juca SI/F56_Estaciones_de_aforo_de_L_Albufera/F56_D7_Serie día.xlsx')
df_albufera = df_albufera.rename(columns = {'INDI_CHJ' : 'id_station','Volumen (hm³)':'quantity_hm3','Fecha' : 'date'})
df_albufera = df_albufera[['id_station','date','quantity_hm3']]
df_albufera = df_albufera.dropna()
df_albufera['date'] = pd.to_datetime(df_albufera['date'], format='%d/%m/%Y')

df_albufera_id = pd.read_csv('Cuenca hidrográfica Juca SI/F56_Estaciones_de_aforo_de_L_Albufera/F56_M0_Estaciones_de_aforo_de_L_Albufera.csv', index_col = 0)
df_albufera_id = df_albufera_id.rename(columns = {'Cód. Estación' : 'id_station','Tipo': 'type','Municipio': 'place'})
df_albufera_id = df_albufera_id[['id_station', 'type','place','longitude','latitude']]
#Crear columna pixels únicos
unique_pixels = df_albufera_id[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
unique_pixels['pixel'] = range(len(unique_pixels))
df_albufera_id = pd.merge(df_albufera_id, unique_pixels, on=['latitude', 'longitude'], how='left')
df_albufera_id.to_csv('df_albufera_id.csv',index = False)
df_albufera.to_csv('df_albufera.csv',index = False)