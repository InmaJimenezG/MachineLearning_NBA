## Librerias
from catboost import CatBoostRegressor, Pool

import json

from math import pi
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
# disable chained assignments
pd.options.mode.chained_assignment = None 
import pickle

import requests

import scipy.stats as stats
import seaborn as sns
sns.set(color_codes=True)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportions_ztest

import utils.functions


## Carga de datos

page = 0 
season = 2022
response = requests.get(f"https://es.global.nba.com/stats2/league/playerstats.json?conference=All&country=All&individual=All&locale=es&pageIndex={page}&position=All&qualified=false&season={season}&seasonType=2&split=All+Team&statType=points&team=All&total=perGame")
json_response = response.json()
data_players = []

page = 0
season = 2022
response = requests.get(f"https://es.global.nba.com/stats2/league/playerstats.json?conference=All&country=All&individual=All&locale=es&pageIndex={page}&position=All&qualified=false&season={season}&seasonType=2&split=All+Team&statType=points&team=All&total=perGame")
json_response = response.json()
data_players += [json_response['payload']['season']]
data_players += json_response["payload"]["players"]

dataplayers_2011 = utils.functions.NBA_players_data(2011)
dataplayers_2012 = utils.functions.NBA_players_data(2012)
dataplayers_2013 = utils.functions.NBA_players_data(2013)
dataplayers_2014 = utils.functions.NBA_players_data(2014)
dataplayers_2015 = utils.functions.NBA_players_data(2015)
dataplayers_2016 = utils.functions.NBA_players_data(2016)
dataplayers_2017 = utils.functions.NBA_players_data(2017)
dataplayers_2018 = utils.functions.NBA_players_data(2018)
dataplayers_2019 = utils.functions.NBA_players_data(2019)
dataplayers_2020 = utils.functions.NBA_players_data(2020)
dataplayers_2021 = utils.functions.NBA_players_data(2021)
dataplayers_2022 = utils.functions.NBA_players_data(2022)

dataseason_2011 = utils.functions.season_data(2011)
dataseason_2012 = utils.functions.season_data(2012)
dataseason_2013 = utils.functions.season_data(2013)
dataseason_2014 = utils.functions.season_data(2014)
dataseason_2015 = utils.functions.season_data(2015)
dataseason_2016 = utils.functions.season_data(2016)
dataseason_2017 = utils.functions.season_data(2017)
dataseason_2018 = utils.functions.season_data(2018)
dataseason_2019 = utils.functions.season_data(2019)
dataseason_2020 = utils.functions.season_data(2020)
dataseason_2021 = utils.functions.season_data(2021)
dataseason_2022 = utils.functions.season_data(2022)

df_players_2011 = utils.functions.df_data_players(dataplayers_2011)
df_players_2012 = utils.functions.df_data_players(dataplayers_2012)
df_players_2013 = utils.functions.df_data_players(dataplayers_2013)
df_players_2014 = utils.functions.df_data_players(dataplayers_2014)
df_players_2015 = utils.functions.df_data_players(dataplayers_2015)
df_players_2016 = utils.functions.df_data_players(dataplayers_2016)
df_players_2017 = utils.functions.df_data_players(dataplayers_2017)
df_players_2018 = utils.functions.df_data_players(dataplayers_2018)
df_players_2019 = utils.functions.df_data_players(dataplayers_2019)
df_players_2020 = utils.functions.df_data_players(dataplayers_2020)
df_players_2021 = utils.functions.df_data_players(dataplayers_2021)
df_players_2022 = utils.functions.df_data_players(dataplayers_2022)

df_season_2011 = utils.functions.df_data_season(dataseason_2011)
df_season_2012 = utils.functions.df_data_season(dataseason_2012)
df_season_2013 = utils.functions.df_data_season(dataseason_2013)
df_season_2014 = utils.functions.df_data_season(dataseason_2014)
df_season_2015 = utils.functions.df_data_season(dataseason_2015)
df_season_2016 = utils.functions.df_data_season(dataseason_2016)
df_season_2017 = utils.functions.df_data_season(dataseason_2017)
df_season_2018 = utils.functions.df_data_season(dataseason_2018)
df_season_2019 = utils.functions.df_data_season(dataseason_2019)
df_season_2020 = utils.functions.df_data_season(dataseason_2020)
df_season_2021 = utils.functions.df_data_season(dataseason_2021)
df_season_2022 = utils.functions.df_data_season(dataseason_2022)

df_2011 = utils.functions.df_total(df_players_2011, df_season_2011)
df_2012 = utils.functions.df_total(df_players_2012, df_season_2012)
df_2013 = utils.functions.df_total(df_players_2013, df_season_2013)
df_2014 = utils.functions.df_total(df_players_2014, df_season_2014)
df_2015 = utils.functions.df_total(df_players_2015, df_season_2015)
df_2016 = utils.functions.df_total(df_players_2016, df_season_2016)
df_2017 = utils.functions.df_total(df_players_2017, df_season_2017)
df_2018 = utils.functions.df_total(df_players_2018, df_season_2018)
df_2019 = utils.functions.df_total(df_players_2019, df_season_2019)
df_2020 = utils.functions.df_total(df_players_2020, df_season_2020)
df_2021 = utils.functions.df_total(df_players_2021, df_season_2021)
df_2022 = utils.functions.df_total(df_players_2022, df_season_2022)

df_2021['temporada'] = '2021-2022'
df_2021['anio'] = int(2021)

df_nba = pd.concat([df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, 
                    df_2017, df_2018, df_2019, df_2020, df_2021, df_2022], axis = 0)

missing = df_nba.loc[(df_nba['eficiencia_tiro_pp']).isnull() | (df_nba['eficiencia_tiro_total']).isnull() | (df_nba['faltas_tecnicas_total']).isnull()]
df_nba = df_nba.loc[(df_nba.nombre_jugador != 'Michaela Pavlickova')]

df_nba.to_csv('/home/inma/Escritorio/The Bridge/Data_Science_Curso/Clase/Repository_Inma/ML/ML_NBA_Inma Jimenez/scr/data/raw_files/data_nba.csv', index = False)



## Procesado de datos
data_nba = pd.read_csv('/home/inma/Escritorio/The Bridge/Data_Science_Curso/Clase/Repository_Inma/ML/ML_NBA_Inma Jimenez/scr/data/raw_files/data_nba.csv')

data_nba.loc[data_nba['nombre_jugador'] == 'T.J. McConnell', 'posicion'] = 'B'
data_nba.loc[data_nba['nombre_jugador'] == 'Duje Dukan', 'posicion'] = 'AP'
data_nba.loc[data_nba['nombre_jugador'] == 'Boban Marjanovic', 'posicion'] = 'P'
data_nba.loc[data_nba['nombre_jugador'] == 'Marcelo Huertas', 'posicion'] = 'B'
data_nba.loc[data_nba['nombre_jugador'] == 'Briante Weber', 'posicion'] = 'B'
data_nba.loc[data_nba['nombre_jugador'] == 'Axel Toupane', 'posicion'] = 'E-A'
data_nba.loc[data_nba['nombre_jugador'] == 'Salah Mejri', 'posicion'] = 'P'
data_nba.loc[data_nba['nombre_jugador'] == 'Cristiano Felicio', 'posicion'] = 'P'
data_nba.loc[data_nba['nombre_jugador'] == 'Christian Wood', 'posicion'] = 'AP'
data_nba.loc[data_nba['nombre_jugador'] == 'Alex Stepheson', 'posicion'] = 'AP'
data_nba.loc[data_nba['nombre_jugador'] == 'Coty Clarke', 'posicion'] = 'A'
data_nba.loc[data_nba['nombre_jugador'] == 'Cliff Alexander', 'posicion'] = 'AP'
data_nba.loc[data_nba['nombre_jugador'] == 'Luis Montero', 'posicion'] = 'E-A'
data_nba.loc[data_nba['nombre_jugador'] == "Maurice Ndour", 'posicion'] = 'P'
data_nba.loc[data_nba['nombre_jugador'] == "JJ O'Brien", 'posicion'] = 'A'

data_nba.dropna(inplace = True)

position = data_nba.posicion.str.replace('B','Base').replace('A','Alero').replace('P','Pivot').replace('AP','Ala-Pivot').replace('A-E','Ala-Escolta').replace('E-A','Escolta-Alero').replace('PG','Base')
data_nba['posicion_juego'] = position
data_nba.drop(columns = 'posicion', inplace = True)

df_nba = data_nba.loc[(data_nba.minutos_total != 0) & (data_nba.minutos_total > 48)]

pir = df_nba.loc[(df_nba.temporada >= '2018-2019') & (df_nba.temporada <= '2020-2021'), ['temporada',
                                                                                         'nombre_jugador',
                                                                                         'nombre_equipo',
                                                                                         'partidos_jugados',
                                                                                         'asistencias_total',
                                                                                         'tapones_total',
                                                                                         'tci_total',
                                                                                         'tce_total',
                                                                                         'tli_total',
                                                                                         'tle_total',                                                                
                                                                                         'tti_total',
                                                                                         'tte_total',
                                                                                         'puntos_total',
                                                                                         'rebotes_total',
                                                                                         'perdidas_total',
                                                                                         'robos_total',
                                                                                         'faltas_total',
                                                                                         'faltas_tecnicas_total']]

for player in pir:
    
    valoracion = (df_nba['asistencias_total'] + df_nba['tapones_total'] + df_nba['tce_total'] + 
                  df_nba['tle_total'] + df_nba['tte_total'] + df_nba['puntos_total'] + 
                  df_nba['rebotes_total'] + df_nba['robos_total']- df_nba['tci_total'] - 
                  df_nba['tli_total'] - df_nba['tti_total'] - df_nba['perdidas_total'] - 
                  df_nba['faltas_total'] - df_nba['faltas_tecnicas_total'])
    
    valoracion_media = valoracion / df_nba['partidos_jugados']
    

df_pir = pd.DataFrame({'pir_medio_total':round(valoracion_media, 2)})    
    
df_nba = pd.concat([df_nba, df_pir], axis=1)

df_nba.loc[((df_nba['nombre_jugador'] == 'Lou Williams') & (df_nba['partidos_jugados'] == 4)), 
           'partidos_jugados'] = 80
df_nba.loc[((df_nba['nombre_jugador'] == 'Lou Williams') & (df_nba['pir_medio_total'] == 136.25)), 
           'pir_medio_total'] = 6.81

df_nba.to_csv('/home/inma/Escritorio/The Bridge/Data_Science_Curso/Clase/Repository_Inma/ML/ML_NBA_Inma Jimenez/scr/data/processed_files/data_nba_processed.csv', index = False)



## Entrenamiento del modelo

data_nba_2 = pd.read_csv('/home/inma/Escritorio/The Bridge/Data_Science_Curso/Clase/Repository_Inma/ML/ML_NBA_Inma Jimenez/scr/data/processed_files/data_nba_processed.csv')

df_nba_2 = data_nba_2.loc[(data_nba_2['temporada'] != '2022-2023')]


'''
Se pretende predecir el PIR de los jugadores. El PIR es una medida calculada a través de otras features de este DataSet, por ello se eliminan las features 
utilizadas para conformar esa feature y aquellas que correlacionen altamente con las variables a través de las que se calcula el PIR medio.
Posteriormente se observa la correlación entre el PIR medio de los jugadores y el resto de las features.

'''
df_final = df_nba_2.copy()

df_final.drop(columns = ['asistencias_total', 'tapones_total', 'tce_total', 'tle_total', 'tte_total',
                              'puntos_total', 'rebotes_total', 'robos_total', 'tci_total', 'tli_total', 
                              'tti_total', 'perdidas_total', 'faltas_total', 'faltas_tecnicas_total', 
                              'partidos_jugados', 'asistencias_pp', 'tapones_pp', 'eficiencia_tiro_pp', 'tci_pp',
                              'tce_pp', 'porcentaje_tc_pp', 'tli_pp', 'tle_pp', 'porcentaje_tl_pp', 'tti_pp', 
                              'tte_pp', 'porcentaje_tt_pp', 'puntos_pp', 'reb_def_pp', 'reb_of_pp', 
                              'total_rebotes_pp', 'faltas_pp', 'robos_pp', 'perdidas_pp', 'eficiencia_tiro_total',
                              'porcentaje_tc_total', 'porcentaje_tl_total', 'porcentaje_tt_total', 
                              'reb_def_total', 'reb_of_total'],
                   inplace = True)

df_final.drop(columns = ['minutos_total', 'altura', 'peso_kg', 'id_jugador', 'draft_anio', 'anio'],
                   inplace = True)


# GridSearch

# Se transforma la variable 'posicion' de categórica a numérica haciendo un 'Ordinal Encoding'
posicion_dict = {'Base': 0,
                 'Alero': 1, 
                 'Ala-Pivot': 2, 
                 'Pivot': 3, 
                 'Escolta-Alero': 4, 
                 'Ala-Escolta': 5 
                }

df_final['posicion_juego'] = df_final.posicion_juego.replace(posicion_dict)

X_6 = df_final.drop(columns = ["pir_medio_total", 'temporada','nombre_jugador',
                                      'ciudad_equipo','nombre_equipo', 'conferencia'])

y_6 = df_final["pir_medio_total"]

# Se definen los hiperparámetros del GridSearch
param_grid = {'learning_rate': [0.01,0.02,0.03,0.04],
              'subsample'    : [0.9, 0.5, 0.2, 0.1],
              'n_estimators' : [100,500,1000, 1500],
              'max_depth'    : [4,6,8,10]
             }

gb_model_6 = GradientBoostingRegressor()

# Se lleva a cabo el GridSearch
grid_search = GridSearchCV(gb_model_6,
                           param_grid,
                           cv=5, 
                           scoring = 'r2',
                           n_jobs = -1
                          )

# Se entrena el modelo
grid_search.fit(X_6, y_6)

grid_search.best_params_
grid_search.best_score_
grid_search.best_estimator_.score(X_6, y_6)

# Se crea el modelo con los best params obtenidos
model_grid_final = GradientBoostingRegressor(learning_rate = 0.04, 
                                             max_depth = 4, 
                                             n_estimators = 1500, 
                                             subsample = 0.2
                                            )


# Se guarda el modelo
with open('/home/inma/Escritorio/The Bridge/Data_Science_Curso/Clase/Repository_Inma/ML/ML_NBA_Inma Jimenez/scr/model_withparams_finished.model', 'wb') as file:
    pickle.dump(model_grid_final, file)