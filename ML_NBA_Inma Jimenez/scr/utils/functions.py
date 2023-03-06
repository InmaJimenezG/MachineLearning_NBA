import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

def NBA_players_data(season):
    
    data_players = []
    
    page = 0
    response = requests.get(f"https://es.global.nba.com/stats2/league/playerstats.json?conference=All&country=All&individual=All&locale=es&pageIndex={page}&position=All&qualified=false&season={season}&seasonType=2&split=All+Team&statType=points&team=All&total=perGame")
    json_response = response.json()

    data_players += json_response["payload"]["players"]
    
    while json_response["payload"]["players"] != []:
        page = page + 1
        response = requests.get(f"https://es.global.nba.com/stats2/league/playerstats.json?conference=All&country=All&individual=All&locale=es&pageIndex={page}&position=All&qualified=false&season={season}&seasonType=2&split=All+Team&statType=points&team=All&total=perGame")
        json_response = response.json()
        data_players += json_response["payload"]["players"]
        
        
    return data_players

def season_data(season):
   
    season_list = []
    
    page = 0
    response = requests.get(f"https://es.global.nba.com/stats2/league/playerstats.json?conference=All&country=All&individual=All&locale=es&pageIndex={page}&position=All&qualified=false&season={season}&seasonType=2&split=All+Team&statType=points&team=All&total=perGame")
    json_response = response.json()

    season_list += [json_response['payload']['season']]
        
        
    return season_list

def df_data_players(dataplayers):
    
    import pandas as pd

    id_jugador = []
    nombre_jugador = []
    draft_anio = []
    posicion = []
    altura = []
    peso = []
    rank = []
    ciudad_equipo = []
    nombre_equipo = []
    conferencia = []
    partidos_jugados = []
    asistencias_pp = []
    tapones_pp = []
    eficiencia_tiro_pp = []
    tiros_campo_intentados_pp = []
    tiros_campo_encestados_pp = []
    porcentaje_tiros_campo_pp = []
    tiros_libres_intentados_pp = []
    tiros_libres_encestados_pp = []
    porcentaje_tiros_libres_pp = []
    tiros_triple_intentados_pp = []
    tiros_triple_encestados_pp = []
    porcentaje_tiros_triple_pp = []
    puntos_pp = []
    minutos_pp = []
    rebotes_defensivos_pp = []
    rebotes_ofensivos_pp = []
    total_rebotes_pp = []
    faltas_pp = []
    robos_pp = []
    perdidas_pp = []
    asistencias_total = []
    tapones_total = []
    eficiencia_tiro_total = []
    tiros_campo_intentados_total = []
    tiros_campo_encestados_total = []
    porcentaje_tiros_campo_total = []
    tiros_libres_intentados_total =  []
    tiros_libres_encestados_total = []
    porcentaje_tiros_libres_total = []
    tiros_triple_intentados_total = []
    tiros_triple_encestados_total = []
    porcentaje_tiros_triple_total = []
    minutos_total = []
    puntos_total = []
    rebotes_defensivos_total = []
    rebotes_ofensivos_total = []
    rebotes_total = []
    perdidas_total = []
    robos_total = []
    faltas_total = []
    faltas_tecnicas_total = []
    


    for item in dataplayers:   

        id_jugador.append(item['playerProfile']['playerId'])   
        nombre_jugador.append(item['playerProfile']['displayName'])
        draft_anio.append(int(item['playerProfile']['draftYear']))
        posicion.append(item['playerProfile']['position'])
        altura.append(item['playerProfile']['height'].replace(',', '.').replace(' ',''))
        peso.append(item['playerProfile']['weight'].replace(' kilogramos','').replace(',', '.'))
        rank.append(item['rank'])
        ciudad_equipo.append(item['teamProfile']['cityEn'])
        nombre_equipo.append(item['teamProfile']['name'])
        conferencia.append(item['teamProfile']['conference'])
        partidos_jugados.append(item['statAverage']['games'])
        asistencias_pp.append(item['statAverage']['assistsPg'])
        tapones_pp.append(item['statAverage']['blocksPg'])
        eficiencia_tiro_pp.append(item['statAverage']['efficiency'])
        tiros_campo_intentados_pp.append(item['statAverage']['fgaPg'])
        tiros_campo_encestados_pp.append(item['statAverage']['fgmPg'])
        porcentaje_tiros_campo_pp.append(item['statAverage']['fgpct'])
        tiros_libres_intentados_pp.append(item['statAverage']['ftaPg'])
        tiros_libres_encestados_pp.append(item['statAverage']['ftmPg'])
        porcentaje_tiros_libres_pp.append(item['statAverage']['ftpct'])
        tiros_triple_intentados_pp.append(item['statAverage']['tpaPg'])
        tiros_triple_encestados_pp.append(item['statAverage']['tpmPg'])
        porcentaje_tiros_triple_pp.append(item['statAverage']['tppct'])
        puntos_pp.append(item['statAverage']['pointsPg'])
        minutos_pp.append(item['statAverage']['minsPg'])
        rebotes_defensivos_pp.append(item['statAverage']['defRebsPg'])
        rebotes_ofensivos_pp.append(item['statAverage']['offRebsPg'])
        total_rebotes_pp.append(item['statAverage']['rebsPg'])
        faltas_pp.append(item['statAverage']['foulsPg'])
        robos_pp.append(item['statAverage']['stealsPg'])
        perdidas_pp.append(item['statAverage']['turnoversPg'])
        asistencias_total.append(item['statTotal']['assists'])
        tapones_total.append(item['statTotal']['blocks'])
        eficiencia_tiro_total.append(item['statTotal']['efficiency'])
        tiros_campo_intentados_total.append(item['statTotal']['fga'])
        tiros_campo_encestados_total.append(item['statTotal']['fgm'])
        porcentaje_tiros_campo_total.append(item['statTotal']['fgpct'])
        tiros_libres_intentados_total.append(item['statTotal']['fta'])
        tiros_libres_encestados_total.append(item['statTotal']['ftm'])
        porcentaje_tiros_libres_total.append(item['statTotal']['ftpct'])
        tiros_triple_intentados_total.append(item['statTotal']['tpa'])
        tiros_triple_encestados_total.append(item['statTotal']['tpm'])
        porcentaje_tiros_triple_total.append(item['statTotal']['tppct'])
        minutos_total.append(item['statTotal']['mins'])
        puntos_total.append(item['statTotal']['points'])
        rebotes_defensivos_total.append( item['statTotal']['defRebs'])
        rebotes_ofensivos_total.append(item['statTotal']['offRebs'])
        rebotes_total.append(item['statTotal']['rebs'])
        perdidas_total.append(item['statTotal']['turnovers'])
        robos_total.append(item['statTotal']['steals'])
        faltas_total.append(item['statTotal']['fouls'])
        faltas_tecnicas_total.append(item['statTotal']['technicalFouls'])
        

    
    df_players = pd.DataFrame({'id_jugador': id_jugador,
                               'nombre_jugador': nombre_jugador,
                               'draft_anio': draft_anio,
                               'posicion': posicion,
                               'altura': altura,
                               'peso_kg': peso,
                               'num_ranking': rank,
                               'ciudad_equipo': ciudad_equipo,
                               'nombre_equipo': nombre_equipo,
                               'conferencia': conferencia,
                               'partidos_jugados': partidos_jugados,
                               'asistencias_pp': asistencias_pp,
                               'tapones_pp': tapones_pp,
                               'eficiencia_tiro_pp': eficiencia_tiro_pp,
                               'tci_pp': tiros_campo_intentados_pp,
                               'tce_pp': tiros_campo_encestados_pp,
                               'porcentaje_tc_pp': porcentaje_tiros_campo_pp,
                               'tli_pp': tiros_libres_intentados_pp,
                               'tle_pp': tiros_triple_encestados_pp,
                               'porcentaje_tl_pp': porcentaje_tiros_libres_pp,
                               'tti_pp': tiros_triple_intentados_pp,
                               'tte_pp': tiros_triple_encestados_pp,
                               'porcentaje_tt_pp': porcentaje_tiros_triple_pp,
                               'puntos_pp': puntos_pp,
                               'minutos_pp': minutos_pp,
                               'reb_def_pp': rebotes_defensivos_pp,
                               'reb_of_pp': rebotes_ofensivos_pp,
                               'total_rebotes_pp': total_rebotes_pp,
                               'faltas_pp': faltas_pp,
                               'robos_pp': robos_pp,
                               'perdidas_pp': perdidas_pp,
                               'asistencias_total': asistencias_total,
                               'tapones_total': tapones_total,
                               'eficiencia_tiro_total':  eficiencia_tiro_total,
                               'tci_total': tiros_campo_intentados_total,
                               'tce_total': tiros_campo_encestados_total,
                               'porcentaje_tc_total': porcentaje_tiros_campo_total,
                               'tli_total': tiros_libres_intentados_total,
                               'tle_total': tiros_libres_encestados_total,
                               'porcentaje_tl_total': porcentaje_tiros_libres_total,
                               'tti_total': tiros_triple_intentados_total,
                               'tte_total': tiros_triple_encestados_total,
                               'porcentaje_tt_total': porcentaje_tiros_triple_total,
                               'minutos_total': minutos_total,
                               'puntos_total': puntos_total,
                               'reb_def_total': rebotes_defensivos_total,
                               'reb_of_total': rebotes_ofensivos_total,
                               'rebotes_total': rebotes_total,
                               'perdidas_total': perdidas_total,
                               'robos_total': robos_total,                            
                               'faltas_total': faltas_total,
                               'faltas_tecnicas_total': faltas_tecnicas_total                               
                              })
    
    return df_players

def df_data_season(dataseason):
    
    temporada = [] # Para que contenga el año de la temporada sobre la que se crea el DataFrame
    year = [] # Para que contenga el año de comienzo de la temporada
    
    for i in dataseason:
        temporada.append(i['yearDisplay'])
        year.append(int(i['year']))
   

    df_season = pd.DataFrame({'temporada': temporada,
                             'anio': year})

    
    return df_season

def df_total(df1, df2):
    
    df = pd.concat([df2, df1], axis = 1)
    
    for row in range(len(df['temporada'])):
        df['temporada'] = df['temporada'].fillna(df['temporada'][0])
        df['anio'] = df['anio'].fillna(df['anio'][0])
    
    return df

def sustitution(df, column_name, row_name, value_1, value_2):
    
    for index, row in (df.loc[df[column_name].isnull()]).iterrows():
    
        if row[row_name] == value_1:
            df[column_name][index] = value_2
            
    
    return df

def data_report(df):

    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T

def plot_feature_importance(features, model_importances):

    indices = np.argsort(model_importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), model_importances[indices], color='y', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()