import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import requests
from bs4 import BeautifulSoup
import re
import urllib.request
from sklearn import linear_model
import math

def getTeamStats(team_id, year, opp_team_id):
    global stats_df, wins_loss, opp_wins_loss
    team = team_id.lower()
    opp_team = opp_team_id.lower()
    url = "https://www.basketball-reference.com/international/teams/" + team + "/" + year + ".html"
    url_2 = "https://www.basketball-reference.com/international/teams/" + opp_team + "/" + year + ".html"
    #url = "https://www.basketball-reference.com/teams/" + team + "/" + year + ".html"
    #print(url)
    # print()
    with urllib.request.urlopen(url_2) as response:
        r2 = response.read().decode('latin-1')
    content2 = re.sub(r'(?m)^\<!--.*\n?', '', r2)
    content2 = re.sub(r'(?m)^\-->.*\n?', '', content2)
    soup2 = BeautifulSoup(content2, 'html.parser')
    opp_wins_losses = soup2.find_all(id='meta')
    opp_wins_loss = re.search("(\d+)[-](\d+)\sin", str(opp_wins_losses))

    with urllib.request.urlopen(url) as response:
        # UTF-8 doesn't support some initial character on the websites for some reason!
        r = response.read().decode('latin-1')

    content = re.sub(r'(?m)^\<!--.*\n?', '', r)
    content = re.sub(r'(?m)^\-->.*\n?', '', content)

    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.findAll('table')
    wins_losses = soup.find_all(id='meta')
    wins_loss = re.search("(\d+)[-](\d+)\sin", str(wins_losses))
    test_df = pd.read_html(str(tables[0]))[0]

    stats_index = 0
    for i in range(len(tables)):
            if len(pd.read_html(str(tables[i]))[0].index) == 8:
                stats_index = i + 1
            stats_table = tables[stats_index]

            stats_df = pd.read_html(str(stats_table))[0]

            stats_header = stats_df.columns.values.tolist()
    return stats_df, stats_header


def get_team_data(team_id, year, opp_team_id, header=False):
    team_stats_tuple = getTeamStats(team_id, year, opp_team_id)

    regular_stats_raw = np.array(team_stats_tuple[0].iloc[0,2:])
    regular_stats = list(map(lambda value: float(value), regular_stats_raw))
    regular_stats_header = team_stats_tuple[1][2:]

    stats_full = np.concatenate((regular_stats), axis=None)
    header_full = np.concatenate((regular_stats_header), axis=None)

    if header == True:
        return header_full
    else:
        return stats_full

def generate_dataframe(rows, header):
    df = pd.DataFrame(rows, columns=header)
    return df

def getTeamDf(team_id, year, opp_team_id):
    df_header = get_team_data(team_id, year, opp_team_id, header=True)
    df_row = [get_team_data(team_id, year, opp_team_id)]
    return generate_dataframe(df_row, df_header)

def elo(win_ratio):
    return (400 * math.log10(win_ratio / (1 - win_ratio)))

def elo_prediction(home_rating,away_rating):
    E_home = 1./(1 + 10 ** ((away_rating - home_rating) / (400.)))
    return E_home * 100

def team_score(team_id, year, opp_team_id):
    df = getTeamDf(team_id, year, opp_team_id)
    df['FTR'] = df['FT']/df['FGA']
    df['ORtg'] = round(100*df['ORB']/(df['ORB']+stats_df['DRB'][1]),3)
    df['DFtg'] = round(100*df['DRB']/(stats_df['ORB'][1]+df['DRB']),3)
    df['DRBT'] = (df['DRB']*stats_df['G'][0])
    df['TOV%'] = round(df['TOV']/( df['FGA'] + 0.44 * df['FTA'] + df['TOV']),3)
    df['eFG%'] = (df['FG']+0.5*df['3P'])/df['FGA']
    df['POS'] = (df['FGA']*stats_df['G'][0]) - (df['ORB']*stats_df['G'][0]) + (df['TOV']*stats_df['G'][0] ) + (0.475*(df['FTA']*stats_df['G'][0]))
    df['OFFEFF%'] = ((round((df['PTS']*stats_df['G'][0]),1)/df['POS']))*100
    df['DEFOFF%'] = ((stats_df['PTS'][1]*stats_df['G'][1])/df['POS'])*100
    df['MOV'] = stats_df['PTS'][0] - stats_df['PTS'][1]
    df['W%'] = float((int(wins_loss[1])/stats_df['G'][0]))
    #temp crossing out SOS to try ELO as SOS isn't compeltely accurate
    #df['SOS'] = (0.25*(df['W%']))+(0.50*(int(opp_wins_loss[1])/(int(opp_wins_loss[1])+int(opp_wins_loss[2]))))+(0.25*(int(wins_loss[2]))/stats_df['G'][0])
    #print(int(wins_loss[1])/stats_df['G'][0]) testing WR
    #print(team_id,df,df['SOS'])
    ##lin regress not great
    ##df['score'] = abs(float((0.76*df['OFFEFF%']) - ((0.87*df['DEFOFF%']) + df['MOV'] + df['TOV%'] + (10.0*df['eFG%']) + (0.66*df['DFtg']) + (0.15*df['FTR']) + df['SOS'])))
    #score = float((0.76*df['OFFEFF%']) - ((0.87*df['DEFOFF%']) + df['MOV'] + df['TOV%'] + (10.0*df['eFG%']) + (0.66*df['DFtg']) + (0.15*df['FTR']) + df['SOS']))
    ##x = df[['FGA','BLK','STL','ORtg','DFtg','TOV%','DRBT','eFG%','MOV','POS','OFFEFF%','DEFOFF%']]
    ##y = df['score']
    win_ratio = float(wins_loss[1])/float(stats_df['G'][0])#* 100 win ratio
    op_game_totals = int(opp_wins_loss[1]) + int(opp_wins_loss[2])
    opp_win_ratio = float(opp_wins_loss[1])/float(op_game_totals)#* 100 opp win ratio
    teamELO = elo(win_ratio) + 1500
    oppELO = elo(opp_win_ratio) + 1500
    teamExpW = elo_prediction(teamELO,oppELO)
    ##regrModel = linear_model.LinearRegression()
    ##regrModel.fit(x, y)
    ##score = regrModel.predict(df[['FGA','BLK','STL','ORtg','DFtg','TOV%','DRBT','eFG%','MOV','POS','OFFEFF%','DEFOFF%']])
    ##score = abs(score)
    ###Score with elo is implemented adnd expected wins but want t omake babetter using the 4 factors and weights
    score = float(0.76*df['OFFEFF%'] - ((0.87*df['DEFOFF%'] + df['MOV'] + 10.0*df['eFG%'] + 0.66*df['DFtg'] + teamExpW/10)))
    #print(team_id,teamELO,oppELO,elo_prediction(teamELO,oppELO)) #testing output
    #works 1 score = float(float(0.76*(df['OFFEFF%']*100) - 0.87*(df['DEFOFF%']*100) + 10.0*(df['eFG%']) + 0.15*df['FTR'])-(10*df['SOS']))#/df['SOS']) #+ df['SOS'])
    #score = float((0.40*df['eFG%']+ 0.25*df['TOV%'] + 0.20*(df['ORtg'] + df['DFtg']) + 0.10*df['FTR'] + 0.02*df['STL'] + 0.02*df['BLK'])/(df['DEFOFF%']*100)**df['OFFEFF%'])
    #devided by games decreases granulaituriity of data
    #print(df['OFFEFF%'])
    #print(team_id,wins_loss[1],wins_loss[2],stats_df['G'][0],df['SOS'],df['DEFOFF%'])
    #print(team_id,1/3*(int(wins_loss[1])/stats_df['G'][0]) + 2/3*(int(wins_loss[2])/stats_df['G'][0]))
    #print(stats_df['G'][1])
    #print(score)
    return abs(score)

def simulate_game(t1_year, t2_year, t1_id, t2_id, epochs=100000, home_variation_max=100, away_variation_max=100, display_info=False):
    
    if display_info:
        print("Simulation Presets:")
        print("Epochs: {}".format(epochs))
        print("Home Team Variation Range Max: {}".format(home_variation_max))
        print("Away Team Variation Range Max: {}".format(away_variation_max))
        print()

    t1_metric = team_score(t1_id, t1_year, t2_id)
    t2_metric = team_score(t2_id, t2_year, t1_id)

    if display_info:
        print('{} has a CORWIN score of {}'.format(t1_id, t1_metric))
        print('{} has a CORWIN score of {}'.format(t2_id, t2_metric))
        print()
    t1_wins = 0
    t2_wins = 0

    for i in range(epochs):
        t1_random_variation = random.randint(0, home_variation_max) / 10  # parameter these
        t2_random_variation = random.randint(0, away_variation_max) / 10   # parameter these
        t1_final_metric = t1_metric + t1_random_variation
        t2_final_metric = t2_metric + t2_random_variation

        if t1_final_metric > t2_final_metric:
            t1_wins +=1
        else:
            t2_wins +=1

    if display_info:
        print('In {} simulated games, {}: {} wins, {}: {} wins'.format(epochs, t1_id, t1_wins, t2_id, t2_wins))
    if t1_wins > t2_wins:
        return t1_year, t1_id, float(t1_wins)/float(epochs/100), t2_year, t2_id
    else:
        return t2_year, t2_id, float(t2_wins)/float(epochs/100), t1_year, t1_id

if __name__ == "__main__":
    try:
        win_team_year, win_team_id, win_team_percent, lose_team_year, lose_team_id = \
            simulate_game(sys.argv[1], sys.argv[3], sys.argv[2], sys.argv[4], display_info=True)
        print("{} {} has a {}% chance of beating {} {}.".format(win_team_year, win_team_id, win_team_percent, lose_team_year, lose_team_id))
    except:
        teamsAcr ='''
            *NOT BETTING ADVICE, JUST A EXPERIMENT, 68% Accuracy or more is the goal*
            Format: <home_year> <home_team_id> <away_year> <away_team_id> \n
            NBL:
            Melbourne UTD - melbourne \n
            Perth Wildcats - perth \n
            Brisbane Bullets - brisbane \n
            Illawarra Hawks - illawarra \n
            Adelaide 36ERS - adelaide \n
            Sydney Kings - sydney \n
            SE-Melbourne Phoenix - se-melbourne \n
            New Zealand Breakers - nbl-new-zealand \n
            Cairns Taipans - cairns \n
            CBA:
            Guangdong Southern Tigers  - guangdong \n
            Liaoning Flying Leopards - liaoning \n
            Zhejiang Golden Bulls - zhejiang-yiwu \n
            Xinjiang Flying Tigers  - xinjiang \n
            Zhejiang Lions  - zhejiang-hangzhou \n
            Shenzhen Aviators - shenzhen \n
            Sichuan Blue Whales - sichuan \n
            Qingdao Eagles - qingdao \n
            Shandong Heroes - shandong \n
            Jilin Northeast Tigers - jilin \n
            Beijing Ducks - beijing-shougang \n
            Beijing Royal Fighters - beijing-bg \n
            Shanghai Sharks  - shanghai \n
            Shanxi Loongs - shanxi \n
            Guangzhou Loong-Lions - guangzhou \n
            Tianjin Pioneers - tianjin \n
            Nanjing Monkey Kings - jiangsu-tongxi \n
            Fujian Sturgeons - fujian \n
            Jiangsu Dragons - jiangsu-dragons \n
            '''
        print(teamsAcr)
