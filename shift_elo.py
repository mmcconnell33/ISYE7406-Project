# -*- coding: utf-8 -*-


'''
TODO:
    
    join shift pbp data with Moneypuck shot data
    matchup 'shotID' from mp 
    

'''

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_win_prob(elo1, elo2):
    team1_xw = 1/(1+10**((elo2-elo1)/400))
    team2_xw = 1/(1+10**((elo1-elo2)/400))
    
    return team1_xw, team2_xw
    

def calc_elo_diff(elo1, elo2, k, mov, team1_win):
    team1_xw, team2_xw = calc_win_prob(elo1, elo2)
    
    
    
    if team1_win:
        return k*(1-team1_xw)*(1+mov)
    else:
        return k*(1-team2_xw)*(1+mov)
    
    
def reset_elo(elo_dict):
    for k in elo_dict.keys():
        elo_dict[k]['elo_current'] = 1500
        elo_dict[k]['elo_list'] = []
        elo_dict[k]['qot_list'] = []
        elo_dict[k]['qoc_list'] = []
        elo_dict[k]['win_prob_list'] = []
        elo_dict[k]['result_list'] = []
        
    return elo_dict
    








# cd into correct directory as necessary
path = r'./pbp data'

# initialize dataframe
pbp_df = pd.DataFrame()

start_time = time.time()
for filename in os.listdir(path):
    
    # should be fine due to consistency in file name
    season = int(filename[7:11])
    
    print(f'Loading {filename} ...')
    current_df = pd.read_csv(f'{path}/{filename}')
    current_df['pbp_season'] = season
    pbp_df = pd.concat([pbp_df, current_df])


end_time = time.time()
run_time = end_time - start_time

print(f'All play by play data loaded in {run_time} seconds')


# load in shot data
path = r'./shot data'

# initialize dataframe
mp_df = pd.DataFrame()

start_time = time.time()
for filename in os.listdir(path):
    
    print(f'Loading {filename} ...')
    current_df = pd.read_csv(f'{path}/{filename}')
    mp_df = pd.concat([mp_df, current_df])


end_time = time.time()
run_time = end_time - start_time
print(f'All shot data loaded in {run_time} seconds')



# create cumulative time column for joining purposes
# time should add 1200 seconds for every period that has past because in 
# the pbp data, the 'Seconds_Elapsed' field resets to 0 at every period
pbp_df['cumulative_time'] = (1200 * (pbp_df['Period']-1))+ pbp_df['Seconds_Elapsed']

# player ids
ap_1_id_list = list(pbp_df['awayPlayer1_id'])
ap_2_id_list = list(pbp_df['awayPlayer2_id'])
ap_3_id_list = list(pbp_df['awayPlayer3_id'])
ap_4_id_list = list(pbp_df['awayPlayer4_id'])
ap_5_id_list = list(pbp_df['awayPlayer5_id'])
hp_1_id_list = list(pbp_df['homePlayer1_id'])
hp_2_id_list = list(pbp_df['homePlayer2_id'])
hp_3_id_list = list(pbp_df['homePlayer3_id'])
hp_4_id_list = list(pbp_df['homePlayer4_id'])
hp_5_id_list = list(pbp_df['homePlayer5_id'])



event_list = list(pbp_df['Event'])
time_list = list(pbp_df['Seconds_Elapsed'])
home_zone_list = list(pbp_df['Home_Zone'])

shift_id = 0
zone_start = 'Neu'
zs_num = 0
shift_start_time = 0


shift_id_list = []
zs_list = []
zs_num_list = []

for i in range(len(pbp_df)-1):
    
    if i % 100000 == 0:
        print(i)
    
    shift_id_list.append(shift_id)
    zs_list.append(zone_start)
    zs_num_list.append(zs_num)
    
    
    
    # get skaters for current event
    current_away_skaters = set([    
        ap_1_id_list[i],
        ap_2_id_list[i],
        ap_3_id_list[i],
        ap_4_id_list[i],
        ap_5_id_list[i] 
        ])
    
    current_home_skaters = set([    
        hp_1_id_list[i],
        hp_2_id_list[i],
        hp_3_id_list[i],
        hp_4_id_list[i],
        hp_5_id_list[i] 
        
        ])
    
    # get skaters for next event
    next_away_skaters = set([     
        ap_1_id_list[i+1],
        ap_2_id_list[i+1],
        ap_3_id_list[i+1],
        ap_4_id_list[i+1],
        ap_5_id_list[i+1] 
        ])
    
    next_home_skaters = set([     
        hp_1_id_list[i+1],
        hp_2_id_list[i+1],
        hp_3_id_list[i+1],
        hp_4_id_list[i+1],
        hp_5_id_list[i+1] 
        ])
    
    
    # if current skaters have changed, caclulate the elo difference
    if (current_away_skaters != next_away_skaters) or (
            current_home_skaters != next_home_skaters):
    
        shift_id +=1
        shift_start_time = time_list[i+1]
        
        if event_list[i+1] == 'FAC':
            zone_start = home_zone_list[i+1]
            
            if zone_start == 'Off':
                zs_num = 1
            else:
                zs_num =-1
            
        else:
            zone_start = 'Fly'
            zs_num = 0
    
shift_id_list.append(shift_id -1)
zs_list.append(zone_start)
zs_num_list.append(zs_num)

pbp_df['shift_id'] = shift_id_list
pbp_df['home_zone_start'] = zs_list
pbp_df['zone_start_num'] = zs_num_list 


# create zone starts for each shift from perspective of the home team:
    # 1 if offensive
    # -1 if defensive
    # 0 if neutral or on the fly


start_time = time.time()
print('Joining and cleaning DataFrames...')
# join the two on game_id, time elapsed, event type (shot, miss, goal)
new_df = pd.merge(pbp_df, mp_df, how = 'inner',
                  left_on = ['Game_Id', 'pbp_season', 'cumulative_time', 'p1_ID', 'Event'],
                  right_on = ['game_id', 'season', 'time', 'shooterPlayerId', 'event'])


# find shots from mp shot but that are missing from joined df
#mp_df[~mp_df.isin(new_df)].dropna()


# filter to 5v5 only
new_df = new_df[new_df['Strength'] == '5x5']

# sort the dataframe chronologically
new_df = new_df.sort_values(by = ['Date'])

end_time = time.time()
run_time = end_time - start_time
print(f'DataFrames joined in {run_time} seconds')


start_time = time.time()
print('Initializing Elo Dictionary...')

# convert to list for speed purposes

# team info
event_team_list = list(new_df['Ev_Team'])
home_team_list = list(new_df['Home_Team'])
away_team_list = list(new_df['Away_Team'])


# player names
ap_1_list = list(new_df['awayPlayer1'])
ap_2_list = list(new_df['awayPlayer2'])
ap_3_list = list(new_df['awayPlayer3'])
ap_4_list = list(new_df['awayPlayer4'])
ap_5_list = list(new_df['awayPlayer5'])
hp_1_list = list(new_df['homePlayer1'])
hp_2_list = list(new_df['homePlayer2'])
hp_3_list = list(new_df['homePlayer3'])
hp_4_list = list(new_df['homePlayer4'])
hp_5_list = list(new_df['homePlayer5'])

# player ids
ap_1_id_list = list(new_df['awayPlayer1_id'])
ap_2_id_list = list(new_df['awayPlayer2_id'])
ap_3_id_list = list(new_df['awayPlayer3_id'])
ap_4_id_list = list(new_df['awayPlayer4_id'])
ap_5_id_list = list(new_df['awayPlayer5_id'])
hp_1_id_list = list(new_df['homePlayer1_id'])
hp_2_id_list = list(new_df['homePlayer2_id'])
hp_3_id_list = list(new_df['homePlayer3_id'])
hp_4_id_list = list(new_df['homePlayer4_id'])
hp_5_id_list = list(new_df['homePlayer5_id'])

# xGoals
xg_list = list(new_df['xGoal'])
score_diff_list = list(new_df['homeTeamGoals'] - new_df['awayTeamGoals'])
zs_list = list(new_df['zone_start_num'])

# other important info
season_list = list(new_df['season'])





# initialize elo dictionary
elo_dict = {}


def clean_id_list(id_list, name_list):
    
    for i,j in enumerate(id_list):
        if np.isnan(j):
            id_list[i] = name_list[i]

# get all player id's and their names into a unique set for the elo dictionary
clean_id_list(ap_1_id_list, ap_1_list) 
clean_id_list(ap_2_id_list, ap_2_list)  
clean_id_list(ap_3_id_list, ap_3_list) 
clean_id_list(ap_4_id_list, ap_4_list) 
clean_id_list(ap_5_id_list, ap_5_list) 
clean_id_list(hp_1_id_list, hp_1_list) 
clean_id_list(hp_2_id_list, hp_2_list) 
clean_id_list(hp_3_id_list, hp_3_list) 
clean_id_list(hp_4_id_list, hp_4_list) 
clean_id_list(hp_5_id_list, hp_5_list)    


all_ids = (
    ap_1_id_list +
    ap_2_id_list +
    ap_3_id_list +
    ap_4_id_list +
    ap_5_id_list +
    hp_1_id_list +
    hp_2_id_list +
    hp_3_id_list +
    hp_4_id_list +
    hp_5_id_list )


all_names = ( 
    ap_1_list +
    ap_2_list +
    ap_3_list +
    ap_4_list +
    ap_5_list +
    hp_1_list +
    hp_2_list +
    hp_3_list +
    hp_4_list +
    hp_5_list )


unique_ids = set(all_ids)


for uid in unique_ids:
    
    id_index = all_ids.index(uid)
    name = all_names[id_index]
    elo_dict[uid] = {'name':name,
                       'elo_current': 1500,
                       'elo_list': [],
                       'qot_list':[],
                       'qoc_list':[],
                       'win_prob_list': [],
                       'result_list': []} 
    
end_time = time.time()
run_time = end_time - start_time
print(f'Dictionary initialized in {run_time} seconds')

elo_dict_original = elo_dict.copy()



# start loop here...
elo_k_list = [3] # 3 or 4 found to be optimal
mov_multipliers = [0] # 0 found to be optimal
score_diff_adj = [15] # 15 - 20 found to be optimal
zone_start_adj = [75] # 75 - 90 found to be optimal


results_dict = {}

for elo_k in elo_k_list:
    for mov_mult in mov_multipliers:
        for sd_adj in score_diff_adj:
            for zs_adj in zone_start_adj:
        
                key = f'{elo_k}, {mov_mult}, {sd_adj}, {zs_adj}'
                
                results_dict[key] = {'probability':[],
                                      'result': []
                                      }
            
                print(f'''\n\nCalculating Elo for k = {elo_k}, mov = {mov_mult},
                      sd = {sd_adj}, zs = {zs_adj}\n\n''')
                
            
                # reset elo_dict
                elo_dict = reset_elo(elo_dict)   
                    
            
                # initialize variables to track
                home_xg = 0
                away_xg = 0
                season = None
                home_win_prob = []
                home_win_result = []
                
                
                start_time = time.time()
                
                #mov_list_test = []
                
                for i in range(len(new_df)-1):
                    
                    if season_list[i] != season:
                        season = season_list[i]
                        print(f'Calculating {season} season... row {i}')
                        
                        
                    # home team took the shot
                    if event_team_list[i] == home_team_list[i]:
                        home_xg += xg_list[i]
                    elif event_team_list[i] == away_team_list[i]:
                        away_xg += xg_list[i]
                    else:
                        print('check event ' + str(i))
                    
                    
                    # get skaters for current event
                    current_away_skaters = set([    
                        ap_1_id_list[i],
                        ap_2_id_list[i],
                        ap_3_id_list[i],
                        ap_4_id_list[i],
                        ap_5_id_list[i] 
                        ])
                    
                    current_home_skaters = set([    
                        hp_1_id_list[i],
                        hp_2_id_list[i],
                        hp_3_id_list[i],
                        hp_4_id_list[i],
                        hp_5_id_list[i] 
                        
                        ])
                    
                    # get skaters for next event
                    next_away_skaters = set([     
                        ap_1_id_list[i+1],
                        ap_2_id_list[i+1],
                        ap_3_id_list[i+1],
                        ap_4_id_list[i+1],
                        ap_5_id_list[i+1] 
                        ])
                    
                    next_home_skaters = set([     
                        hp_1_id_list[i+1],
                        hp_2_id_list[i+1],
                        hp_3_id_list[i+1],
                        hp_4_id_list[i+1],
                        hp_5_id_list[i+1] 
                        ])
                    
                    
                    # if current skaters have changed, caclulate the elo difference
                    if (current_away_skaters != next_away_skaters) or (
                            current_home_skaters != next_home_skaters):
                        
                        home_avg_elo = np.mean(
                            [elo_dict[i]['elo_current'] for i in current_home_skaters])
                        
                        away_avg_elo = np.mean(
                            [elo_dict[i]['elo_current'] for i in current_away_skaters])
                        
                        
                        # adjust effective Elo based on score effects
                        home_avg_elo -= score_diff_list[i]*sd_adj
                        home_avg_elo += zs_list[i]*zs_adj
                        
                        
                        home_wp, away_wp = calc_win_prob(home_avg_elo, away_avg_elo)
                        results_dict[key]['probability'].append(home_wp)
                        
    
    
                        mov = abs(home_xg - away_xg)*mov_mult
                        
                        
                        # home team "won" the shift
                        if home_xg > away_xg:
                            elo_diff = calc_elo_diff(home_avg_elo, away_avg_elo, elo_k, mov, True)
                            results_dict[key]['result'].append(1) 
                            
                            for uid in current_home_skaters:
                                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                                elo_dict[uid]['qoc_list'].append(away_avg_elo)
                                elo_dict[uid]['qot_list'].append(home_avg_elo)
                                elo_dict[uid]['elo_current'] += elo_diff
                                                                              
                                
                            for uid in current_away_skaters:
                                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                                elo_dict[uid]['qoc_list'].append(home_avg_elo)
                                elo_dict[uid]['qot_list'].append(away_avg_elo)
                                elo_dict[uid]['elo_current'] -= elo_diff
                            
                            
                            
                        else:
                            elo_diff = calc_elo_diff(home_avg_elo, away_avg_elo, elo_k, mov, False)
                            results_dict[key]['result'].append(0)
                
                            for uid in current_away_skaters:
                                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                                elo_dict[uid]['qoc_list'].append(home_avg_elo)
                                elo_dict[uid]['qot_list'].append(away_avg_elo)
                                elo_dict[uid]['elo_current'] += elo_diff
                                                                              
                                
                            for uid in current_home_skaters:
                                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                                elo_dict[uid]['qoc_list'].append(away_avg_elo)
                                elo_dict[uid]['qot_list'].append(home_avg_elo)
                                elo_dict[uid]['elo_current'] -= elo_diff
                            
                            
                        
                        # reset xg
                        home_xg = 0
                        away_xg = 0
                        
                 
                
                        
                 
                elo_df = pd.DataFrame.from_dict(elo_dict).T
                elo_df = elo_df.sort_values(by = ['elo_current'])
                
                
                end_time = time.time()
                run_time = end_time - start_time
                print(f'\n\nCompleted run in {run_time} seconds\n\n')


print(elo_df[['name', 'elo_current']].iloc[0:50])
print(elo_df[['name', 'elo_current']].iloc[-50:])


name = 'Patrice Bergeron'

elo_df[elo_df['name'] == name.upper()]['qoc_list']




def calc_log_loss(prob_list, result_list):
                
        log_loss = 0
        for i, j in enumerate(prob_list):
            
            error = 1 - abs(result_list[i] - j)
            
            log_loss += -np.log(error)
            
        log_loss = log_loss/len(prob_list)
            
        return log_loss
    
def calc_brier_score(prob_list, result_list):
                
        brier_score = 0
        for i, j in enumerate(prob_list):
            
            brier_score += (result_list[i] - j)**2
            
        brier_score = brier_score/len(prob_list)
            
        return brier_score
    
            
    
def classification_error(prob_list, threshold, result_list):
    
    pred_list = [1 if p >= threshold else 0 for p in prob_list]
    return sum(i != j for i, j in zip(pred_list, result_list))/len(pred_list)



def kpi_measures(results_dict):
    
    graph_dict = {}
    
    for k in results_dict.keys():
        probs = results_dict[k]['probability']
        results = results_dict[k]['result']
        log_loss = calc_log_loss(probs, results)
        brier_score = calc_brier_score(probs, results)
        class_error = classification_error(probs, 0.5, results)
        
        graph_dict[k] = {}
        
        graph_dict[k]['log_loss'] = log_loss
        graph_dict[k]['brier_score'] = brier_score
        graph_dict[k]['classification_error'] = class_error
        
        
    
    return graph_dict


# turn dictionary into a dataframe
kpi_dict = kpi_measures(results_dict)
kpi_df = pd.DataFrame(kpi_dict).T

# add columns from index
k_col = []
mov_col = []
sd_col = []
zs_col = []

for val in kpi_df.index.values:
    params = val.split(',')
    k_col.append(int(params[0]))
    mov_col.append(int(params[1]))
    sd_col.append(int(params[2]))
    zs_col.append(int(params[3]))
    
kpi_df['k'] = k_col
kpi_df['xG Margin Multiplier'] = mov_col
kpi_df['Score Effect Adjustment'] = sd_col
kpi_df['Zone Start Adjustment'] = zs_col

print(kpi_df['classification_error'])
print(kpi_df['log_loss'])
print(kpi_df['brier_score'])



'''
np.where(kpi_df['log_loss'] == min(kpi_df['log_loss']))[0][0]
min(kpi_df['log_loss'])

# group dfs
df_dict = {}
for label, grp in kpi_df.groupby(['margin']):
    df_dict[label] = grp
    
x = df_dict[' add']['k']
y1 = df_dict[' add']['classification_error']
y2 = df_dict[' none']['classification_error']

plt.plot(x, y1, label = 'add')
plt.plot(x, y2, label = 'none')
plt.show()    
'''    


def kpi_graph(x, y, metric, save = True):
    
    min_index = np.where(y == min(y))[0][0]
    min_x = x[min_index]

    plt.plot(x, y, linewidth=2, markersize=2)
    plt.axhline(y = min(y), color = 'black', linestyle = 'dashed')
    plt.scatter(min_x, min(y), c = 'red', marker = 'x')
    plt.xlabel(x.name)
    plt.ylabel(metric)
    plt.title(f'Elo {metric} for Varying Parameter {x.name}')
    
    if save:
        plt.savefig(f'Elo {metric} - {x.name}')
    plt.show()
    
    


x = kpi_df['Score Effect Adjustment']
y_ll = kpi_df['log_loss']
y_bs = kpi_df['brier_score']
y_ce = kpi_df['classification_error']

kpi_graph(x, y_ll, 'Log Loss', save = False)
kpi_graph(x, y_bs, 'Brier Score', save = False)
kpi_graph(x, y_ce, 'Classification Error', save = False)



#plt.hist(results_dict[1]['probability'], bins = 20)


def graph_elo(name, window):
    
    # TODO:
        # plot an x shift moving average
        
    y = elo_df[elo_df['name']==name.upper()]['elo_list'].iloc[0]
    x = [i for i in range(len(y))]
    
    
    #window = int(round(len(y)/25,-2))
    
    y_series = pd.Series(y)
    elo_ma = y_series.rolling(window).mean()
    avg_elo = np.mean(y)
    
    min_index = y.index(min(y))
    max_index = y.index(max(y))
    
    min_x = x[min_index]
    max_x = x[max_index]
    
    plt.plot(x,y)
    plt.plot(x, elo_ma, color = 'black')
    plt.axhline(y = 1500, color = 'black', linestyle = 'dashed')
    plt.axhline(y = avg_elo, color = 'red', linestyle = 'dashed')
    plt.scatter(min_x, min(y), c = 'red', marker = 'x')
    plt.scatter(max_x, max(y), c = 'green', marker = 'x')
    plt.title(f'{name.upper()} - 5v5 Shift Elo')
    plt.xlabel('Shift Number')
    plt.ylabel('Elo Rating')
    plt.legend(['Shift Elo', f'{window}-shift Moving Avg', 'League Avg Elo', 'Player Avg Elo'], loc = 'upper left')
    plt.show()
    
graph_elo('brendan gallagher', 500)    

# multi graph...
def graph_multiple_elo(player_list):
    player_list = [player.upper() for player in player_list]
    
    
    names = elo_df[elo_df['name'].isin(player_list)]['name']
    y = elo_df[elo_df['name'].isin(player_list)]['elo_list']
    
    colors_list = ['#C8102E', 'black', 'blue', '#FFB81C']
    
    for i in range(len(y)):
        plt.plot(y.iloc[i], color = colors_list[i])
        
    plt.axhline(y = 1500, color = 'black', linestyle = 'dashed')
    plt.title('Player Elo Comparison')
    plt.xlabel('Shift Number')
    plt.ylabel('Elo Rating')
    plt.legend(names,
               loc = 'upper left')
    plt.show()
    

graph_multiple_elo(['brendan gallagher', 'zach parise'])
    

    
def player_corr(player1, player2, shift_start, shift_end):
    p1 = pd.Series(elo_df[elo_df['name'] == player1.upper()]['elo_list'][0][shift_start:shift_end])
    p2 = pd.Series(elo_df[elo_df['name'] == player2.upper()]['elo_list'][0][shift_start:shift_end])
    
    return p1.corr(p2)

player_corr('brendan gallagher', 'zach parise', 0, 10000)

    
    



elo_df['elo_current'].hist()
elo_df['avg_elo'] = elo_df['elo_list'].apply(lambda x: np.mean(x))
elo_df['max_elo'] = elo_df['elo_list'].apply(lambda x: np.max(x))
elo_df['min_elo'] = elo_df['elo_list'].apply(lambda x: np.min(x))
elo_df['shift_count'] = elo_df['elo_list'].apply(lambda x: len(x))
elo_df['elo_per_shift'] = (elo_df['elo_current']-1500)/elo_df['shift_count']


elo_df = elo_df.sort_values(by = ['min_elo'])
     
elo_filter_df = elo_df[elo_df['shift_count'] > 1000]           


print(elo_df[['name',  'min_elo', 'elo_per_shift', 'shift_count']].iloc[0:50])
print(elo_df[['name', 'elo_current', 'avg_elo', 'max_elo']].iloc[-10:])


plt.hist(elo_filter_df['elo'])





    
    
    
    

#for k,v in elo_dict.items():
#    print(k, v['name'], v['elo_current'])
         

