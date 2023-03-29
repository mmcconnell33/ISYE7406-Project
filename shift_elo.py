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


def calc_elo_diff(elo1, elo2, k, mov, team1_win):
    team1_xw = 1/(1+10**((elo2-elo1)/400))
    team2_xw = 1/(1+10**((elo1-elo2)/400))
    
    if team1_win:
        return k*(1-team1_xw)*mov
    else:
        return k*(1-team2_xw)*mov
    








# cd into correct directory as necessary
path = r'./pbp data'

# initialize dataframe
pbp_df = pd.DataFrame()

start_time = time.time()
for filename in os.listdir(path):
    
    print(f'Loading {filename} ...')
    current_df = pd.read_csv(f'{path}/{filename}')
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


# join the two on game_id, time elapsed, event type (shot, miss, goal)
new_df = pd.merge(pbp_df, mp_df, how = 'inner',
                  left_on = ['Game_Id', 'cumulative_time', 'p1_ID', 'Event'],
                  right_on = ['game_id', 'time', 'shooterPlayerId', 'event'])


# find shots from mp shot but that are missing from joined df
#mp_df[~mp_df.isin(new_df)].dropna()


# filter to 5v5 only
new_df = new_df[new_df['Strength'] == '5x5']


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



# initialize elo dictionary
elo_dict = {}

# get all player id's and their names into a unique set for the elo dictionary
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
                       'qot_avg':1500,
                       'qot_list':[],
                       'qoc_avg':1500,
                       'qoc_list':[]}  




home_xg = 0
away_xg = 0

for i in range(len(new_df)-1):
    
    if i % 1000 == 0:
        print('Loading row ' + str(i))
        
    # home team took the shot
    if event_team_list[i] == home_team_list[i]:
        home_xg += xg_list[i]
    elif event_team_list[i] == away_team_list[i]:
        away_xg += xg_list[i]
    else:
        print('check event ' + str(i))
    
    
    
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
    
    
    # if lines have changed, caclulate the elo difference
    if (current_away_skaters != next_away_skaters) or (
            current_home_skaters != next_home_skaters):
        
        home_avg_elo = np.mean(
            [elo_dict[i]['elo_current'] for i in current_home_skaters])
        
        away_avg_elo = np.mean(
            [elo_dict[i]['elo_current'] for i in current_away_skaters])
        
        
        mov = abs(home_xg - away_xg)
        
        # home team "won" the shift
        if home_xg > away_xg:
            elo_diff = calc_elo_diff(home_avg_elo, away_avg_elo, 20, mov, True)
            
            for uid in current_home_skaters:
                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                elo_dict[uid]['qoc_list'].append(away_avg_elo)
                elo_dict[uid]['qot_list'].append(home_avg_elo)
                
                #elo_dict[uid]['qoc_avg'] = np.mean(elo_dict[uid]['qoc_list'])
                #elo_dict[uid]['qot_avg'] = np.mean(elo_dict[uid]['qot_list'])
                elo_dict[uid]['elo_current'] += elo_diff
                                                              
                
            for uid in current_away_skaters:
                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                elo_dict[uid]['qoc_list'].append(home_avg_elo)
                elo_dict[uid]['qot_list'].append(away_avg_elo)
                
                #elo_dict[uid]['qoc_avg'] = np.mean(elo_dict[uid]['qoc_list'])
                #elo_dict[uid]['qot_avg'] = np.mean(elo_dict[uid]['qot_list'])                
                elo_dict[uid]['elo_current'] -= elo_diff
            
            
            
        else:
            elo_diff = calc_elo_diff(home_avg_elo, away_avg_elo, 20, mov, False)
            

            for uid in current_away_skaters:
                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                elo_dict[uid]['qoc_list'].append(home_avg_elo)
                elo_dict[uid]['qot_list'].append(away_avg_elo)
                
                #elo_dict[uid]['qoc_avg'] = np.mean(elo_dict[uid]['qoc_list'])
                #elo_dict[uid]['qot_avg'] = np.mean(elo_dict[uid]['qot_list'])
                elo_dict[uid]['elo_current'] += elo_diff
                                                              
                
            for uid in current_home_skaters:
                elo_dict[uid]['elo_list'].append(elo_dict[uid]['elo_current'])
                elo_dict[uid]['qoc_list'].append(away_avg_elo)
                elo_dict[uid]['qot_list'].append(home_avg_elo)

                #elo_dict[uid]['qoc_avg'] = np.mean(elo_dict[uid]['qoc_list'])
                #elo_dict[uid]['qot_avg'] = np.mean(elo_dict[uid]['qot_list'])                
                elo_dict[uid]['elo_current'] -= elo_diff
            
            
        
        # reset xg
        home_xg = 0
        away_xg = 0
        
 

        
 
elo_df = pd.DataFrame.from_dict(elo_dict).T
elo_df = elo_df.sort_values(by = ['elo_current'])


print(elo_df[['name', 'elo_current']].iloc[0:50])
print(elo_df[['name', 'elo_current']].iloc[-50:])


name = 'Patrice Bergeron'

elo_df[elo_df['name'] == name.upper()]['qoc_list']



def graph_elo(name):
    
    y = elo_df[elo_df['name']==name.upper()]['elo_list'].iloc[0]
    x = [i for i in range(len(y))]
    
    plt.plot(x,y)
    plt.show()
    
    
graph_elo('claude giroux')

    
    
    
    
    

#for k,v in elo_dict.items():
#    print(k, v['name'], v['elo_current'])
         

