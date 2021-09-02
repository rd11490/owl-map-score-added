import pandas as pd
from utils.constants import Maps, Teams

# Pandas options for better printing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)



map_scores_df = pd.read_csv('../results/scored_maps.csv')

map_scores_df = map_scores_df[map_scores_df['season'] == 2021]

team = Teams.Defiant

team_map_scores = map_scores_df[(map_scores_df['team_one_name'] == team) | (map_scores_df['team_two_name'] == team)].copy()

def calculate_net(row, team):
    if row['team_one_name'] == team:
        return row['team_one_score'] - row['team_two_score']
    else:
        return row['team_two_score']-row['team_one_score']

def average_map_score_diff(frame, team, map_type):
    map_type_df = frame[frame['map_type'] == map_type].copy()
    map_type_df['net score'] = map_type_df.apply(calculate_net, axis=1,args=[team])
    wins = map_type_df[map_type_df['net score'] > 0]['net score']
    losses = map_type_df[map_type_df['net score'] <= 0]['net score']
    in_wins = round(wins.mean(), 2)
    in_losses = round(losses.mean(),2)
    net_avg = round(map_type_df['net score'].mean(), 2)

    print(map_type_df)
    print('Average Net Score for {} on {} is {}. {} in {} wins, {} in {} losses'.format(team, map_type, net_avg, in_wins, wins.shape[0], in_losses, losses.shape[0]))

average_map_score_diff(team_map_scores, team, Maps.Control)
average_map_score_diff(team_map_scores, team, Maps.Escort)
average_map_score_diff(team_map_scores, team, Maps.Hybrid)
average_map_score_diff(team_map_scores, team, Maps.Assault)
