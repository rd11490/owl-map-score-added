import pandas as pd
from predict_functions import build_rmsa_map, calculate_tournament_table, sort_table, predict_match
from utils.constants import Maps, Teams, calc_map_type

# Pandas options for better printing
from utils.utils import calc_match_date, calc_season

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

match_data = pd.read_csv('map_data/match_map_stats.csv')

# Determine the map type, match date, and season for every map played
match_data['match_date'] = match_data['round_end_time'].apply(calc_match_date)
match_data['season'] = match_data['round_end_time'].apply(calc_season)


def convert_to_match_winner(group):
    team_one = group['team_one_name'].values[0]
    team_two = group['team_two_name'].values[0]
    season = group['season'].values[0]
    stage = group['stage'].values[0]
    date = group['match_date'].values[0]

    team_one_map_wins = 0
    team_two_map_wins = 0

    for ind in group.index:
        row = group.loc[ind, :]

        if row['map_winner'] == team_one:
            team_one_map_wins += 1
        elif row['map_winner'] == team_two:
            team_two_map_wins += 1

    if team_one_map_wins > team_two_map_wins:
        winner = team_one
        loser = team_two
    else:
        winner = team_two
        loser = team_one

    return pd.Series({
        'team_one': team_one,
        'team_one_map_wins': team_one_map_wins,
        'team_two': team_two,
        'team_two_map_wins': team_two_map_wins,
        'winner': winner,
        'loser': loser,
        'season': season,
        'stage': stage,
        'date': date
    })

results = match_data[
    ['match_id', 'stage', 'match_date', 'game_number', 'map_winner', 'map_loser', 'team_one_name', 'team_two_name',
     'season']] \
    .drop_duplicates()\
    .groupby(by=['match_id', 'game_number'])\
    .head(1)\
    .reset_index()\
    .groupby(by='match_id') \
    .apply(convert_to_match_winner) \
    .reset_index()

results.to_csv('results/match_results.csv', index=False)
print(results)
