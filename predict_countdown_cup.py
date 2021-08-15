import pandas as pd
from predict_functions import build_rmsa_map, calculate_tournament_table, sort_table, predict_match, \
    predict_tournament_cycle
from utils.constants import Maps, Teams

# Pandas options for better printing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

week_one_rotation = [Maps.Control, Maps.Hybrid, Maps.Escort, Maps.Assault, Maps.Control]
week_two_rotation = [Maps.Control, Maps.Escort, Maps.Assault, Maps.Hybrid, Maps.Control]
week_three_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control]
finals_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control, Maps.Hybrid, Maps.Escort]


def map_rotation(date):
    if '2021-07-30' <= date < '2021-08-01':
        return week_one_rotation
    elif '2021-08-06' <= date < '2021-08-08':
        return week_one_rotation
    else:
        return week_one_rotation

schedule_frame = pd.read_csv('results/2021_league_schedule.csv')
rmsa_frame = pd.read_csv('results/rmsa.csv')

rmsa_map = build_rmsa_map(rmsa_frame)

countdown_cup = schedule_frame[
    (schedule_frame['startDate'] >= '2021-08-15') & (schedule_frame['startDate'] <= '2021-08-14')]

countdown_cup['map_rotation'] = countdown_cup['startDate'].apply(map_rotation)

known_results = pd.read_csv('results/manual_results.csv')

match_results = pd.read_csv('results/match_results.csv')
match_results = match_results[match_results['season'] == 2021]
cycle_results = match_results[match_results['date'] >= '2021/07/29']
cycle_results = cycle_results[['team_one', 'team_two', 'team_one_map_wins', 'team_two_map_wins', 'winner', 'loser']]

known_results = pd.concat([known_results, cycle_results])


all_east_results = []
all_west_results = []
all_match_results = []
tournament_results = []

def convert_tournament_lp_to_frame(lp):
    return pd.DataFrame([{'team': k, 'points': lp[k]} for k in lp.keys()])

for i in range(0, 10000):
    east, west, tournament_lp, results = predict_tournament_cycle(countdown_cup, rmsa_map, known_results)
    tournament_frame = convert_tournament_lp_to_frame(tournament_lp)
    tournament_frame['sim'] = i
    tournament_results.append(tournament_frame)

    east['sim_number'] = i
    west['sim_number'] = i

    east['rank'] = list(range(1, len(Teams.East) + 1))
    west['rank'] = list(range(1, len(Teams.West) + 1))

    all_east_results.append(east)
    all_west_results.append(west)
    all_match_results.append(results)

all_east_results = pd.concat(all_east_results, axis=0)
all_west_results = pd.concat(all_west_results, axis=0)

all_east_results.to_csv('results/cc_sim_results_east.csv', index=False)
all_west_results.to_csv('results/cc_sim_results_west.csv', index=False)

west_avg = all_west_results[['team', 'wins', 'losses', 'map_differential', 'rank']].groupby('team').mean().reset_index()
east_avg = all_east_results[['team', 'wins', 'losses', 'map_differential', 'rank']].groupby('team').mean().reset_index()

west_avg = west_avg.sort_values(by='rank')
east_avg = east_avg.sort_values(by='rank')



tournament_results = pd.concat(tournament_results, axis=0)

print(west_avg)
print(east_avg)
print(tournament_results.groupby(by='team').agg(['mean', 'count']).reset_index())

tournament_results.to_csv('results/countdown_cup_predictions.csv', index=False)
