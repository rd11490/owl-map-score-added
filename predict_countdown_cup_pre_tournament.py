import pandas as pd
from predict_functions import build_rmsa_map, calculate_tournament_table, sort_table, predict_match
from utils.constants import Maps, Teams

# Pandas options for better printing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

week_one_rotation = [Maps.Control, Maps.Hybrid, Maps.Escort, Maps.Assault, Maps.Control]
week_two_rotation = [Maps.Control, Maps.Escort, Maps.Assault, Maps.Hybrid, Maps.Control]
week_three_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control]


def map_rotation(date):
    if '2021-07-30' <= date < '2021-08-01':
        return week_one_rotation
    elif '2021-08-06' <= date < '2021-08-08':
        return week_one_rotation
    else:
        return week_one_rotation

def predict_matches(schedule, rsa_for_lookup):
    results_arr = []
    for ind in schedule.index:
        match = schedule.loc[ind, :]
        match_result = predict_match(match['team1Name'], match['team2Name'], map_rotation(match['startDate']),
                                     rsa_for_lookup, 3)
        results_arr.append(match_result)
    return results_arr

schedule_frame = pd.read_csv('results/2021_league_schedule.csv')
rmsa_frame = pd.read_csv('results/rmsa.csv')

rmsa_map = build_rmsa_map(rmsa_frame)

countdown_cup = schedule_frame[
    (schedule_frame['startDate'] >= '2021-07-30') & (schedule_frame['startDate'] <= '2021-08-14')]

all_east_results = []
all_west_results = []
all_match_results = []
for i in range(0, 1000):
    results = predict_matches(countdown_cup, rmsa_map)
    results_frame = pd.DataFrame(results)
    results_frame['sim'] = i
    all_match_results.append(results_frame)
    east, west = calculate_tournament_table(results_frame)

    east = sort_table(east)
    west = sort_table(west)

    east['sim_number'] = i
    west['sim_number'] = i

    east['rank'] = list(range(1, len(Teams.East) + 1))
    west['rank'] = list(range(1, len(Teams.West) + 1))

    all_east_results.append(east)
    all_west_results.append(west)

all_east_results = pd.concat(all_east_results, axis=0)
all_west_results = pd.concat(all_west_results, axis=0)

west_avg = all_west_results[['team', 'wins', 'losses', 'map_differential', 'rank']].groupby('team').mean().reset_index()
east_avg = all_east_results[['team', 'wins', 'losses', 'map_differential', 'rank']].groupby('team').mean().reset_index()

west_avg = west_avg.sort_values(by='rank')
east_avg = east_avg.sort_values(by='rank')

all_results_frame = pd.concat(all_match_results, axis=0)
res = all_results_frame.groupby(by=['team_one', 'team_two']).mean().reset_index()
print(res)

print(west_avg)
print(east_avg)
