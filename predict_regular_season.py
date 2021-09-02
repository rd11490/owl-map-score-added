import pandas as pd
from predict_functions import build_rmsa_map, calculate_tournament_table, predict_match, \
    predict_tournament_cycle, sort_season_table
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


playoff_dates = [
    '2021/05/02',
    '2021/05/07',
    '2021/05/08',
    '2021/05/09',

    '2021/06/06',
    '2021/06/11',
    '2021/06/12',
    '2021/06/13',

    '2021/07/11',
    '2021/07/12',
    '2021/07/16',
    '2021/07/17',
    '2021/07/18'
]

schedule_frame = pd.read_csv('results/2021_league_schedule.csv')
rmsa_frame = pd.read_csv('results/rmsa.csv')
rmsa_map = build_rmsa_map(rmsa_frame)

match_results = pd.read_csv('results/match_results.csv')
match_results = match_results[match_results['season'] == 2021]
match_results = match_results[match_results['date'].isin(playoff_dates) == False]

cycle_results = match_results[match_results['date'] >= '2021/07/29']

cycle_results = cycle_results[['team_one', 'team_two', 'team_one_map_wins', 'team_two_map_wins', 'winner', 'loser']]
season_results = match_results[match_results['date'] < '2021/07/29']
season_results = season_results[['team_one', 'team_two', 'team_one_map_wins', 'team_two_map_wins', 'winner', 'loser']]

remaining_schedule = schedule_frame[
    (schedule_frame['startDate'] >= '2021-08-15') & (schedule_frame['startDate'] <= '2021-08-14')]

remaining_schedule['map_rotation'] = remaining_schedule['startDate'].apply(map_rotation)


manual_results = pd.read_csv('results/manual_results.csv')

all_cycle_results = pd.concat([manual_results, cycle_results])

bonus_points = {
    'Shanghai Dragons': 8,
    'Chengdu Hunters': 2,
    'Florida Mayhem': 1,
    'Dallas Fuel': 6,
    'Atlanta Reign': 1
}

def calculate_league_points(row, total_bonus_points):
    bonus = 0
    if row['team'] in total_bonus_points.keys():
        bonus = total_bonus_points[row['team']]

    return row['wins'] + bonus




all_east_results = []
all_west_results = []
all_match_results = []
for i in range(0, 100):
    east, west, tournament_lp, tourny_results = predict_tournament_cycle(remaining_schedule, rmsa_map, all_cycle_results)
    all_results = pd.concat([tourny_results, season_results])
    season_east, season_west = calculate_tournament_table(all_results)

    bonus_lp = {} # need to do a deep copy
    for k in bonus_points.keys():
        bonus_lp[k] = bonus_points[k]

    for k in tournament_lp.keys():
        if k in bonus_points.keys():
            bonus_lp[k] += tournament_lp[k]
        else:
            bonus_lp[k] = tournament_lp[k]


    season_east['league_points'] = season_east.apply(calculate_league_points, args=[bonus_lp], axis=1)
    season_west['league_points'] = season_west.apply(calculate_league_points, args=[bonus_lp], axis=1)

    season_east = sort_season_table(season_east)
    season_west = sort_season_table(season_west)

    season_east['sim_number'] = i
    season_west['sim_number'] = i


    season_east['rank'] = list(range(1, len(Teams.East) + 1))
    season_west['rank'] = list(range(1, len(Teams.West) + 1))

    all_east_results.append(season_east)
    all_west_results.append(season_west)
    all_match_results.append(all_results)

all_east_results = pd.concat(all_east_results, axis=0)
all_west_results = pd.concat(all_west_results, axis=0)

all_east_results.to_csv('results/sim_results_east.csv', index=False)
all_west_results.to_csv('results/sim_results_west.csv', index=False)


west_avg = all_west_results[['team', 'league_points', 'wins', 'losses', 'map_differential', 'rank']].groupby('team').mean().reset_index()
east_avg = all_east_results[['team', 'league_points', 'wins', 'losses', 'map_differential', 'rank']].groupby('team').mean().reset_index()

west_avg = west_avg.sort_values(by='rank')
east_avg = east_avg.sort_values(by='rank')

all_results_frame = pd.concat(all_match_results, axis=0)
print(west_avg)
print(east_avg)
