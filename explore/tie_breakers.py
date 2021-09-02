import pandas as pd

# Pandas options for better printing
from utils.constants import Maps, Teams
import collections

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

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
    '2021/07/18',

    '2021/08/15'
]

match_results = pd.read_csv('../results/match_results.csv')
match_results = match_results[match_results['season'] == 2021]
match_results = match_results[match_results['date'].isin(playoff_dates) == False]

match_results = match_results[['team_one', 'team_two', 'team_one_map_wins', 'team_two_map_wins', 'winner', 'loser']]

manual_results = pd.read_csv('../results/manual_results.csv')

all_results = pd.concat([manual_results, match_results])

atl = all_results[(all_results['team_one'] == Teams.Reign) | (all_results['team_two'] == Teams.Reign)]
glads = all_results[(all_results['team_one'] == Teams.Gladiators) | (all_results['team_two'] == Teams.Gladiators)]

opponent = []


def determine_opponent(row, team):
    if row['team_one'] == team:
        opponent.append(
            {'team': row['team_two'], 'diff': row['team_one_map_wins'] - row['team_two_map_wins'], 'count': 1})
    else:
        opponent.append(
            {'team': row['team_one'], 'diff': row['team_two_map_wins'] - row['team_one_map_wins'], 'count': 1})


atl.apply(determine_opponent, args=[Teams.Reign], axis=1)

atl_opponent = opponent

opponent = []
glads.apply(determine_opponent, args=[Teams.Gladiators], axis=1)
glads_opponents = opponent


def group_opponents(arr):
    oppos = {}
    for item in arr:
        win = 1 if item['diff'] > 0 else 0
        if item['team'] != Teams.Reign and item['team'] != Teams.Gladiators:
            if item['team'] in oppos.keys():
                oppos[item['team']] = {'diff': item['diff'] + oppos[item['team']]['diff'],
                                       'count': item['count'] + oppos[item['team']]['count'],
                                       'wins': win + oppos[item['team']]['wins']}
            else:
                oppos[item['team']] = {'diff': item['diff'], 'count': item['count'], 'wins': win}

    return oppos


atl_opponent = group_opponents(atl_opponent)
glads_opponents = group_opponents(glads_opponents)


def order_dict(dict):
    return collections.OrderedDict(sorted(dict.items()))

atl_opponent = order_dict(atl_opponent)
glads_opponents = order_dict(glads_opponents)

print(atl_opponent)
print(glads_opponents)

# for team in atl_opponent.keys():
#     atl_res = atl_opponent[team]
#     glads_res = glads_opponents[team]
#
#     if atl_res['count'] == glads_res['count']:
#         print(team)
#         print(atl_res)
#         print(glads_res)
