from predict_functions import predict_match, build_rmsa_map, calculate_tournament_table, calculate_joint_table, \
    sort_table
from utils.constants import Teams, Maps
import itertools
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

teams = Teams.East + Teams.West
match_combos = list(itertools.combinations(teams, 2))

week_one_rotation = [Maps.Control, Maps.Hybrid, Maps.Escort, Maps.Assault, Maps.Control]
week_two_rotation = [Maps.Control, Maps.Escort, Maps.Assault, Maps.Hybrid, Maps.Control]
week_three_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control]

rmsa_frame = pd.read_csv('results/rmsa.csv')
rmsa_map = build_rmsa_map(rmsa_frame)

rankings = []
for i in range(0, 1000):
    results = []
    for map_order in [week_one_rotation, week_two_rotation, week_three_rotation]:
        for combo in match_combos:
            result = predict_match(combo[0], combo[1], map_order, rmsa_map, 3)
            results.append(result)

    results_frame = pd.DataFrame(results)

    table = calculate_joint_table(results_frame)
    table = sort_table(table)
    table['rank'] = list(range(1,21))
    table['sim'] = i
    rankings.append(table)

ranking_avg = pd.concat(rankings).groupby('team').mean().reset_index()
ranking_avg['relative strength'] = ranking_avg['wins']/(ranking_avg['wins'] + ranking_avg['losses'])
ranking_avg = ranking_avg.sort_values(by='relative strength', ascending=False)
ranking_avg = ranking_avg.rename(columns={'rank': 'average rank'})
ranking_avg['rank'] = list(range(1,21))
print(ranking_avg[['rank', 'team', 'relative strength', 'average rank']])
