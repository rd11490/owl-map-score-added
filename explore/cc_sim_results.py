import pandas as pd

# Pandas options for better printing
from utils.constants import Teams

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

final_tables_west = pd.read_csv('../results/cc_sim_results_west.csv')
final_tables_east = pd.read_csv('../results/cc_sim_results_east.csv')

west_ranks = []
east_ranks = []

def rank_teams(frame):
    sims = frame.shape[0]
    rank_hist = frame['rank'].value_counts().sort_index()

    return 100 * rank_hist / sims

for team in Teams.West:
    team_results = final_tables_west[final_tables_west['team'] == team][['wins', 'rank']]
    ranks = rank_teams(team_results)
    ranks['team'] = team
    west_ranks.append(ranks)

for team in Teams.East:
    team_results = final_tables_east[final_tables_east['team'] == team][['wins', 'rank']]
    ranks = rank_teams(team_results)
    ranks['team'] = team
    east_ranks.append(ranks)

west_ranks = pd.DataFrame(west_ranks)
east_ranks = pd.DataFrame(east_ranks)

west_ranks = west_ranks.fillna(0.0)
west_columns = ['team', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
west_ranks = west_ranks[west_columns]
west_ranks = west_ranks.sort_values(by=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ascending=False)
print(west_ranks)

east_ranks = east_ranks.fillna(0.0)
east_columns = ['team', 1, 2, 3, 4, 5, 6, 7, 8]
east_ranks = east_ranks[east_columns]
east_ranks = east_ranks.sort_values(by=[1, 2, 3, 4, 5, 6, 7, 8], ascending=False)

print(east_ranks)