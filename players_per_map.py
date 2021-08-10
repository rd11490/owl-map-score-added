import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)

season4 = pd.read_csv('./player_data/phs_2021_1.csv')

season4 = season4[['esports_match_id', 'map_name', 'team_name', 'player_name']]
season4.columns = ['match_id', 'map_name', 'team', 'player_name']
season4 = season4.drop_duplicates()

def to_player_array(group):
    players = list(group['player_name'])
    players.sort()
    return pd.Series({
        'player1': players[0],
        'player2': players[1],
        'player3': players[2],
        'player4': players[3],
        'player5': players[4],
        'player6': players[5],
    })

season4 = season4.groupby(by=['match_id', 'map_name', 'team']).apply(to_player_array).reset_index()
season4.to_csv('results/players_per_map.csv', index=False)