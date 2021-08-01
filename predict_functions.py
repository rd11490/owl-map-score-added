import pandas as pd
import numpy as np
from utils.constants import Maps, Teams


def predict_match(team_one, team_two, map_order, rmsa, maps_to_win=3):
    # initialize each team to have 0 projected wins
    team_one_projected_wins = 0
    team_two_projected_wins = 0

    # iterate over the game mode order, determine the expected winner of the game mode, increment wins, and continue
    # until one team reaches 3 wins
    for map_type in map_order:
        team_one_attack = rmsa[map_type][team_one]['attack']
        team_one_attack_stdev = rmsa[map_type][team_one]['attack stdev']
        team_one_defend = rmsa[map_type][team_one]['defend']
        team_one_defend_stdev = rmsa[map_type][team_one]['defend stdev']

        team_two_attack = rmsa[map_type][team_two]['attack']
        team_two_attack_stdev = rmsa[map_type][team_two]['attack stdev']
        team_two_defend = rmsa[map_type][team_two]['defend']
        team_two_defend_stdev = rmsa[map_type][team_two]['defend stdev']

        team_one_attack_estimate = np.random.normal(team_one_attack, team_one_attack_stdev)
        team_one_defend_estimate = np.random.normal(team_one_defend, team_one_defend_stdev)

        team_two_attack_estimate = np.random.normal(team_two_attack, team_two_attack_stdev)
        team_two_defend_estimate = np.random.normal(team_two_defend, team_two_defend_stdev)

        # estimate the map score for each team on the map
        team_one_attack_expected = team_one_attack_estimate - team_two_defend_estimate
        team_two_attack_expected = team_two_attack_estimate - team_one_defend_estimate

        # use estimated map score to determine the map winner
        if team_one_attack_expected > team_two_attack_expected:
            team_one_projected_wins += 1
        else:
            team_two_projected_wins += 1

        # break once a team reaches 3 map wins
        if team_one_projected_wins >= maps_to_win:
            projected_winner = team_one
            loser = team_two
            break

        if team_two_projected_wins >= maps_to_win:
            projected_winner = team_two
            loser = team_one
            break
    return {
        'team_one': team_one,
        'team_one_map_wins': team_one_projected_wins,
        'team_two': team_two,
        'team_two_map_wins': team_two_projected_wins,
        'winner': projected_winner,
        'loser': loser
    }


def build_rmsa_map(rmsa_frame):
    rmsa = {
        Maps.Control: {},
        Maps.Hybrid: {},
        Maps.Escort: {},
        Maps.Assault: {}
    }
    for index in rmsa_frame.index:
        row = rmsa_frame.iloc[index]
        rmsa[row['map_type']][row['team']] = {'attack': row['rmsa attack'], 'attack stdev': row['rmsa attack stdev'],
                                              'defend': row['rmsa defend'], 'defend stdev': row['rmsa defend stdev']}
    return rmsa


def build_table(teams):
    table = {}
    for t in teams:
        table[t] = {
            'team': t,
            'wins': 0,
            'losses': 0,
            'maps_won': 0,
            'maps_lost': 0,
            'results': [],  # { 'opponent': <TEAM>, 'score': <-3 to 3>
        }
    return table


"""
columns = ['team_one', 'team_one_map_wins', 'team_two', 'team_two_map_wins', 'winner', 'loser']
"""


def update_table(row, table):
    if row['winner'] == row['team_one']:
        winner = row['team_one']
        loser = row['team_two']

        winner_table = table[winner]
        winner_table['wins'] += 1
        winner_table['maps_won'] += row['team_one_map_wins']
        winner_table['maps_lost'] += row['team_two_map_wins']
        winner_table['results'] = winner_table['results'] + [
            {'opponent': loser, 'score': row['team_one_map_wins'] - row['team_two_map_wins']}]

        loser_table = table[loser]
        loser_table['losses'] += 1
        loser_table['maps_won'] += row['team_two_map_wins']
        loser_table['maps_lost'] += row['team_one_map_wins']
        loser_table['results'] = loser_table['results'] + [
            {'opponent': winner, 'score': row['team_two_map_wins'] - row['team_one_map_wins']}]
    else:
        winner = row['team_two']
        loser = row['team_one']

        winner_table = table[winner]
        winner_table['wins'] += 1
        winner_table['maps_won'] += row['team_two_map_wins']
        winner_table['maps_lost'] += row['team_one_map_wins']
        winner_table['results'] = winner_table['results'] + [
            {'opponent': loser, 'score': row['team_two_map_wins'] - row['team_one_map_wins']}]

        loser_table = table[loser]
        loser_table['losses'] += 1
        loser_table['maps_won'] += row['team_one_map_wins']
        loser_table['maps_lost'] += row['team_two_map_wins']
        loser_table['results'] = loser_table['results'] + [
            {'opponent': winner, 'score': row['team_one_map_wins'] - row['team_two_map_wins']}]
    table[winner] = winner_table
    table[loser] = loser_table
    return table


def calculate_opponent_tie_breakers(table):
    for k in table.keys():
        team_table = table[k]
        opponent_points = 0
        opponent_differential = 0
        for result in team_table['results']:
            opponent_points += table[result['opponent']]['wins']
            opponent_differential += (table[result['opponent']]['maps_won'] - table[result['opponent']]['maps_lost'])
        team_table['opponent_points'] = opponent_points
        team_table['opponent_differential'] = opponent_differential
        team_table['map_differential'] = team_table['maps_won'] - team_table['maps_lost']
        table[k] = team_table
    return table


def sort_table(frame):
    frame = frame.sort_values(by=['wins', 'map_differential', 'opponent_points', 'opponent_differential'],
                              ascending=False)
    frame = frame[
        ['team', 'wins', 'losses', 'map_differential', 'maps_won', 'maps_lost', 'opponent_points',
         'opponent_differential']]
    return frame


"""
League Points in qualifying matches
Map differential in qualifying matches
Head-to-head record in qualifying matches
Sum of opponents' League Points in qualifying matches
Sum of opponents' map differentials in qualifying matches

columns = ['team_one', 'team_one_map_wins', 'team_two', 'team_two_map_wins', 'winner', 'loser']
"""


def calculate_tournament_table(match_results_frame):
    east_table = build_table(Teams.East)
    west_table = build_table(Teams.West)

    for ind in match_results_frame.index:
        row = match_results_frame.loc[ind, :]
        if row['team_one'] in Teams.East:
            east_table = update_table(row, east_table)
        else:
            west_table = update_table(row, west_table)

    east_table = calculate_opponent_tie_breakers(east_table)
    west_table = calculate_opponent_tie_breakers(west_table)

    east_frame = pd.DataFrame(east_table.values())
    west_frame = pd.DataFrame(west_table.values())

    return east_frame, west_frame
