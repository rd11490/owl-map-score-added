import pandas as pd
import numpy as np
from utils.constants import Maps, Teams


def predict_matches(schedule, rmsa_for_lookup):
    results_arr = []
    for ind in schedule.index:
        match = schedule.loc[ind, :]
        match_result = predict_match(match['team1Name'], match['team2Name'], match['map_rotation'], rmsa_for_lookup, 3)
        results_arr.append(match_result)
    return results_arr


def predict_hawaii(east1, east2, west1, west2,rmsa_map, map_rotation):
    finals_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control, Maps.Hybrid, Maps.Escort]

    east1west2 = predict_match(east1, west2, map_rotation, rmsa_map, 3)
    east2west1 = predict_match(east2, west1, map_rotation, rmsa_map, 3)

    losers_round_one = predict_match(east1west2['loser'], east2west1['loser'], map_rotation, rmsa_map, 3)
    winners_finals = predict_match(east1west2['winner'], east2west1['winner'], map_rotation, rmsa_map, 3)
    losers_finals = predict_match(losers_round_one['winner'], winners_finals['loser'], map_rotation, rmsa_map, 3)
    finals = predict_match(winners_finals['winner'], losers_finals['winner'], finals_rotation, rmsa_map, 4)

    if len(set([finals['winner'], finals['loser'], losers_finals['loser'], losers_round_one['loser']])) < 4:
        print('Bad List!')
        print('first ', finals['winner'])
        print('second ', finals['loser'])
        print('third ', losers_finals['loser'])
        print('fourth ', losers_round_one['loser'])
        print('\n')

    return {
        finals['winner']: 3,
        finals['loser']: 2,
        losers_finals['loser']: 1,
        losers_round_one['loser']: 0
    }

# takes in 2 league tables, outputs dictionary of bonus points
def predict_tournament(east, west, rmsa_map):
    week_three_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control]

    east_playins = east.head(4)['team'].values
    west_playins = west.head(6)['team'].values

    east1 = predict_match(east_playins[0], east_playins[3], week_three_rotation, rmsa_map, 3)
    east2 = predict_match(east_playins[1], east_playins[2], week_three_rotation, rmsa_map, 3)

    west36 = predict_match(west_playins[2], west_playins[5], week_three_rotation, rmsa_map, 3)
    west45 = predict_match(west_playins[3], west_playins[4], week_three_rotation, rmsa_map, 3)
    west1 = predict_match(west_playins[0], west45['winner'], week_three_rotation, rmsa_map, 3)
    west2 = predict_match(west_playins[1], west36['winner'], week_three_rotation, rmsa_map, 3)

    return predict_hawaii(east1['winner'], east2['winner'], west1['winner'], west2['winner'], rmsa_map, week_three_rotation)


def predict_tournament_cycle(schedule, rmsa_map, completed_matches=None):
    results = predict_matches(schedule, rmsa_map)
    results_frame = pd.DataFrame(results)

    if completed_matches is not None:
        results_frame = pd.concat([completed_matches, results_frame], ignore_index=True)
    east, west = calculate_tournament_table(results_frame)

    east = sort_table(east)
    west = sort_table(west)

    tournament_league_points = predict_tournament(east, west, rmsa_map)

    east['rank'] = list(range(1, len(Teams.East) + 1))
    west['rank'] = list(range(1, len(Teams.West) + 1))

    return east, west, tournament_league_points, results_frame


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


def sort_season_table(frame):
    frame = frame.sort_values(by=['league_points', 'wins', 'map_differential', 'opponent_points', 'opponent_differential'],
                              ascending=False)
    frame = frame[
        ['team', 'league_points', 'wins', 'losses', 'map_differential', 'maps_won', 'maps_lost', 'opponent_points',
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


def calculate_joint_table(match_results_frame):
    table = build_table(Teams.East + Teams.West)

    for ind in match_results_frame.index:
        row = match_results_frame.loc[ind, :]
        table = update_table(row, table)


    table = calculate_opponent_tie_breakers(table)

    table = pd.DataFrame(table.values())

    return table

