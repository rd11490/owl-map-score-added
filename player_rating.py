import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn import metrics

from utils.constants import Maps

# Pandas options for better printing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

# Read in our scored maps generated in the map_score script
map_scores = pd.read_csv('results/scored_maps.csv')

# Limit our results to 2021 maps
map_scores = map_scores[map_scores['season'] == 2021]

# Duplicate and mirror our map scores so that we can get a row for each team on attack and defense on each map
map_scores_swapped = map_scores.copy(deep=True)
map_scores_swapped['team_one_score'] = map_scores['team_two_score']
map_scores_swapped['team_two_score'] = map_scores['team_one_score']
map_scores_swapped['team_one_name'] = map_scores['team_two_name']
map_scores_swapped['team_two_name'] = map_scores['team_one_name']
map_scores = pd.concat([map_scores_swapped, map_scores])
map_scores = map_scores.dropna()

players_per_map = pd.read_csv('results/players_per_map.csv')

full_frame = map_scores \
    .merge(players_per_map, left_on=['match_id', 'map_name', 'team_one_name'],
           right_on=['match_id', 'map_name', 'team']) \
    .merge(players_per_map, left_on=['match_id', 'map_name', 'team_two_name'],
           right_on=['match_id', 'map_name', 'team'], suffixes=('_team_one', '_team_two'))

full_frame = full_frame[
    ['team_one_score', 'player1_team_one', 'player2_team_one', 'player3_team_one', 'player4_team_one',
     'player5_team_one', 'player6_team_one', 'player1_team_two', 'player2_team_two', 'player3_team_two',
     'player4_team_two', 'player5_team_two', 'player6_team_two']]

players = list(full_frame[
                   ['player1_team_one', 'player2_team_one', 'player3_team_one', 'player4_team_one', 'player5_team_one',
                    'player6_team_one', 'player1_team_two', 'player2_team_two', 'player3_team_two', 'player4_team_two',
                    'player5_team_two', 'player6_team_two']].stack().unique())
players.sort()


# Convert an input row into a row of our sparse matrix
def map_players(row_in, players):
    t1_players = [row_in[0], row_in[1], row_in[2], row_in[3], row_in[4], row_in[5]]
    t2_players = [row_in[6], row_in[7], row_in[8], row_in[9], row_in[10], row_in[11]]

    row_out = np.zeros([len(players) * 2])

    for p in t1_players:
        row_out[players.index(p)] = 1
    for p in t2_players:
        row_out[players.index(p) + len(players)] = -1
    return row_out

# Take in our input data, convert the teams into our sparse design matrix and the map scores into our target column
def extract_X_Y(frame):
    stints_x_base = frame[['player1_team_one', 'player2_team_one', 'player3_team_one', 'player4_team_one', 'player5_team_one',
                    'player6_team_one', 'player1_team_two', 'player2_team_two', 'player3_team_two', 'player4_team_two',
                    'player5_team_two', 'player6_team_two']].values

    stint_X_rows = np.apply_along_axis(map_players, 1, stints_x_base, players)

    stint_Y_rows = frame[['team_one_score']].values
    return stint_X_rows, stint_Y_rows

# Convert lambda value to alpha needed for ridge CV
def lambda_to_alpha(lambda_value, samples):
    return (lambda_value * samples) / 2.0


# Convert RidgeCV alpha back into a lambda value
def alpha_to_lambda(alpha_value, samples):
    return (alpha_value * 2.0) / samples


# Calculate Regularized Map Score Added
def calculate_rmts(stint_X_rows, stint_Y_rows):
    # We will perform cross validation across a number of different lambdas
    lambdas = [.01, 0.025, .05, 0.075, .1, .125, .15, .175, .2, .225, .25]

    # convert the lambdas into alpha values
    alphas = [lambda_to_alpha(l, stint_X_rows.shape[0]) for l in lambdas]

    # Create our ridge CV model
    clf = RidgeCV(alphas=alphas, cv=5, fit_intercept=True, normalize=False)

    # Fit our data
    model = clf.fit(stint_X_rows, stint_Y_rows)

    # extract our teams, and coefficients and combine them into a single matrix (20 x 3)
    team_arr = np.transpose(np.array(players).reshape(1, len(players)))
    coef_array_attack = np.transpose(model.coef_[:, 0:len(players)])
    coef_array_def = np.transpose(model.coef_[:, len(players):])
    team_coef_arr = np.concatenate([team_arr, coef_array_attack, coef_array_def], axis=1)

    # build a dataframe from our matrix
    rmts = pd.DataFrame(team_coef_arr)
    intercept = model.intercept_[0]

    # Rename columns to include the current map type
    attack_str = 'rmsa attack'
    defend_str = 'rmsa defend'

    rmts.columns = ['player', attack_str, defend_str]
    rmts[attack_str] = rmts[attack_str].astype(float)
    rmts[defend_str] = rmts[defend_str].astype(float)

    # Calculate a total RMSA
    rmts['rmsa'] = rmts[attack_str] + rmts[defend_str]
    rmts['intercept'] = intercept

    # Generate a couple of error statistics
    lambda_picked = alpha_to_lambda(model.alpha_, stint_X_rows.shape[0])
    print('r^2: ', model.score(stint_X_rows, stint_Y_rows))
    print('lambda: ', lambda_picked)
    print('intercept: ', intercept)

    pred = model.predict(stint_X_rows)
    print('MAE: ', metrics.mean_absolute_error(stint_Y_rows, pred))
    print('MSE: ', metrics.mean_squared_error(stint_Y_rows, pred))
    rmts = rmts.sort_values(by='rmsa', ascending=False)
    return rmts

x, y = extract_X_Y(full_frame)
rmsa = calculate_rmts(x, y)
rmsa['rank'] = rmsa['rmsa'].rank(ascending=False)

rmsa = rmsa[['rank', 'player', 'rmsa', 'rmsa attack', 'rmsa defend']]
print(rmsa)

rmsa.to_csv('results/player_rating.csv', index=False)

