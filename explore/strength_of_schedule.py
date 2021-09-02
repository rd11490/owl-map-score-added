import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn import metrics

from utils.constants import Maps, Teams

# Pandas options for better printing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

# Read in our scored maps generated in the map_score script
map_scores_df = pd.read_csv('../results/scored_maps.csv')

# Limit our results to 2021 maps
map_scores_df = map_scores_df[map_scores_df['season'] == 2021]

# Duplicate and mirror our map scores so that we can get a row for each team on attack and defense on each map
map_scores_swapped_df = map_scores_df.copy(deep=True)
map_scores_swapped_df['team_one_score'] = map_scores_df['team_two_score']
map_scores_swapped_df['team_two_score'] = map_scores_df['team_one_score']
map_scores_swapped_df['team_one_name'] = map_scores_df['team_two_name']
map_scores_swapped_df['team_two_name'] = map_scores_df['team_one_name']
map_scores_df = pd.concat([map_scores_swapped_df, map_scores_df])
map_scores_df = map_scores_df.dropna()

# Generate a list of teams
teams = list(set(list(map_scores_df['team_one_name'].values) + list(map_scores_df['team_two_name'].values)))
teams = [str(p) for p in teams]
teams = sorted(teams)


# Convert an input row into a row of our sparse matrix
def map_teams(row_in, teams):
    t1 = row_in[0]
    t2 = row_in[1]

    row_out = np.zeros([len(teams) * 2])

    row_out[teams.index(t1)] = 1
    row_out[teams.index(t2) + len(teams)] = -1
    return row_out


# Take in our input data, convert the teams into our sparse design matrix and the map scores into our target column
def extract_X_Y(frame):
    stints_x_base = frame[['team_one_name', 'team_two_name']].values

    stint_X_rows = np.apply_along_axis(map_teams, 1, stints_x_base, teams)

    stint_Y_rows = frame[['team_one_score']].values
    return stint_X_rows, stint_Y_rows


# Convert lambda value to alpha needed for ridge CV
def lambda_to_alpha(lambda_value, samples):
    return (lambda_value * samples) / 2.0


# Convert RidgeCV alpha back into a lambda value
def alpha_to_lambda(alpha_value, samples):
    return (alpha_value * 2.0) / samples


# Calculate Regularized Map Type Score
def calculate_rmts(stint_X_rows, stint_Y_rows, map_type):
    # We will perform cross validation across a number of different lambdas
    lambdas = [.01, 0.025, .05, 0.075, .1, .125, .15, .175, .2, .225, .25]

    # convert the lambdas into alpha values
    alphas = [lambda_to_alpha(l, stint_X_rows.shape[0]) for l in lambdas]

    # Create our ridge CV model
    clf = RidgeCV(alphas=alphas, cv=5, fit_intercept=True, normalize=False)

    # Fit our data
    model = clf.fit(stint_X_rows, stint_Y_rows)

    # extract our teams, and coefficients and combine them into a single matrix (20 x 3)
    team_arr = np.transpose(np.array(teams).reshape(1, len(teams)))
    coef_array_attack = np.transpose(model.coef_[:, 0:len(teams)])
    coef_array_def = np.transpose(model.coef_[:, len(teams):])
    team_coef_arr = np.concatenate([team_arr, coef_array_attack, coef_array_def], axis=1)

    # build a dataframe from our matrix
    rmts = pd.DataFrame(team_coef_arr)
    intercept = model.intercept_[0]

    # Rename columns to include the current map type
    attack_str = 'rmsa attack'
    defend_str = 'rmsa defend'

    rmts.columns = ['team', attack_str, defend_str]
    rmts[attack_str] = rmts[attack_str].astype(float)
    rmts[defend_str] = rmts[defend_str].astype(float)

    # Calculate a total RMSA
    rmts['rmsa'] = rmts[attack_str] + rmts[defend_str]
    rmts['intercept'] = intercept
    rmts['map_type'] = map_type

    # Generate a couple of error statistics
    lambda_picked = alpha_to_lambda(model.alpha_, stint_X_rows.shape[0])
    rmts = rmts.sort_values(by='rmsa', ascending=False)

    ## Closed Form
    # B = (XtX + lambdaI)^-1 * XtY
    intercept_row = np.ones((stint_X_rows.shape[0], 1))
    close_form_stint_x = np.concatenate([stint_X_rows, intercept_row], axis=1)
    xtx = np.matmul(np.transpose(close_form_stint_x), close_form_stint_x) + lambda_to_alpha(lambda_picked,stint_X_rows.shape[0])  * np.eye(
        close_form_stint_x.shape[1])
    xtx_inv = np.linalg.inv(xtx)
    xty = np.matmul(np.transpose(close_form_stint_x), stint_Y_rows)
    betas = np.matmul(xtx_inv, xty)

    pred = np.matmul(close_form_stint_x, betas)
    resid = stint_Y_rows - pred


    scalar = 1 / (close_form_stint_x.shape[0] - 40 - 1)

    errs = np.matmul(np.transpose(resid), resid)
    var = errs * xtx_inv * scalar

    var_map = {}
    for ind, team in enumerate(team_arr):

        attack_var = var[ind][ind]
        defend_var = var[ind + len(team_arr)][ind + len(team_arr)]
        var_map[team[0]] = {'team': team[0], 'attack_variance': attack_var,
                         'defend_variance': defend_var}
    variance = pd.DataFrame(var_map.values())

    rmts = rmts.merge(variance, on='team')
    rmts['rmsa attack stdev'] = np.sqrt(rmts['attack_variance'])
    rmts['rmsa defend stdev'] = np.sqrt(rmts['defend_variance'])
    rmts = rmts[['team', 'rmsa attack', 'rmsa attack stdev', 'rmsa defend', 'rmsa defend stdev', 'rmsa', 'map_type']]
    rmts['normalized rmsa'] = 100 * ((2 * (rmts['rmsa'] - rmts['rmsa'].min())/(rmts['rmsa'].max()-rmts['rmsa'].min())) - 1)
    return rmts


def calculate_total_rmsa(group):
    total = group['normalized rmsa'].sum() + group[group['map_type'] == Maps.Control]['normalized rmsa'].sum()
    return pd.Series({'rmsa': total/5})

def calculate_rmsa(team, map_scores):
    map_scores = map_scores[(map_scores['team_one_name'] != team) & (map_scores['team_two_name'] != team)]
    # Generate RMSA for control maps
    control = map_scores[map_scores['map_type'] == Maps.Control]
    control_X, control_Y = extract_X_Y(control)
    control_rmts = calculate_rmts(control_X, control_Y, Maps.Control)

    # Generate RMSA for Escort maps
    escort = map_scores[map_scores['map_type'] == Maps.Escort]
    escort_X, escort_Y = extract_X_Y(escort)
    escort_rmts = calculate_rmts(escort_X, escort_Y, Maps.Escort)

    # Generate RMSA for Hybrid maps
    hybrid = map_scores[map_scores['map_type'] == Maps.Hybrid]
    hybrid_X, hybrid_Y = extract_X_Y(hybrid)
    hybrid_rmts = calculate_rmts(hybrid_X, hybrid_Y, Maps.Hybrid)

    # Generate RMSA for Assault maps
    assault = map_scores[map_scores['map_type'] == Maps.Assault]
    assault_X, assault_Y = extract_X_Y(assault)
    assault_rmts = calculate_rmts(assault_X, assault_Y, Maps.Assault)

    all_rmsa = pd.concat([control_rmts, escort_rmts, hybrid_rmts, assault_rmts])

    all_rmsa['team dropped'] = team

    total_rmsa = all_rmsa[['team', 'map_type', 'normalized rmsa']].groupby(by='team').apply(
        calculate_total_rmsa).reset_index().sort_values(by='rmsa', ascending=False)

    total_rmsa['team dropped'] = team
    total_rmsa = round(total_rmsa, 2)

    return total_rmsa


total_rmsa_team_dropped = []
for team in Teams.East + Teams.West:
    total_rmsa = calculate_rmsa(team, map_scores_df)
    total_rmsa_team_dropped.append(total_rmsa)


rmsa_frame = pd.concat(total_rmsa_team_dropped)

playoff_dates = [
    '2021-05-02',
    '2021-05-06',
    '2021-05-07',
    '2021-05-08',
    '2021-05-09',

    '2021-06-06',
    '2021-06-10',
    '2021-06-11',
    '2021-06-12',
    '2021-06-13',

    '2021-07-11',
    '2021-07-12',
    '2021-07-15',
    '2021-07-16',
    '2021-07-17',
    '2021-07-18',
    '2021-08-16',
    '2021-08-15'
]

schedule_frame = pd.read_csv('../results/2021_league_schedule.csv')
schedule_frame = schedule_frame[schedule_frame['startDate'].isin(playoff_dates) == False]

schedule = {}
for t in Teams.East + Teams.West:
    schedule[t] = []

for i in schedule_frame.index:
    row = schedule_frame.loc[i, :]
    schedule[row['team1Name']].append(row['team2Name'])
    schedule[row['team2Name']].append(row['team1Name'])


sos = []
for t in Teams.East + Teams.West:
    rmsa_team_dropped = rmsa_frame[rmsa_frame['team dropped'] == t]
    schedule_arr = schedule[t]
    score = 0
    for oppo in schedule_arr:
        score =+ rmsa_team_dropped[rmsa_team_dropped['team'] == oppo]['rmsa'].values[0]
    sos.append({'team': t, 'sos': score})
sos_df = pd.DataFrame(sos)
sos_df = sos_df.sort_values(by='sos', ascending=False)
sos_df.to_csv('sos/sos.csv', index=False)
rmsa_frame.to_csv('sos/rmsa_sos.csv', index=False)

print(sos_df)

