import pandas as pd

# Pandas options for better printing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

stats = pd.read_csv('results/countdown_cup_predictions.csv')

total_sims = int(stats.shape[0]/4)
print(total_sims)

stats['count'] = 1
out = stats.groupby(by=['team', 'points']).count().reset_index()
out = out.sort_values(by=['team', 'points'])
out = out[['team', 'points', 'count']]

def get_or_zero(item):
    if len(item) == 0:
        return 0
    else:
        return item[0]


def convert_to_percent(cnt):
    return '{}%'.format(round(100 * cnt / total_sims, 2))

def convert_to_row(group):
    fourth = get_or_zero(group[group['points'] == 0]['count'].values)
    third = get_or_zero(group[group['points'] == 1]['count'].values)
    second = get_or_zero(group[group['points'] == 2]['count'].values)
    first = get_or_zero(group[group['points'] == 3]['count'].values)

    total_tourny = fourth + third + second + first
    missed = total_sims - total_tourny

    return pd.Series({
        'first': convert_to_percent(first),
        'second':  convert_to_percent(second),
        'third':  convert_to_percent(third),
        'fourth':  convert_to_percent(fourth),
        'missed':  convert_to_percent(missed)
    })

out = out.groupby(by='team').apply(convert_to_row).reset_index().sort_values(by='team')
print(out)