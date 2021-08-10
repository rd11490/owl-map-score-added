import pandas as pd
from collections import Counter

# Pandas options for better printing
from utils.constants import Teams

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)


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
    '2021-07-18'
]

schedule_frame = pd.read_csv('results/2021_league_schedule.csv')
print(schedule_frame)
schedule_frame = schedule_frame[schedule_frame['startDate'].isin(playoff_dates) == False]

schedule = {}
for t in Teams.East + Teams.West:
    schedule[t] = []

for i in schedule_frame.index:
    row = schedule_frame.loc[i, :]
    schedule[row['team1Name']].append(row['team2Name'])
    schedule[row['team2Name']].append(row['team1Name'])


for team in schedule.keys():
    cnt = Counter(schedule[team])
    if team in Teams.East:
        check = 2
    else:
        check = 1

    print('{}: '.format(team), [k for k, v in cnt.items() if v > check])
