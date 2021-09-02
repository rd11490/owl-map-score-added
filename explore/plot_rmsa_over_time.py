import pandas as pd
import matplotlib.pyplot as plt
from utils.constants import Teams

# Pandas options for better printing

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

rmsa_by_week = pd.read_csv('results/total_rmsa.csv')
print(rmsa_by_week)
weeks = sorted(list(rmsa_by_week['week'].unique()))

fig = plt.figure(figsize=(16, 8))
plt.title('Map Score Added By Week - West')
plt.xlabel('Week')
plt.ylabel('Map Score Added')
plt.ylim((-100, 100))
plt.xticks(weeks)
plt.xlim((0, 23))


for t in Teams.West:
    color = Teams.TeamColors[t]
    team_rmsa = rmsa_by_week[rmsa_by_week['team'] == t]
    plt.plot(team_rmsa['week'], team_rmsa['rmsa'], color=color, label=t)
    height = team_rmsa['rmsa'].values[-1] + 0.005
    if t == Teams.Mayhem:
        height = height - 5
    plt.text(team_rmsa['week'].values[-1] + 0.005, height, t, color=color, weight='bold')
plt.legend(loc='lower center', ncol=6)
plt.show()

fig = plt.figure(figsize=(16, 8))
plt.title('Map Score Added By Week - East')
plt.xlabel('Week')
plt.ylabel('Map Score Added')
plt.ylim((-100, 100))
plt.xticks(weeks)
plt.xlim((0, 23))

for t in Teams.East:
    color = Teams.TeamColors[t]
    team_rmsa = rmsa_by_week[rmsa_by_week['team'] == t]
    plt.plot(team_rmsa['week'], team_rmsa['rmsa'], color=color, label=t)
    height = team_rmsa['rmsa'].values[-1] + 0.005
    plt.text(team_rmsa['week'].values[-1] + 0.005, height, t, color=color, weight='bold')

plt.legend(loc='upper center', ncol=6)
plt.show()
