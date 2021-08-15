import pandas as pd

# Pandas options for better printing
from utils.constants import Teams

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

west = pd.read_csv('../results/cc_sim_results_west.csv')

fl_fifth = west[(west['team'] == Teams.Mayhem) & (west['rank'] < 6)]
print('FL make CC and are better than 6th in CC standings: ', fl_fifth.shape[0])

sim_examples = fl_fifth['sim_number'].head(10).values

for sim in sim_examples:
    five_example = west[west['sim_number'] == sim]
    print(five_example)

#
#
# five_example = west[west['sim_number'] == fl_fifth['sim_number'].values[0]]
# print(five_example)
#
# fl_sixth = west[(west['team'] == Teams.Mayhem) & (west['rank'] == 6)]
# print('FL comes in 6th in CC standings: ', fl_sixth.shape[0])