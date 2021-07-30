import numpy as np

stints = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]])
weights = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

res = np.transpose(np.matmul(np.transpose(stints),weights))
print('')
print(res)