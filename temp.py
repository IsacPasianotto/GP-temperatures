#imports
import numpy as np
from sklearn.metrics import mean_squared_error as mse

# read the data from the csv file

FILE_TEST = 'y_test_4_2.npy'
FILE_PRED = 'y_pred_4_2.npy'

y_test = np.load(FILE_TEST)
y_pred = np.load(FILE_PRED)

# print(f"y_test", y_test)
print(f"y_pred", y_pred)
