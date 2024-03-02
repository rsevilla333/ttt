import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys


print(sys.path)

x = 1

# load database function
def load_data(file_name):
    return np.loadtxt("datasets/tictac_" + file_name + ".txt")

final_bc = load_data("final")
multi_bc = load_data("multi")
single_bc = load_data("single")