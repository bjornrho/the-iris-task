import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style




attribute_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]



# Reading data from files and saving in matrix 
class_data = [
    pd.read_csv("class_1", names=attribute_names), 
    pd.read_csv("class_2", names=attribute_names),
    pd.read_csv("class_3", names=attribute_names)]

# Size of variables and step size alpha
n_attributes = len(attribute_names)
n_classes = len(class_data)
alpha = 1

# Add class column defining what class this frame belongs to
for i in range(len(class_data)):
    class_data[i]["class"] = i

print(class_data)

# Convert pandas frames to numpy training and test data sets
train_x = np.concatenate([df.iloc[0:30,:-1].to_numpy() for df in class_data])
#train_x = np.concatenate([df.iloc[20:,:-1].to_numpy() for df in class_data])

train_y_labels = np.concatenate([df.iloc[0:30,-1].to_numpy() for df in class_data])
#train_y_labels = np.concatenate([df.iloc[20:,-1].to_numpy() for df in class_data])

train_y = np.zeros((train_y_labels.shape[0], n_classes))
for i, label in np.ndenumerate(train_y_labels):
    train_y[i][round(label)] = 1

test_x = np.concatenate([df.iloc[30:, :-1].to_numpy() for df in class_data])
#test_x = np.concatenate([df.iloc[0:20,:-1].to_numpy() for df in class_data])

test_y_labels = np.concatenate([df.iloc[30:,-1].to_numpy() for df in class_data])
#test_y_labels = np.concatenate([df.iloc[0:20,-1].to_numpy() for df in class_data])

test_y = np.zeros((test_y_labels.shape[0],n_classes))
for i, label in np.ndenumerate(test_y_labels):
    test_y[i][round(label)] = 1


# Initialize the total matrix containing the weight matrix and bias vectot
W = np.zeros((n_classes, n_attributes+1))

"""
def predict()

def MSE()

def train()

def run()
"""

print("shitboy")