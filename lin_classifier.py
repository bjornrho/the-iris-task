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


"""
def predict()

def MSE()

def train()

def run()
"""

print("shitboy")