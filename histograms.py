import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import style

attribute_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

n_attributes = 4
n_classes = 3

bins = np.linspace(0, 8, 64)

class_data = [
    pd.read_csv("class_1", names = attribute_names),
    pd.read_csv("class_2", names = attribute_names),
    pd.read_csv("class_3", names = attribute_names)]

for i, df in enumerate(class_data):
    df["class"] = i

fig, axis = plt.subplots(n_attributes, 1, sharey = True, tight_layout=True, figsize=(10, 6))

for i in range(n_attributes):
    for j in range(n_classes):
        axis[i].hist(class_data[j][attribute_names[i]].to_numpy(), bins=bins,histtype="barstacked", alpha=0.5, label="Class " + str(j), edgecolor="k")
    axis[i].legend(loc='upper right')
    axis[i].set_title(attribute_names[i])

plt.show()