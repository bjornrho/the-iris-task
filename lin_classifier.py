import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style




attribute_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
dropped_attributes = []


# Reading data from files and saving in matrix 
class_data = [
    pd.read_csv("class_1", names=attribute_names), 
    pd.read_csv("class_2", names=attribute_names),
    pd.read_csv("class_3", names=attribute_names)]

# Size of variables, step size alpha and number of iterations to run
n_attributes = len(attribute_names)
n_classes = len(class_data)
alpha = 0.0035
iterations = 500

# Add class column defining what class this frame belongs to and dropping attributes
for i in range(len(class_data)):
    class_data[i]["class"] = i
    class_data[i] = class_data[i].drop(columns=dropped_attributes)

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


# Initialize the total matrix containing the weight matrix and bias vector
W = np.zeros((n_classes, n_attributes))
b = np.zeros((n_classes,))


# Takes input data, weight and biases
# Returns predicted class labels
# Se Sidmoid
def predict(x, W, b):
    n = x.shape[0]
    prediction = np.array(
        [1.0/(1.0+np.exp(-(np.matmul(W,x[i])+b))) for i in range(n)])
    return prediction


# Se side 77, (3.19)
# Returns mean-square error of our prediction
def MSE(pred, y):
    # (1/N)*sum(error^2)
    return np.sum(np.matmul(np.transpose(pred-y),(pred-y))) / pred.shape[0]


# Se (3.22)
# Performs one training iteration
def train(pred, y, x, W, b, alpha):
    pred_error = pred - y
    pred_z_error = np.multiply(pred,(1-pred))
    squarebracket = np.multiply(pred_error, pred_z_error)

    dW = np.zeros(W.shape)
    # Gradient of MSE with respect to W
    for i in range (x.shape[0]):
        dW = np.add(dW, np.outer(squarebracket[i], x[i]))
    
    dB = np.sum(squarebracket, axis=0)

    return ((W-alpha*dW), (b-alpha*dB))

# Setup for plotting
style.use('fivethirtyeight')

figure = plt.figure()
axis = figure.add_subplot(1,1,1)

# Arrays for storing plot values
plot_iteration = []
train_losses = []
test_losses = []


# Plot of one iteration of predictioning and training
def run(i):
    global W
    global b

    train_prediction = predict(train_x, W,b)
    test_prediction = predict(test_x, W, b)

    train_loss = MSE(train_prediction, train_y)
    test_loss = MSE(test_prediction, test_y)

    plot_iteration.append(float(i))
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    axis.clear()
    axis.plot(plot_iteration, train_losses, "blue")
    axis.plot(plot_iteration, test_losses, "red")

    W, b = train(train_prediction, train_y, train_x, W, b, alpha)

# Generate confusion matrix from predictions
def generate_confusion_matrix(x, y, W, b):
    pred = predict(x, W, b)

    confusion_matrix = np.zeros((n_classes, n_classes))

    for i in range(pred.shape[0]):
        confusion_matrix[np.argmax(y[i])][np.argmax(pred[i])] += 1

    return confusion_matrix


# Returns error rate based on predictions
def get_error_rate(x, y, W, b):
    pred = predict(x, W, b)

    mistakes = 0

    for i in range(pred.shape[0]):
        if np.argmax(y[i]) != np.argmax(pred[i]):
            mistakes += 1

    return mistakes/pred.shape[0]

print("Initial W: ")
print(W)

print("Initial b: ")
print(b)

animate = animation.FuncAnimation(figure, run, interval=16, frames=iterations, repeat = False)
plt.show()

print("Final W: ")
print(W)

print("Final b: ")
print(b)

print("Training error rate: ")
print(get_error_rate(train_x, train_y, W, b))

print("Training confusion matrix: ")
print(generate_confusion_matrix(train_x, train_y, W, b))

print("Testing error rate: ")
print(get_error_rate(test_x, test_y, W, b))

print("Testing confusion matrix")
print(generate_confusion_matrix(test_x, test_y, W, b))

