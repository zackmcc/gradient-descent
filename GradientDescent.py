###
#   Zackary McClamma
#   CPS 580 Final Project: Gradient Descent
#   06 AUG 2020
###

import numpy as np
import time
import copy
from tkinter import Tk, filedialog

# Function to read and parse input data
def readDataFromFile (filename):
    gender = []
    height = []
    weight = []
    i = 0
    for line in open(filename, 'r'):
        current_line = line.strip().split(',')

        if current_line[2].replace('.', '', 1).isdigit():
            gender.append(current_line[0])
            height.append(float(current_line[1]))
            weight.append(float(current_line[2]))

    return [gender, height, weight]


# Sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# This function runs the hypothesis function (theta0*x0 + theta1*x1 + theta2*x2) for all values of x and returns an array
# of the results for each set of input values
def hypothesis(x, theta):
    return np.dot(theta, x)


def gradientDescent(x, theta, y, convergence):
    m = len(y)
    converged = 0
    iterations = 0
    while not converged:
        temp = copy.deepcopy(theta)
        z = hypothesis(x, theta)
        h = sigmoid(z)
        iterations += 1
        theta[0] -= np.sum((h - y) * x[0]) / m
        theta[1] -= np.sum((h - y) * x[1]) / m
        theta[2] -= np.sum((h - y) * x[2]) / m
        if abs(temp[0] - theta[0]) <= convergence and abs(temp[1] - theta[1]) <= convergence \
                                 and abs(temp[2] - theta[2]) <= convergence:
            converged = 1
    print("Number of iterations to convergence: " + str(iterations))
    return theta

# Prediction function used to test the model
def predict(x, theta):
    prediction = []
    z = hypothesis(x, theta)
    h = sigmoid(z)

    for i in range(len(h)):
        if h[i] > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    return np.array(prediction)


# Opens a pop up window to select input file
def returnFileName():
    root = Tk()
    filename = filedialog.askopenfilename()
    root.destroy()

    return filename

def main():
    print('Entering Main method')
    done = 0
    trained = 0
    while not done:
        options = input("Enter 1 to train the model or 2 to test the model (after selecting an option a pop up window\n"
                        "will appear for you to select an input file: ")
        input_file = returnFileName()

        [gender, height, weight] = readDataFromFile(input_file)
        y = np.array(gender)

        # Replace string values with 1, and 0
        y[y == '"Male"'] = 1
        y[y == '"Female"'] = 0
        y = y.astype(np.int)

        m = len(y)
        x = np.array([np.ones(m), height, weight])

        # Normalize input data
        # I noticed that when I don't normalize the input data that the values I pass into the sigmoid function are
        # large negative values so the sigmoid always yields a value of 1
        x = (x - x.min()) / (x.max() - x.min())
        x[0] = np.ones(m)

        if int(options) == 1:
            convergence_val = float(input("Enter the convergence value for the theta calculation (this value should be a "
                                    "floating point value less than 1 and close to zero because the input data\n"
                                    " has been normalized NOTE: If the value is too small "
                                    "the model will take a long time to run (smallest tested value=.0001): "))

            # verify convergence input
            while convergence_val < 0 or convergence_val > 1:
                convergence_val = float(input("Invalid convergence value please try a new value: "))
            # If the model has not been previously trained generate random theta values
            if not trained:
                theta = np.random.uniform(-1/np.sqrt(3), (1/np.sqrt(3)), 3)
            start = time.time()
            theta = gradientDescent(x, theta, y, convergence_val)
            end = time.time()
            print("Gradient Descent took " + str((end - start)*1000) + "ms")
            print("Resulting theta values: theta0= " + str(theta[0]) + " theta1= " + str(theta[1]) + " theta2= "
                  + str(theta[2]) + "\n")
            trained = 1

        elif int(options) == 2:
            # The predictor can only be run if the model has been trained
            if trained:
                prediction = predict(x, theta)
                accuracy = [prediction == y]
                accuracy = np.sum(accuracy)/m
                print("Model predicted testing values with " + str(accuracy*100) + "% accuracy\n")
            else:
                print("ERROR: Cannot run predictor without first training the model, please try again")
        done = int(input("If you would like to continue enter 0, if you want to exit the program enter a 1: "))


if __name__ == '__main__':
    main()
