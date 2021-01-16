import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())
predict = "G3"

X = np.array(data.drop([predict],axis=1))  # Features
y = np.array(data[predict])  # Labels

# Train Test Split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

best = 0
for _ in range(30):

    # Train Test Split
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Linear Regression model

    linear = linear_model.LinearRegression()

    # Train our model

    linear.fit(x_train,y_train)

    # Evaluate the model with the help of testing data

    acc = linear.score(x_test,y_test)

    print(acc)

    if acc > best:
        best = acc
        # Create a pickle file to save our model

        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# Load the pickle file

pickle_in = open("studentmodel.pickle","rb")
liner = pickle.load(pickle_in)

# Viewing the constants

print("Coefficient : \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# Predict the data on specific student

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])

p = 'G2'
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()