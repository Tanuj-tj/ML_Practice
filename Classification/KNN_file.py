import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv("Car Data Set/car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

X = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

#Model Building

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

#----------------ELBOW METHOD-----------------

error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plot a graph which will compare error rate and it's k value
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K value')
plt.xlabel('K')
plt.ylabel('Error Rate');
#plt.show();

#-------------------------------------------------

predicted = model.predict(x_test)
class_names = ["unacc","acc","good","vgood"]

for x in range(len(x_test)):
    print("Predicted: ",class_names[predicted[x]]," Data: ",x_test[x]," Actual: ",class_names[y_test[x]])
    model.kneighbors([x_test[x]])