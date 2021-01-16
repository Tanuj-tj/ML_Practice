import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print("FEATURES: \n",cancer.feature_names)
print("LABELS: \n",cancer.target_names)

x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

print(x_train,y_train)
classes = ["Malignant","Benign"]

clf_model = svm.SVC(kernel="linear")
clf_model.fit(x_train,y_train)
clf_pred = clf_model.predict(x_test)
acc_SVM = metrics.accuracy_score(y_test,clf_pred)
print("SVM Accuracy: ",acc_SVM)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train,y_train)
knn_pred = knn_model.predict(x_test)
acc_knn = metrics.accuracy_score(y_test,knn_pred)
print("KNN Accuracy: ",acc_knn)


