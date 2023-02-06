import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn import metrics
# plt.rcParams['figure.figsize']=(10,6)

iris_dataset = datasets.load_iris()
x = iris_dataset.data
y = iris_dataset.target
# feature_names = iris_dataset.feature_names
# iris = pd.DataFrame(x,columns=feature_names)
# iris['Species']=y

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=0)
# model_knn = KNeighborsClassifier(n_neighbors=3)
# model_knn.fit(train_x,train_y)
# prediction_knn = model_knn.predict(test_x)

# print('The accuracy of the KNN is',model_knn.score(test_x,test_y))
#
# print('true value     ',test_y)
# print('predicted value',prediction_knn)

result = pd.Series()
for i in range(1,10):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_x,train_y)
    prediction_knn = model.predict(test_x)
    result = result.append(pd.Series(model.score(test_x, test_y)))

plt.plot(list(range(1,10)),result)
plt.show()
