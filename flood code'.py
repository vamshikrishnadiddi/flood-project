
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('finaldata.csv')
X = dataset.iloc[:, 2:8].values
y = dataset.iloc[:, 8].values
vamshi=22
vamshi=vamshi*22
lplp
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 20)
sns.pairplot(dataset,palette='coolwarm')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
k=cm[0][0]+cm[1][1]
l=(cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1])
(k/l)*100


from sklearn.metrics import r2_score
coefficient_of_determination = r2_score(y_test, y_pred)
coefficient_of_determination
