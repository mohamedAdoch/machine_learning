import numpy as np
import matplotlib.pylab as plt

from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

class KNeigborsClassifier:
    def __init__(self, k=3):
        self.k = k
        self._dictance_function = lambda x1, x2: np.sqrt(np.sum((x1-x2)**2))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        classe_x = [self._predict(x) for x in X]
        return np.array(classe_x)

    def _predict(self, x):
        distances = np.array([self._dictance_function(x,x1) for x1 in self.X_train])

        indices_x = np.argsort(distances)[:self.k]

        labels_neigbors_x = [self.y_train[i] for i in indices_x]

        neighbors = Counter(labels_neigbors_x).most_common(1)[0][0]
        return neighbors
breast_cancer = load_breast_cancer()
X = breast_cancer['data']
y = breast_cancer['target']
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=.2, random_state=1234)
clf = KNeigborsClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

plt.figure()
plt.scatter(X[:, 0], X[:, 1],c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()
print (y_test[:10])
print(prediction[:10])




