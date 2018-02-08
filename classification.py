import numpy as np
import pandas as pd
import pydot

data = pd.read_csv("data.csv")
data = data.iloc[:, :-1]
y = data['diagnosis']
X = data.drop(['id', 'diagnosis'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#print(X)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
clf = classifier.fit(X_train, y_train)

from sklearn import tree
tree.export_graphviz(clf, out_file='tree.dot') 
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

from sklearn.metrics import accuracy_score
print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))
print('Validation accuracy', accuracy_score(y_test, classifier.predict(X_test)))