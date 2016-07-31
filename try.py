from uci_loader import *
from randombitsforest import RandomBitsForest
X, y = getdataset('diabetes')

from sklearn.ensemble.forest import RandomForestClassifier

classifier = RandomBitsForest()
classifier.fit(X[:len(y)/2], y[:len(y)/2])
p = classifier.predict(X[len(y)/2:])
print "Random Bits Forest Accuracy:", np.mean(p == y[len(y)/2:])

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X[:len(y)/2], y[:len(y)/2])
print "Random Forest Accuracy:", np.mean(classifier.predict(X[len(y)/2:]) == y[len(y)/2:])