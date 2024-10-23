from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import sys
results = open(os.path.basename(__file__)+'.csv', 'w')
from get_data import GetData
getData = GetData()
fields = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
accuracy = {}
features = getData.getAllFeatures()
symbols = getData.getAllSymbols()
for symbol in symbols:
    accuracy[symbol] = []
    for field in range(1, 5):
        labels = getData.getSymbolCLFLabels(symbol, field)
        X_test, X_train, y_test,  y_train = train_test_split(
            features, labels, test_size=.5)
        my_classifier = tree.DecisionTreeClassifier()
        my_classifier.fit(X_train, y_train)
        predictions = my_classifier.predict(X_test)
        accuracy[symbol].append(str(round(accuracy_score(y_test, predictions)*100, 2))+'%')
        print("[INFO] %s: %3.2f%%" %
            (symbol, accuracy_score(y_test, predictions)*100), file=sys.stderr)
    print(symbol + ', ' + ', '.join(accuracy[symbol]), file=results)    