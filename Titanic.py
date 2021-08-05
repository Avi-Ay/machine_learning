
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy.random

# Set seed
numpy.random.seed(42)

ctor = DecisionTreeClassifier

tbl = pd.read_csv("data.csv")
print(tbl.info())

tbl.fillna(0, inplace=True)
tbl['Sex'] = tbl['Sex'].replace('male', 0).replace('female', 1)

# check correlation
corr_tbl = tbl.corr().iloc[:, 1].sort_values()
corr_tbl_abs = corr_tbl.abs().sort_values()

X = tbl.drop(["Survived"], axis=1, )
Y = tbl[["Survived"]]

X = pd.get_dummies(X)
Y = pd.get_dummies(Y)

# split the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y)

alg = ctor()
alg.fit(X_train, y_train)

initial_score = alg.score(X_test, y_test)
print("overall initial score:", initial_score)

columns = list(tbl.columns)
columns.remove('Survived')

columns_results = {}

for c in columns:
    X = tbl.drop(["Survived"], axis=1, )
    X.drop(c, axis=1, inplace=True)
    Y = tbl[["Survived"]]

    X_d = pd.get_dummies(X)
    Y_d = pd.get_dummies(Y)

    # split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X_d, Y_d)

    alg = ctor()
    alg.fit(X_train, y_train)

    score = alg.score(X_test, y_test)
    print("overall score after remove {col}".format(col=c), score)
    columns_results[c] = initial_score - score

# sort dictionary by values
columns_results = {k: v for k, v in sorted(columns_results.items(), key=lambda item: item[1])}
