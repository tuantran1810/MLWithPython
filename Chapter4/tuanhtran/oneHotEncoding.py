import os
import mglearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

splitStr = "\n" + "=" * 100 + "\n"

adultPath = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(adultPath, header = None, index_col = None,
	names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
		'marital-status', 'occupation', 'relationship', 'race', 'gender',
		'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
		'income'])

data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
	'occupation', 'income']]

print(splitStr)
print(data.head())
print(splitStr)
print(data.gender.value_counts())
print(splitStr)

print("Original features: \n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies: \n", list(data_dummies.columns), "\n")

print(splitStr)
print(data_dummies.head())
print(splitStr)

features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

logreg = LogisticRegression().fit(X_train, y_train)
print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))

