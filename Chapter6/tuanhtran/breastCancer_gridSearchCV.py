from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)
scaler = MinMaxScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
				'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

X_test_scaled = scaler.transform(X_test)

grid = GridSearchCV(SVC(), param_grid = param_grid, cv = 5)
grid.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.3f}".format(grid.best_score_))
print("Best parameter: ", grid.best_params_)
print("Test set accuracy: {:.3f}".format(grid.score(X_test_scaled, y_test)))

