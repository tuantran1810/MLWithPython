from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv = 5)
print(splitStr)
print("Cross validation score: {}".format(scores))
print("Average cross validation score: {:.3f}".format(scores.mean()))

res = cross_validate(logreg, iris.data, iris.target, cv = 5, return_train_score = True)
print(splitStr)
res_df = pd.DataFrame(res)
print(res_df)
print(splitStr)
print("Mean time and score:\n{}".format(res_df.mean()))
