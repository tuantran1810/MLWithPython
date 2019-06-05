from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

X, y = make_circles(noise = 0.25, factor = 0.5, random_state = 1)
y_named = np.array(["blue", "red"])[y]

X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state = 0)

gbrt = GradientBoostingClassifier(random_state = 0).fit(X_train, y_train_named)
decisionFunc = gbrt.decision_function(X_test)
print(splitStr)
print("X_test.shape = {}". format(X_test.shape))
print("Decision function shape: {}".format(decisionFunc.shape))
print("Decision function: ", decisionFunc)
print("Thresholed decision function: ", decisionFunc > 0)
print("Prediction: ", gbrt.predict(X_test))
print("Training set score: ", gbrt.score(X_train, y_train_named))
print("Test set score: ", gbrt.score(X_test, y_test_named))
greaterZero = (decisionFunc > 0).astype(int)
pred = gbrt.classes_[greaterZero]
print("pred is equal to prediction: ", np.all(pred == gbrt.predict(X_test)))
print("Decision function minimum: {}, maximum: {}".format(
	np.min(decisionFunc), np.max(decisionFunc)))

fig, axes = plt.subplots(1, 2, figsize = (13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax = axes[0], alpha = 0.4,
	fill = True, cm = mglearn.cm2)
score_image = mglearn.tools.plot_2d_scores(gbrt, X, ax = axes[1], 
	alpha = 0.4, cm = mglearn.ReBl)
for ax in axes:
	mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
		markers = '^', ax = ax)
	mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
		markers = 'o', ax = ax)
	ax.set_xlabel("Feature 0")
	ax.set_ylabel("Feature 1")
cbar = plt.colorbar(score_image, ax = axes.tolist())
axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
	"Train class 1"], ncol = 4, loc = (0.1, 1.1))
plt.show()

print(splitStr)

predictProba = gbrt.predict_proba(X_test)
print("predict_proba result: \n", predictProba)
print("predict_proba shape: ", predictProba.shape)

fig, axes = plt.subplots(1, 2, figsize = (13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax = axes[0], alpha = 0.4,
	fill = True, cm = mglearn.cm2)
score_image = mglearn.tools.plot_2d_scores(gbrt, X, ax = axes[1], 
	alpha = 0.5, cm = mglearn.ReBl, function = 'predict_proba')

for ax in axes:
	mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
		markers = '^', ax = ax)
	mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
		markers = 'o', ax = ax)
	ax.set_xlabel("Feature 0")
	ax.set_ylabel("Feature 1")
cbar = plt.colorbar(score_image, ax = axes.tolist())
axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
	"Train class 1"], ncol = 4, loc = (0.1, 1.1))
plt.show()
