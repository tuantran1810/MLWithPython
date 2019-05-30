import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# mglearn.plots.plot_linear_regression_wave()
# plt.show()

X, y = mglearn.datasets.make_wave(n_samples = 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_ = ", lr.coef_)
print("lr.intercept_= ", lr.intercept_)
print("training score = ", lr.score(X_train, y_train))
print("test score = ", lr.score(X_test, y_test))
