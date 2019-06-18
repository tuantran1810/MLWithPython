import mglearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
import numpy as np
import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

citibike = mglearn.datasets.load_citibike()
print("City bike data: \n{}".format(citibike.head()))

plt.figure(figsize = (10, 3))
xticks = pd.date_range(start = citibike.index.min(), end = citibike.index.max(), freq = 'D')

plt.xticks(xticks.astype("int"), xticks.strftime("%a %m-%d"), rotation = 90, ha = "left")
plt.plot(citibike, linewidth = 1)
plt.xlabel("Date")
plt.ylabel("Rentals")

y = citibike.values
X = citibike.index.astype("int64").values.reshape(-1, 1)

n_train = 184

def eval_on_features(features, target, regressor):
	X_train, X_test = features[:n_train], features[n_train:]
	y_train, y_test = target[:n_train], target[n_train:]
	regressor.fit(X_train, y_train)
	print("test set n^2: {:.3f}".format(regressor.score(X_test, y_test)))

	y_pred = regressor.predict(X_test)
	y_pred_train = regressor.predict(X_train)

	plt.figure(figsize = (10, 3))
	plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation = 90, ha = "left")
	plt.plot(range(n_train), y_train, label = "train")
	plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label = "test")
	plt.plot(range(n_train), y_pred_train, '--', label = "prediction train")
	plt.plot(range(n_train, len(y_test) + n_train), y_pred, label = "prediction test")
	plt.legend(loc = "best")
	plt.xlabel("Date")
	plt.ylabel("Rentals")

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
eval_on_features(X, y, regressor)

X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)

X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1),
							citibike.index.hour.values.reshape(-1, 1)])
eval_on_features(X_hour_week, y, regressor)
eval_on_features(X_hour_week, y, LinearRegression())

enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
print(X_hour_week.shape)
print(enc.fit_transform(X_hour_week).toarray().shape)
eval_on_features(X_hour_week_onehot, y, Ridge())

poly_transformer = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)

lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)

hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
features = day + hour
features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

plt.figure(figsize = (15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation = 90)
plt.xlabel("Feature names")
plt.ylabel("Feature magnitude")

plt.show()