from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import mglearn

reviews_train = load_files("aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target

print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[6]:\n{}".format(text_train[6]))

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
print("samples per trainning class: {}".format(np.bincount(y_train)))

reviews_test = load_files("aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target

print("type of text_test: {}".format(type(text_test)))
print("length of text_test: {}".format(len(text_test)))
print("text_test[6]:\n{}".format(text_test[6]))

text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
print("samples per test class: {}".format(np.bincount(y_test)))

bards_words =["The fool doth think he is wise,",
              "but the wise man knows himself to be a fool"]

vec = CountVectorizer(ngram_range = (2, 2)).fit(bards_words)

print("vocabulary length = {}".format(len(vec.vocabulary_)))
print("vocabulary content:\n{}".format(vec.vocabulary_))

bag_of_words = vec.transform(bards_words)
print("bag_of_words:\n{}".format(bag_of_words))
print("bag_of_words.toarray:\n{}".format(bag_of_words.toarray()))

vec = CountVectorizer(min_df = 5, stop_words = "english").fit(text_train)
X_train = vec.transform(text_train)
print("X_train:\n{}".format(repr(X_train)))

feature_names = vec.get_feature_names()

print("every 2000th features: {}".format(feature_names[::2000]))

#==================================================================================================================

# scores = cross_val_score(LogisticRegression(), X_train, y_train, cv = 5)
# print("Mean cross-validation accuracy: {:.3f}".format(np.mean(scores)))

# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
# grid.fit(X_train, y_train)

# print("Best cross-validation score: {:.3f}".format(grid.best_score_))
# print("Best parameters: {}".format(grid.best_params_))

# X_test = vec.transform(text_test)
# print("Test set score: {:.3f}".format(grid.score(X_test, y_test)))

#==================================================================================================================

# pipe = make_pipeline(TfidfVectorizer(min_df = 5), LogisticRegression())
# param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

# grid = GridSearchCV(pipe, param_grid, cv = 5).fit(text_train, y_train)
# print("Best cross-validation score: {:.3f}".format(grid.best_score_))

# vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# X_train = vectorizer.transform(text_train)
# max_value = X_train.max(axis = 0).toarray().ravel()
# sorted_by_tfidf = max_value.argsort()

# feature_names = np.array(vectorizer.get_feature_names())

# print("features with lowest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:20]]))
# print("features with highest tfidf:\n{}".format(feature_names[sorted_by_tfidf[-20:]]))

# sorted_by_tfidf = np.argsort(vectorizer.idf_)
# print("features with highest idf:\n{}".format(feature_names[sorted_by_tfidf[:100]]))

# mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps["logisticregression"].coef_,
# 	feature_names, n_top_features = 40)

#==================================================================================================================

pipe = make_pipeline(TfidfVectorizer(min_df = 5), LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10],
	"tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv = 5, n_jobs = 6).fit(text_train, y_train)
print("Best cross-validation score: {:.3f}".format(grid.best_score_))
print("Best cross-validation params: {}".format(grid.best_params_))

scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T

heatmap = mglearn.tools.heatmap(
	scores, xlabel = "C", ylabel = "ngram_range", cmap = "viridis", fmt = "%.3f",
	xticklabels = param_grid['logisticregression__C'],
	yticklabels = param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)

vect = grid.best_estimator_.named_steps['tfidfvectorizer']
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features = 40)

mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
mglearn.tools.visualize_coefficients(coef.ravel()[mask], feature_names[mask], n_top_features = 40)

plt.show()
