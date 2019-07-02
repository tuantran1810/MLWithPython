from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
import mglearn

reviews_train = load_files("aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

reviews_test = load_files("aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

vec = CountVectorizer(max_features = 10000, max_df = 0.15)
X = vec.fit_transform(text_train)

# lda = LatentDirichletAllocation(n_components = 10, learning_method = "batch", max_iter = 25, random_state = 0, n_jobs = 6)
# document_topic = lda.fit_transform(X)

# print("lda.components_.shape: {}".format(lda.components_.shape))

# sorting = np.argsort(lda.components_, axis = 1)[:, ::-1]
# feature_names = np.array(vec.get_feature_names())

# mglearn.tools.print_topics(topics = range(10), feature_names = feature_names, sorting = sorting,
# 	topics_per_chunk = 5, n_words = 10)

lda100 = LatentDirichletAllocation(n_components = 100, learning_method = "batch", max_iter = 25, random_state = 0, n_jobs = 6)
document_topic100 = lda100.fit_transform(X)

topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis = 1)[:, ::-1]
feature_names = np.array(vec.get_feature_names())
mglearn.tools.print_topics(topics = topics, feature_names = feature_names, sorting = sorting,
	topics_per_chunk = 5, n_words = 20)

music = np.argsort(document_topic100[:, 45])[::-1]

for i in music[:10]:
	print(b".".join(text_train[i].split(b".")[:2]) + b".\n")

fig, ax = plt.subplots(1, 2, figsize = (10, 10))

topic_names = ["{:>2}".format(i) + " ".join(words) for i, words in enumerate(feature_names[sorting[:, :2]])]
for col in [0, 1]:
	start = col * 50
	end = (col + 1) * 50
	ax[col].barh(np.arange(50), np.sum(document_topic100, axis = 0)[start:end])
	ax[col].set_yticks(np.arange(50))
	ax[col].set_yticklabels(topic_names[start:end], ha = 'left', va = "top")
	ax[col].invert_yaxis()
	ax[col].set_xlim(0, 2000)
	yax = ax[col].get_yaxis()
	yax.set_tick_params(pad = 130)

plt.tight_layout()
plt.show()
