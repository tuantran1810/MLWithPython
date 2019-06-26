from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

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

vec = CountVectorizer().fit(bards_words)

print("vocabulary length = {}".format(len(vec.vocabulary_)))
print("vocabulary content:\n{}".format(vec.vocabulary_))

bag_of_words = vec.transform(bards_words)
print("bag_of_words:\n{}".format(bag_of_words))
print("bag_of_words.toarray:\n{}".format(bag_of_words.toarray()))

vec = CountVectorizer().fit(text_train)
X_train = vec.transform(text_train)
print("X_train:\n{}".format(repr(X_train)))

feature_names = vec.get_feature_names()

print("every 2000th features: {}".format(feature_names[::2000]))
