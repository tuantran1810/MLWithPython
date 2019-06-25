from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.decomposition import PCA
import numpy as np

pipe_short = make_pipeline(MinMaxScaler(), SVC(C = 100))
print("pipeline steps: \n", pipe_short.steps)

pipe = make_pipeline(StandardScaler(), PCA(n_components = 2), StandardScaler())
print("pipeline steps: \n", pipe.steps)

