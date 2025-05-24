from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import os
import sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import random

print("Usage: python train.py <batch directory> <test dataset LLM classes 1-4>...")

batch_path = sys.argv[1]

batch_files = os.listdir(batch_path)

X = np.empty((0,10), int)
y = np.array([])
for i in range(0, len(batch_files), 2):
    batch_X = np.load(batch_path+batch_files[i], allow_pickle=True)
    batch_y = np.load(batch_path+batch_files[i+1], allow_pickle=True)
    X = np.vstack((X, batch_X))
    y = np.append(y, batch_y)

classifiers = [
    (RandomForestClassifier(n_jobs=-1), {"n_estimators": range(5, 306, 50), "criterion": ["gini", "entropy"]}),
    (LogisticRegression(n_jobs=-1), {"penalty": ["l2", None]}),
    (LinearSVC(), {"loss": ["hinge", "squared_hinge"], "penalty": ["l2", "l1"], "fit_intercept": [True]})
    ]

X_train, X_test, y_train, y_test = np.empty((0,10), int), np.empty((0,10), int), np.array([]), np.array([])
test_classes = [int(x) for x in sys.argv[2:]]

for X_item, y_item in zip(X, y):
    if (y_item in test_classes) or ((not y_item) and random.random() < 0.5):
        X_test = np.vstack((X_test, X_item)) 
        y_test = np.append(y_test, y_item)
    else:
        X_train = np.vstack((X_train, X_item))
        y_train = np.append(y_train, y_item)
print("Train:",np.unique(y_train, return_counts=True))
print("Test:",np.unique(y_test, return_counts=True))
y_train = np.clip(y_train, a_min=0, a_max=1) # нам интересно, что это бот, и размерности его конкретного типа,
y_test = np.clip(y_test, a_min=0, a_max=1) # но не интересно, какой это тип бота
alphas = np.linspace(0.001, 2, 10)
best_classifiers = []
for classifier, hyperparameters in classifiers:
    print("Grid searching for %s..." % classifier.__class__.__name__)
    by_alpha = [-1, None, -1]
    for alpha_index in range(0, 10):
        gs = GridSearchCV(classifier, hyperparameters, scoring="balanced_accuracy", n_jobs=-1)
        gs.fit(X_train[:, alpha_index:alpha_index+1], y_train)
        y_pred = gs.best_estimator_.predict(X_test[:, alpha_index:alpha_index+1])
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        if balanced_accuracy > by_alpha[0]:
            by_alpha[0] = balanced_accuracy
            by_alpha[1] = gs.best_estimator_
            by_alpha[2] = alpha_index
    best_classifiers.append(by_alpha[1])
    y_pred = by_alpha[1].predict(X_test[:, by_alpha[2]:by_alpha[2]+1])
    print("Balanced accuracy: %.4f, alpha %.6f"%(balanced_accuracy_score(y_test, y_pred), alphas[by_alpha[2]]))
    save_filename = "separated_test_%s_%s_balanced_accuracy_%.4f_alpha_%.4f.jl"\
    % ("_".join([str(x) for x in test_classes]), classifier.__class__.__name__, by_alpha[0], alphas[by_alpha[2]])
    print(save_filename)
    joblib.dump(by_alpha[1], open(save_filename, "wb"))
print("Done")