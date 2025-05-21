import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

state = 0
np.random.seed(state)

# data preparation
data = pd.read_csv('./data_feature.csv')
y = data['similarity']
x = data[
    ['CHAM830107', 'RADA880108', 'CIDH920101', 'CHOC760102', 'COHE430101', 'N-Glycosylation', 'N2_A', 'N2_B', 'N2_C',
     'N2_D', 'N2_E', 'Distance']]
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3, random_state=0)

standard = StandardScaler()
standard.fit(X_train)
joblib.dump(standard, './model/scale.pkl')
X_train = pd.DataFrame(standard.transform(X_train), columns=list(X_train))
X_test = pd.DataFrame(standard.transform(X_test), columns=list(X_train))

dirt = 'model/'
if not os.path.exists(dirt):
    os.makedirs(dirt)

# random search
param_grid_lg = {
    'C': list(np.random.uniform(0.1, 2.2, size=1000)),
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],
    'random_state': [state],
    'max_iter': [5000]
}

lg = LogisticRegression()
random_search_lg = RandomizedSearchCV(lg, param_grid_lg, n_iter=500, random_state=state, cv=5, n_jobs=25,
                                      return_train_score=True)
random_search_lg.fit(X_train, y_train)
best_lg = random_search_lg.best_estimator_
lg_params = random_search_lg.best_params_
print(random_search_lg.best_params_)

param_grid_svm = {
    'C': list(np.random.uniform(0.1, 10, size=1000)),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'random_state': [state],
}

svm = SVC()
random_search_svm = RandomizedSearchCV(svm, param_grid_svm, n_iter=500, random_state=state, cv=5, n_jobs=25,
                                       return_train_score=True)
random_search_svm.fit(np.array(X_train), y_train)
best_svm = random_search_svm.best_estimator_
svm_params = random_search_svm.best_params_
print(random_search_svm.best_params_)

param_grid_rf = {
    'n_estimators': list(np.random.randint(50, 200, size=200)),
    'max_depth': list(np.random.randint(2, 15, size=100)),
    'min_samples_split': list(np.random.randint(2, 20, size=100)),
    'min_samples_leaf': list(np.random.randint(2, 10, size=100)),
    'criterion': ['gini', 'entropy', 'log_loss'],
    'random_state': [state]
}
rf = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(rf, param_grid_rf, n_iter=500, random_state=state, cv=5, n_jobs=25,
                                      return_train_score=True)
random_search_rf.fit(X_train, y_train)
best_rf = random_search_rf.best_estimator_
test_score = best_rf.score(X_test, y_test)
cross_val_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
rf_params = random_search_rf.best_params_
print(random_search_rf.best_params_)

np.random.seed(state)
param_grid_xgb = {
    'n_estimators': list(np.random.randint(50, 200, size=200)),
    'max_depth': list(np.random.randint(1, 15, size=100)),
    'min_child_weight': list(np.random.randint(1, 10, size=100)),
    'learning_rate': list(np.random.uniform(0.001, 0.30, size=100)),
    'gamma': list(np.random.uniform(0, 1, size=100)),
    'reg_lambda': list(np.random.uniform(0.01, 0.5, size=100)),
    'random_state': [state],
}
xgb = XGBClassifier()
random_search_xgb = RandomizedSearchCV(xgb, param_grid_xgb, n_iter=500, random_state=state, cv=5, n_jobs=25,
                                       return_train_score=True)
random_search_xgb.fit(X_train, y_train)
best_xgb = random_search_xgb.best_estimator_
xgb_params = random_search_xgb.best_params_
print(random_search_xgb.best_params_)

param_grid_knn = {
    'n_neighbors': list(np.random.randint(1, 50, size=100)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': list(np.random.randint(1, 5, size=100))
}
knn_clf = KNeighborsClassifier()
random_search_knn = RandomizedSearchCV(knn_clf, param_grid_knn, n_iter=500, random_state=state, cv=5, n_jobs=25,
                                       return_train_score=True)
random_search_knn.fit(X_train, y_train)
best_knn = random_search_knn.best_estimator_
knn_params = random_search_knn.best_params_
print(knn_params)

random_forest = RandomForestClassifier(**rf_params)
xgboost = XGBClassifier(**xgb_params)
lg = LogisticRegression(**lg_params)
knn = KNeighborsClassifier(**knn_params)
svm = SVC(**svm_params)

models = [random_forest, xgboost, knn, lg, svm]
model_names = ['Random Forest', 'XGBoost', 'KNN', 'LG', 'svm']
auc = []
acc = []
f1 = []
precision = []
recall = []

for model, model_name in zip(models, model_names):
    model.fit(X_train, y_train)
    print(model_name)
    auc.append(roc_auc_score(y_test, model.predict(X_test)))
    acc.append(accuracy_score(y_test, model.predict(X_test)))
    f1.append(f1_score(y_test, model.predict(X_test)))
    precision.append(precision_score(y_test, model.predict(X_test)))
    recall.append(recall_score(y_test, model.predict(X_test)))
    joblib.dump(model, dirt + model_name + '.pkl')

result = {'AUC': auc,
          'Acc': acc,
          'F1-score': f1,
          'Precision': precision,
          'Recall': recall,
          }
result = pd.DataFrame(result)
result['Model'] = model_names
result.to_csv('./model_performance.csv', index=False)
