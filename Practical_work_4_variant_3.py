import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
data = pandas.read_csv('/content/drive/MyDrive/Colab Notebooks/aug_test.csv', index_col='city')
data_sel = data.loc[:, data.columns.isin(['enrollee_id', 'city_ development _index', 'gender', 'lastnewjob', 'training_hours', 'target'])] # , 
data_sel = data_sel.dropna()
data_sel['gender'] = np.where(data_sel['gender'] == 'Male', 0, 1)
Survived = data_sel.loc[:, data_sel.columns.isin(['gender'])]
X = data_sel.loc[:, data_sel.columns.isin(['enrollee_id', 'city_ development _index', 'lastnewjob', 'training_hours', 'target'])] # 

print(X)
from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation = train_test_split(X, Survived, test_size=.33, random_state=1)


T = DecisionTreeClassifier(random_state=241, max_depth = 4)

T = T.fit(x_train, y_train)
T

from google.colab import drive
drive.mount('/content/drive')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

#подбор параметров разделяющей гиперплоскости и гиперсферы по методу опорных векторов , , 'decision_function_shape':('ovo','ovr'), 'gamma': (1,2,3,'auto'),'shrinking':(True,False)
svm = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75), 'decision_function_shape':('ovo','ovr'), 'gamma': (1,2,3,'auto')}
clf = GridSearchCV(svm, parameters)
y_train = pd.DataFrame(y_train)
x_train = pd.DataFrame(x_train)
y_validation = pd.DataFrame(y_validation)
print('x_train:  ', x_train)
print('y_train:  ', y_train)
clf.fit(x_train.values, y_train.values.ravel())
print("SVM")
print("accuracy_1:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='accuracy'))))
print("f1_1:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='f1'))))
print("precision_1:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='precision'))))
print("recall_1:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='recall'))))

#логистическая регрессия
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x_train,y_train)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy_2 :",logreg_cv.best_score_)
print("accuracy_2:"+str(np.average(cross_val_score(logreg_cv, x_validation, y_validation, scoring='accuracy'))))
print("f1_2:"+str(np.average(cross_val_score(logreg_cv, x_validation, y_validation, scoring='f1'))))
print("precision_2:"+str(np.average(cross_val_score(logreg_cv, x_validation, y_validation, scoring='precision'))))
print("recall_2:"+str(np.average(cross_val_score(logreg_cv, x_validation, y_validation, scoring='recall'))))

parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

#градиентный бустинг
clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
print(clf.best_params_)
print("accuracy_3:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='accuracy'))))
print("f1_3:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='f1'))))
print("precision_3:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='precision'))))
print("recall_3:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='recall'))))

accuracy_1:0.8859813084112149
f1_1:0.0
precision_1:0.0
recall_1:0.0

tuned hyperparameters :(best parameters)  {'C': 0.001, 'penalty': 'l2'}
accuracy_2 : 0.9079170914033302
f1_2: 0.0
precision_2: 0.0
recall_2: 0.0

0.9079189686924494

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score


parameters = {
    "loss":["deviance"],
    "learning_rate": [ 0.05],
    "min_samples_split": np.linspace(0.39, 0.39),
    "min_samples_leaf": np.linspace(0.1, 0.1),
    "max_depth":[5],
    "max_features":["log2"],
    "criterion": ["friedman_mse"],
    "subsample":[0.9],
    "n_estimators":[10]
    }

clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
print(clf.best_params_) 
print("f1:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='f1'))))
print("precision:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='precision'))))
print("recall:"+str(np.average(cross_val_score(clf, x_validation, y_validation, scoring='recall'))))

0.9079189686924494
f1:0.0
precision: 0.0
recall: 0.0

Type Markdown and LaTeX:  𝛼2

# Случайный лес
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV

param_grid = { 'n_estimators': [200, 300, 400],'max_features': ['auto'],'max_depth' : list(range(1, 20)), 'criterion' :['gini']}

RFC = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 5, refit = True) 
RFC.fit(x_train, y_train)

print("accuracy:"+str(np.average(cross_val_score(RFC.best_estimator_, x_validation, y_validation, scoring='accuracy'))))
print("f1:"+str(np.average(cross_val_score(RFC.best_estimator_, x_validation, y_validation, scoring='f1'))))
print("precision:"+str(np.average(RFC(grid_search_cv.best_estimator_, x_validation, y_validation, scoring='precision'))))
print("recall:"+str(np.average(RFC(grid_search_cv.best_estimator_, x_validation, y_validation, scoring='recall'))))

from sklearn.model_selection import cross_val_score
print("accuracy:"+str(np.average(cross_val_score(RFC.best_estimator_, x_validation, y_validation, scoring='accuracy'))))
print("f1:"+str(np.average(cross_val_score(RFC.best_estimator_, x_validation, y_validation, scoring='f1'))))
print("precision:"+str(np.average(cross_val_score(RFC.best_estimator_, x_validation, y_validation, scoring='precision'))))
print("recall:"+str(np.average(cross_val_score(RFC.best_estimator_, x_validation, y_validation, scoring='recall'))))
