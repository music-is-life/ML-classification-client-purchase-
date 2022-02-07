# importing libaries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# import dataset
dataset = pd.read_csv('/home/daniel/Documents/מטלות/למידת מכונה/מטלה 4 (סיווג)/dataset.csv')

#checking data set:
print(dataset.isna().sum()) # no NaN values..
print(dataset.shape)

X = dataset.drop(columns=['Purchased'])
y = dataset.iloc[:,2:]

# scaling:
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

# DF for summery table:
final_table = pd.DataFrame(columns=['Model', 'best_Parameters', 'Mean F1 score','STD test score'])
final_table['Model'] = ['KNN', 'Logistic regression',
                            'Linear SVC', 'Polynomial SVC', 'Gaussian SVC']

# splitting data to train and test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


### KNN classifier:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
cv = RepeatedKFold(n_splits=1000, n_repeats=3, random_state=0)
knn2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 21)}
knn_gscv = GridSearchCV(knn2, param_grid, cv=cv, scoring= 'f1',n_jobs=-1)
knn_gscv.fit(X, y)
results_dic = {'K Value': knn_gscv.cv_results_['param_n_neighbors'],
               'Mean F1 score': knn_gscv.cv_results_['mean_test_score'],
               'STD test score': knn_gscv.cv_results_['std_test_score']}
results_df = pd.DataFrame(data = results_dic)
print(knn_gscv.best_params_)
print(results_df)
final_table['best_Parameters'][0] = knn_gscv.best_params_
final_table['Mean F1 score'][0] = knn_gscv.best_score_
final_table['STD test score'][0] = results_df.iloc[0,2]


### Logistic regression:
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
cv_lr = cross_val_score(lr, X, y, cv=cv, scoring='f1', n_jobs=-1)
final_table['best_Parameters'][1] = 'NaN'
final_table['Mean F1 score'][1] = cv_lr.mean()
final_table['STD test score'][1] = cv_lr.std()


### Linear SVC:
from sklearn.svm import SVC, LinearSVC
svc_model = LinearSVC().fit(X_train,y_train)
cv_l_svc = cross_val_score(svc_model, X, y, cv=cv, scoring='f1', n_jobs=-1)
final_table['best_Parameters'][2] = 'NaN'
final_table['Mean F1 score'][2] = cv_l_svc.mean()
final_table['STD test score'][2] = cv_l_svc.std()

### Polynomial SVC:
poly_model = SVC(kernel='poly')
param_grid = {'degree': np.arange(2, 6)}
poly_SVC = GridSearchCV(poly_model, param_grid, cv=cv, scoring= 'f1',n_jobs=-1)
poly_SVC.fit(X, y)
results_dic_2 = {'Degree': poly_SVC.cv_results_['param_degree'],
               'Mean F1 score': poly_SVC.cv_results_['mean_test_score'],
               'STD test score': poly_SVC.cv_results_['std_test_score']}
results_df_2 = pd.DataFrame(data = results_dic_2)
final_table['best_Parameters'][3] = poly_SVC.best_params_
final_table['Mean F1 score'][3] = results_df_2.iloc[0,1]
final_table['STD test score'][3] = results_df_2.iloc[0,2]


### Gaussian SVC:

guass_model = SVC(kernel='rbf')
param_grid = {'C': (0.2, 0.5,1.2,1.8,3)}
guass_SVC = GridSearchCV(guass_model, param_grid, cv=cv, scoring= 'f1',n_jobs=-1)
guass_SVC.fit(X, y)
results_dic_3 = {'C': guass_SVC.cv_results_['params'],
               'Mean F1 score': guass_SVC.cv_results_['mean_test_score'],
               'STD test score': guass_SVC.cv_results_['std_test_score']}
results_df_3 = pd.DataFrame(data = results_dic_3)
final_table['best_Parameters'][4] = guass_SVC.best_params_
final_table['Mean F1 score'][4] = results_df_3.iloc[2,1]
final_table['STD test score'][4] = results_df_3.iloc[2,2]
