# -*- coding: utf-8 -*-
"""Copy of predicting_an_employee_s_access_needs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rWtBzUTGHvmlrixZkK1GDBu7hGUkam3m

- The given problem is related with time wasted in granting and revoking access to the employee within company.  For employee to access any resources he/she needs prior permission i.e. access of that resource. The access granting and revoking process is manual, handled by superviso. As employees move throughout a company, this access discovery/recovery cycle wastes a nontrivial amount of time and money.

- <b>Objective:</b> We have to build a model, learned using historical data, that will determine an employee's access needs, such that manual access transactions (grants and revokes) are minimized as the employee's attributes change over time. The model will take an employee's role information and a resource code and will return whether or not access should be granted.


- <b>Data:</b> In training dataset, each row has the ACTION (ground truth), RESOURCE, and information about the employee's role at the time of approval.
- Following are the features present in the training dataset:
    - ACTION: Target variable. ACTION is 1 if the resource was approved, 0 if the resource was not approved.
    - RESOURCE: An ID for each resource
    - MGR_ID: The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time
    - ROLE_ROLLUP_1: Company role grouping category id 1 (e.g. US Engineering)
    - ROLE_ROLLUP_2: Company role grouping category id 2 (e.g. US Retail)
    - ROLE_DEPTNAME: Company role department name (e.g. Retail)
    - ROLE_TITLE: Company role business title description (e.g. Senior Engineering Retail Manager)
    - ROLE_FAMILY_DESC: Company role family extended description (e.g. Retail Manager, Software Engineering)
    - ROLE_FAMILY: Company role family description (e.g. Retail Manager)
    - ROLE_CODE: Company role code; this code is unique to each role (e.g. Manager)
 
 
- All features has numerical values but all features are categorical features.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#Importing Datasets

ls

data = pd.read_csv('./sample_data/train.csv')
print(data.shape)
data.head()

"""## Step 2: Data Exploration

# New Section
"""

data_explore = data.copy()

data.info()

data_explore.info()

"""There is no column with null values."""

data_explore.nunique()

"""- In dataset description it is mention that an employee can have only one manager at a time, then we can consider that the dataset contains information of maximum 4243 employees.
- There are same number of unique values for ROLE_TITLE and ROLE_CODE. There is 1-to-1 mapping between these columns. So for our problem only one feature is sufficent.
"""

sns.countplot(x='ACTION', data=data_explore)

"""- We can see that we have imbalance dataset. There are very less records of not granting the access.

- Lets find out top 15 Resources, Role department, Role family, Role codes for which most access is requested.
"""

data_explore_resources = data_explore[['RESOURCE', "ACTION"]].groupby(by='RESOURCE').count()
data_explore_resources.sort_values('ACTION', ascending=False).head(n=15).transpose()

data_explore_role_dept = data_explore[['ROLE_DEPTNAME', "ACTION"]].groupby(by='ROLE_DEPTNAME').count()
data_explore_role_dept.sort_values('ACTION', ascending=False).head(n=15).transpose()

data_explore_role_codes = data_explore[['ROLE_CODE', "ACTION"]].groupby(by='ROLE_CODE').count()
data_explore_role_codes.sort_values('ACTION', ascending=False).head(n=15).transpose()

data_explore_role_family = data_explore[['ROLE_FAMILY', "ACTION"]].groupby(by='ROLE_FAMILY').count()
data_explore_role_family.sort_values('ACTION', ascending=False).head(n=15).transpose()

plt.figure(figsize=(12, 7))
corr_matrix = data_explore.corr()
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True, cbar=False)
plt.tight_layout()

corr_matrix['ACTION'].sort_values(ascending=False)

"""- There is no attribute to which target variable is strongly correlated.

## Step 3: Data Preprocessing
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

X = data.drop(columns=['ACTION'], axis=1).copy()
y = data['ACTION'].copy()
X.shape, y.shape

cat_attrs = list(X.columns)
cat_attrs

## Creating a MODEL and splitting the dataset into Train and Test data

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, y):
    strat_train_set = data.iloc[train_index]
    strat_test_set = data.iloc[test_index]

X_train = strat_train_set.drop('ACTION', axis=1)
y_train = strat_train_set['ACTION'].copy()
X_test = strat_test_set.drop('ACTION', axis=1)
y_test = strat_test_set['ACTION'].copy()
X_train.shape, X_test.shape

!pip install catboost

cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('cat_enc', OneHotEncoder(handle_unknown='ignore'))])

pre_process = ColumnTransformer([('cat_process', cat_pipeline, cat_attrs)], remainder='passthrough')

X_train_transformed = pre_process.fit_transform(X_train)
X_test_transformed = pre_process.transform(X_test)
X_train_transformed.shape, X_test_transformed.shape

"""- Since I will be using CatBoost Classifier. For CatBoost model, there is no need of encoding categorical model. Hence I will be creating a separate preprocessing pipeline for CatBoost model."""

cat_boost_pre_process = ColumnTransformer([('imputer', SimpleImputer(strategy='most_frequent'), cat_attrs)], remainder='passthrough')

X_cb_train_transformed = cat_boost_pre_process.fit_transform(X_train)
X_cb_test_transformed = cat_boost_pre_process.transform(X_test)
X_cb_train_transformed.shape, X_cb_test_transformed.shape

feature_columns = list(pre_process.transformers_[0][1]['cat_enc'].get_feature_names(cat_attrs))
len(feature_columns)

"""## Step 4: Modelling

- Evaluation metric for this competition is ROC AUC Score.
- Since we have imbalance dataset, I will use Matthews correlation coefficient (MCC) as another evaluation metric. 
- Value of MCC is lies between -1 to +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.
    - ${MCC} = \frac{(TP + TN) * (FP + FN)}{\sqrt{( (TP +FP) * (TP + FN) * (TN + FP) * (TN + FN))}}$
    
- MCC value will be high only if model has high accuracy on predictions of negative data instances as well as of positive data instances.
- I will be selecting the best model with highest ROC AUC Score.
"""

from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score, roc_curve
Matthew = make_scorer(matthews_corrcoef)

results = []

def plot_custom_roc_curve(clf_name, y_true, y_scores):
    auc_score = np.round(roc_auc_score(y_true, y_scores), 3)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, linewidth=2, label=clf_name+" (AUC Score: {})".format(str(auc_score)))
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel("FPR", fontsize=16)
    plt.ylabel("TPR", fontsize=16)
    plt.legend()
    
    
def performance_measures(model, X_tr=X_train_transformed, y_tr=y_train, X_ts=X_test_transformed, y_ts=y_test,
                         store_results=True):
    train_mcc = cross_val_score(model, X_tr, y_tr, scoring=Matthew, cv=kf, n_jobs=-1)
    test_mcc = cross_val_score(model, X_ts, y_ts, scoring=Matthew, cv=kf, n_jobs=-1)
    print("Mean Train MCC: {}\nMean Test MCC: {}".format(train_mcc.mean(), test_mcc.mean()))

    
    train_roc_auc = cross_val_score(model, X_tr, y_tr, scoring='roc_auc', cv=kf, n_jobs=-1)
    test_roc_auc = cross_val_score(model, X_ts, y_ts, scoring='roc_auc', cv=kf, n_jobs=-1)
    print("Mean Train ROC AUC Score: {}\nMean Test ROC AUC Score: {}".format(train_roc_auc.mean(), test_roc_auc.mean()))
    
    if store_results:
        results.append([model.__class__.__name__, np.round(np.mean(train_roc_auc), 3), np.round(np.mean(test_roc_auc), 3), np.round(np.mean(train_mcc), 3), np.round(np.mean(test_mcc), 3)])

def plot_feature_importance(feature_columns, importance_values, top_n_features=10):
    feature_imp = [ col for col in zip(feature_columns, importance_values)]
    feature_imp.sort(key=lambda x:x[1], reverse=True)
    
    if top_n_features:
        imp = pd.DataFrame(feature_imp[0:top_n_features], columns=['feature', 'importance'])
    else:
        imp = pd.DataFrame(feature_imp, columns=['feature', 'importance'])
    plt.figure(figsize=(10, 8))
    sns.barplot(y='feature', x='importance', data=imp, orient='h')
    plt.title('Most Important Features', fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.xlabel("")
    plt.show()

from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression(solver='liblinear', C=1, penalty='l2', max_iter=1000, random_state=42, n_jobs=-1)
logistic_reg.fit(X_train_transformed, y_train)

plot_feature_importance(feature_columns, logistic_reg.coef_[0], top_n_features=15)

performance_measures(logistic_reg)

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42,n_jobs=-1)
forest_clf.fit(X_train_transformed, y_train)

plot_feature_importance(feature_columns, forest_clf.feature_importances_, top_n_features=15)

performance_measures(forest_clf)

from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=300, max_depth=16, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_clf.fit(X_train_transformed, y_train)

plot_feature_importance(feature_columns, xgb_clf.feature_importances_, top_n_features=15)

performance_measures(xgb_clf)

!pip install catboost

from catboost import CatBoostClassifier

catboost_clf = CatBoostClassifier(loss_function='Logloss', iterations=500, depth=6, l2_leaf_reg=1, 
                                  cat_features=list(range(X_cb_train_transformed.shape[1])), 
                                  eval_metric='AUC', random_state=42, verbose=0)
catboost_clf.fit(X_cb_train_transformed, y_train)

performance_measures(catboost_clf, X_tr=X_cb_train_transformed, X_ts=X_cb_test_transformed)

plot_feature_importance(feature_columns, catboost_clf.feature_importances_, top_n_features=15)

logistic_reg_pipeline = Pipeline([('pre_process', pre_process), ('logistic_reg', logistic_reg)])
forest_clf_pipeline = Pipeline([('pre_process', pre_process), ('forest_clf', forest_clf)])
xgb_clf_pipeline = Pipeline([('pre_process', pre_process), ('xgb_clf', xgb_clf)])
catboost_clf_pipeline = Pipeline([('pre_process', cat_boost_pre_process), ('catboost_clf', catboost_clf)])

named_estimators = [('logistic_reg', logistic_reg_pipeline), ('forest_clf', forest_clf_pipeline), 
                    ('xgb_clf', xgb_clf_pipeline), ('catboost_clf', catboost_clf_pipeline)]

from sklearn.ensemble import VotingClassifier

voting_reg = VotingClassifier(estimators=named_estimators, voting='soft', n_jobs=-1)
voting_reg.fit(X_train, y_train)

performance_measures(voting_reg, X_tr=X_train, X_ts=X_test)

"""## Step 5: Model Evaluation"""

result_df = pd.DataFrame(results, columns=['Model', 'CV Train AUC Score', 'CV Test AUC Score', 'CV Train MCC', 'CV Test MCC'])
result_df

plt.figure(figsize=(8, 5))
plot_custom_roc_curve('Logistic Regression', y_test, logistic_reg.decision_function(X_test_transformed))
plot_custom_roc_curve('Random Forest', y_test, forest_clf.predict_proba(X_test_transformed)[:,1])
plot_custom_roc_curve('XGBoost', y_test, xgb_clf.predict_proba(X_test_transformed)[:,1])
plot_custom_roc_curve('CatBoost', y_test, catboost_clf.predict_proba(X_cb_test_transformed)[:,1])
plot_custom_roc_curve('Soft Voting', y_test, voting_reg.predict_proba(X_test)[:,1])
plt.show()

"""## Step 6: Make submission

- Since Catboost Classifier has better ROC AUC Score and also good MCC value, I will be selecting Catboost as final model to make predictions on test dataset.
"""

final_model = Pipeline([('pre_process', cat_boost_pre_process),
                        ('catboost', catboost_clf)])
final_model.fit(X_train, y_train)

test_data = pd.read_csv('./sample_data/test.csv')
test_data.head()

output = pd.DataFrame(test_data['id'])
test_data = test_data.drop('id', axis=1)

test_data.info()

predictions = final_model.predict(test_data)

output['ACTION'] = predictions.copy()

output

output.to_csv("/content/sample_data/output.csv", index=False)

"""#"""