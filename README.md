# predicting_an_employees_access_needs

# 1. Importing Datasets :

#importing required packages,
import matplotlib.pyplot as plt [#plotting library extension for numpy]
import seaborn as sns [ #visualisation library based on matplotlib]
import numpy as np [ #numerical python , library consisting of multidimensional array objects,used for processing of arrays]
import pandas as pd [ #data analysis tool]
#Importing Datasets
ls
[#displays all files ]
data = pd.read_csv('./sample_data/train.csv')[#loading data]
print(data.shape)[#printing data]
data.head()


# 2. Data Exploration :

data_explore = data.copy()  
#copying data
data_explore.info()
#finding null values
data_explore.nunique()
an employee can have only one manager at a time, then we can consider that the dataset contains information of maximum 4243 employees.
There are same number of unique values for ROLE_TITLE and ROLE_CODE. There is 1-to-1 mapping between these columns. So for our problem only one feature is sufficent.

sns.countplot(x='ACTION', data=data_explore)
#finding dataset is balanced and imbalanced.

#finding out top 15 Resources, Role department, Role family, Role codes for which most access is requested.
data_explore_resources = data_explore[['RESOURCE', "ACTION"]].groupby(by='RESOURCE').count()
data_explore_resources.sort_values('ACTION', ascending=False).head(n=15).transpose()

data_explore_role_codes = data_explore[['ROLE_CODE', "ACTION"]].groupby(by='ROLE_CODE').count()
data_explore_role_codes.sort_values('ACTION', ascending=False).head(n=15).transpose()

data_explore_role_family = data_explore[['ROLE_FAMILY', "ACTION"]].groupby(by='ROLE_FAMILY').count()
data_explore_role_family.sort_values('ACTION', ascending=False).head(n=15).transpose()

#ploting correlation matrix 
plt.figure(figsize=(12, 7))
corr_matrix = data_explore.corr()
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True, cbar=False)
plt.tight_layout()
corr_matrix['ACTION'].sort_values(ascending=False)
