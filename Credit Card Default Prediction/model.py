# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df_credit= pd.read_csv('default.csv')

# Sample n rows
n = df_credit.shape[0]
df_credit = df_credit.sample(n)

#Select the variables to be one-hot encoded
one_hot_features = ['student', 'default']
# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(df_credit[one_hot_features],drop_first=True)
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)


# Replacing categorical columns with dummies
fdf = df_credit.drop(one_hot_features,axis=1)
fdf = pd.concat([fdf, one_hot_encoded] ,axis=1)
fdf.head()

#Standardize rows into uniform scale

X = fdf.drop(['default_Yes'],axis=1)
y = fdf['default_Yes']

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(C=1e42,class_weight='balanced')

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[52, 1235, 0]]))'''