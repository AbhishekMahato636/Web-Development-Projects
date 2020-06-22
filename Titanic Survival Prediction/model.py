# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

titanic = pd.read_csv('train_kaggle.csv')

#Filling null values with median in age with respect to SibSp, Parch and Pclass
def fill_age_missing_values(titanic):
    Age_Nan_Indices = list(titanic[titanic["Age"].isnull()].index)

    #for loop that iterates over all the missing age indices
    for index in Age_Nan_Indices:
        #temporary variables to hold SibSp, Parch and Pclass values pertaining to the current index
        temp_Pclass = titanic.iloc[index]["Pclass"]
        temp_SibSp = titanic.iloc[index]["SibSp"]
        temp_Parch = titanic.iloc[index]["Parch"]
        age_median = titanic["Age"][((titanic["Pclass"] == temp_Pclass) & (titanic["SibSp"] == temp_SibSp) & (titanic["Parch"] == temp_Parch))].median()
        if titanic.iloc[index]["Age"]:
            titanic["Age"].iloc[index] = age_median
        if np.isnan(age_median):
            titanic["Age"].iloc[index] = titanic["Age"].median()
    return titanic

titanic = fill_age_missing_values(titanic)


#Filling missing value in Embarked column with mode
titanic['Embarked'].mode()
titanic['Embarked']=titanic['Embarked'].fillna('S')

#Filling missing value in Fare column with median
titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())

#By adding SibSp and Parch we can have a new column like total family
titanic["Ftotal"] = 1 + titanic["SibSp"] + titanic["Parch"]

titanic["Sex"] = titanic["Sex"].astype('category')
titanic.dtypes

titanic["sex"] = titanic["Sex"].cat.codes
titanic.head()

titanic["Embarked"] = titanic["Embarked"].astype('category')
titanic.dtypes

titanic["embarked"] = titanic["Embarked"].cat.codes
titanic.head()

#Dropping unnecessary columns from dataset
titanic=titanic.drop(['Sex','Cabin','Embarked','SibSp','Parch','Name','Ticket','PassengerId'],axis=1)

titanic=titanic.drop(['Fare'],1)

#Assigning X and y variables
X = titanic.drop('Survived',1)
y = titanic['Survived']



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

#Fitting model with trainig data
classifier.fit(X, y)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))