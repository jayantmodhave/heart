# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle




# data Preprocessing
dataset = pd.read_csv('Heart.csv')
print(dataset)
 
# Check for Null Values and replace null values with median
dataset["education"] = dataset["education"].fillna(dataset["education"].median())
dataset["cigsPerDay"] = dataset["cigsPerDay"].fillna(dataset["cigsPerDay"].median())
dataset["BPMeds"] = dataset["BPMeds"].fillna(dataset["BPMeds"].median())
dataset["totChol"] = dataset["totChol"].fillna(dataset["totChol"].median())
dataset["BMI"] = dataset["BMI"].fillna(dataset["BMI"].median())
dataset["heartRate"] = dataset["heartRate"].fillna(dataset["heartRate"].median())
dataset["glucose"] = dataset["glucose"].fillna(dataset["glucose"].median())

#drop education cloumn because it is negative correlation to targat variable
dataset.drop(['education'],inplace= True, axis=1)


# Check for outliers and treating of outliers
# getting median
totChol_dataset= pd.DataFrame(dataset["totChol"])
totChol_median = totChol_dataset.median()

# IQR for totChol 
Q3 = totChol_dataset.quantile(q=0.75)
Q1 = totChol_dataset.quantile(q=0.25)
IQR = Q3-Q1

# boundries of outliers
IQR_LL = int(Q1-1.5*IQR)
IQR_UL = int(Q3-1.5*IQR)

# finding and treating of outliers
dataset.loc[dataset['totChol']>IQR_UL, 'totChol'] = int(totChol_dataset.quantile(q=0.99))
dataset.loc[dataset['totChol']>IQR_LL, 'totChol'] = int(totChol_dataset.quantile(q=0.01))





'''
The distribution is highly imbalanced. As in, 
the number of negative cases outweigh the number of positive cases. 
This would lead to class imbalance problem while fitting our models. 
Therefore, this problem needs to be addressed
'''
# RESAMPLING IMBALANCED DATASET
target1=dataset[dataset['TenYearCHD']==1]
target0=dataset[dataset['TenYearCHD']==0]

from sklearn.utils import resample
target1=resample(target1,replace=True,n_samples=len(target0),random_state=40)

target=pd.concat([target0,target1])
target['TenYearCHD'].value_counts()
dataset=target




'''
Using inbuilt class feature_importances of tree based classifiers.
this inbuild function gives top 9 important attributes.
'''
dataset=dataset[['age','gender','cigsPerDay','prevalentHyp','heartRate','BMI','sysBP','diaBP','glucose','TenYearCHD']]





# spitting data into training and testing part
from sklearn.model_selection import train_test_split

X = dataset.drop('TenYearCHD', axis=1)
y = dataset['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





# Testing data using Random Forest 
X = dataset.drop('TenYearCHD', axis=1)
y = dataset.iloc[:,-1]

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()
model.fit(X_train, y_train)
result = model.fit(X_train,y_train)
predictions = result.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
print('Accuracy=',accuracy_score(y_test,predictions))
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[35,1,1,0,75.0,20.2,130.0,90.0,85.0]]))

