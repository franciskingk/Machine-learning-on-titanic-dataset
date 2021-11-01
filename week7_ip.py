
# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




#importing the datasets
train_df= pd.read_csv('/home/francis/Downloads/train (5).csv')
test_df=pd.read_csv('/home/francis/Downloads/test (1).csv')


# checking for null values in the train dataset in percentages
train_df.isnull().sum()* 100/ len(train_df)




#plotting the null values
sns.heatmap(train_df.isnull(),cbar=False)


# The cabin has the largest number of null values. above 70%. therefore we will drop this column for this section.
# For the age column, we will replace the nulls with the average age.



# dropping the Cabin column
train_df.drop(['Cabin'],axis=1,inplace=True)


# replacing the nulls in the age column
train_df.fillna(train_df['Age'].mean(),inplace=True)



# dropping the cabin column 
test_df.drop(['Cabin'],axis=1,inplace=True)




#replacing the null age 
test_df.fillna(test_df['Age'].mean(),inplace=True)


train_df.drop(['Name','Sex','Ticket','Embarked','PassengerId'],axis=1,inplace=True)

# ## KNN classification



#splitting the train and test data
#importting the train test split
from sklearn.model_selection import train_test_split
X=train_df.drop(['Survived'],axis=1)
y=train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)



# checking the shape of the train test
X_train.shape,y_train.shape


#checking the datatypes
X_train.dtypes




#training the model
# importing the library
from sklearn.neighbors import KNeighborsClassifier

#training
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)


#making the prediction
y_pred=knn.predict(X_test)


# checking the score of the model

#importing the library
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))