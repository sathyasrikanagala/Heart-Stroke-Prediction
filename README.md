# Heart-Stroke-Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
data=pd.read_csv('/content/healthcare-dataset-stroke-data.csv')
data
data.shape
data.info()
data.head()
data.tail()
data.describe()
data["gender"].value_counts()
data["bmi"].value_counts()
print('The highest hypertension was of:',data['hypertension'].max())
print('The lowest hypertension was of:',data['hypertension'].min())
print('The average hypertension in the data:',data['hypertension'].mean())


# Data-Visualization
import matplotlib.pyplot as plt

# Line plot
plt.plot(data['avg_glucose_level'])
plt.xlabel("avg_glucose_level")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['stroke']==1]['smoking_status'].value_counts()

ax1.hist(data_len,color='red')
ax1.set_title('Having stroke')

data_len=data[data['stroke']==0]['smoking_status'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having stroke')

fig.suptitle('Stroke')
plt.show()
data.duplicated()
newdata=data.drop_duplicates()
newdata
data.isnull().sum() #checking for total null values
data.avg_glucose_level=data['avg_glucose_level'].astype('int64')
data['bmi'] = data['bmi'].fillna(0).astype('int64')
data.age=data['age'].astype('int64')
data.head(5)


#Normalization

data[1:5]
from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:,2:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["age", "hypertension","heart_disease"])
scaled_df.head()

#One-Hot-Encoding

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['gender','ever_married','Residence_type','work_type','smoking_status']
encoder = OneHotEncoder(sparse=False, drop='first')  # 'drop' parameter removes one of the one-hot encoded columns to avoid multicollinearity
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
encoded_cols=encoded_cols.astype(int)
data = pd.concat([data, encoded_cols], axis=1)
data.drop(categorical_cols, axis=1, inplace=True)
data.head()

#split into train and test in the ration 70:30

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['stroke'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['stroke']
len(train_X), len(train_Y), len(test_X), len(test_Y)
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)
model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())

# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)

#Evaluate the model using various metrices
mse = mean_squared_error(test_Y, prediction)
rmse = mean_squared_error(test_Y, prediction, squared=False) #Calculate the square root of MSE
mae = mean_absolute_error(test_Y, prediction)
r_squared=r2_score(test_Y, prediction)

print('Mean squared Error:',mse)
print('Root Mean Squared Error:',rmse)
print('Mean Absolute Error:',mae)
print('R-squared:',r_squared)
