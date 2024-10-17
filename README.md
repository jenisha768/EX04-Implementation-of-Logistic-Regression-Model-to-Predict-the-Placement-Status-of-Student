# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing
2. Splitting the Data
3. Train the Logistic Regression Model
4. Evaluate the Model

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JENISHA TEENA ROSE F
RegisterNumber:2305001010
*/

import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print("Accuracy score:\n",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification report:\n",classification)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:

![1](https://github.com/user-attachments/assets/246a7764-ca1b-496a-87c1-2d1d22a03b24)
![2](https://github.com/user-attachments/assets/bbb05d08-2462-4eb9-a875-d33a852e8a0b)
![3](https://github.com/user-attachments/assets/c99d57c5-db40-4f95-8096-df89e8da872c)
![Screenshot (79)](https://github.com/user-attachments/assets/fe255eed-6b90-46da-b07f-50adcc2cd906)
![Screenshot (81)](https://github.com/user-attachments/assets/03b0e548-409a-44c8-bf2c-716968d04e0d)
![Screenshot (82)](https://github.com/user-attachments/assets/868bf5c5-a2cf-44f4-a350-bdf38cf4265c).
![Screenshot (83)](https://github.com/user-attachments/assets/7e82915d-649f-4511-9342-337e44e1d1ce)
![Screenshot (84)](https://github.com/user-attachments/assets/51956685-eecd-49fc-a1ef-3a6717805389)
![Screenshot (85)](https://github.com/user-attachments/assets/5534ce69-691f-4154-b49b-3d822e797c63)
![Screenshot (86)](https://github.com/user-attachments/assets/a2a03162-3707-4d79-9102-8068ae7c02d9)
![Screenshot (87)](https://github.com/user-attachments/assets/e4d9a101-5d20-4968-9f6a-fc6457810153)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
