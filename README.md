# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```python
/*
import pandas as pd 
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VISHAL S
RegisterNumber: 212224040364
data = pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]) 
*/
```

## Output:
HEAD:
![image](https://github.com/user-attachments/assets/2b9125e9-1332-4cf2-84cd-41d88026777a)
COPY:
![image](https://github.com/user-attachments/assets/68a6d715-bcdc-4462-aa25-e8967e8f4de9)
FIT TRANSFORM:
![image](https://github.com/user-attachments/assets/a69177f4-2131-4de8-a63b-f9add3ff73ea)
LOGISTIC REGRESSION:
![image](https://github.com/user-attachments/assets/5f83c23d-e028-49a9-9728-a9df7c6a777d)
ACCURACY SCORE:
![image](https://github.com/user-attachments/assets/2354720c-761e-4c28-92f3-a1c2376ca028)
CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/c2527233-37fc-402d-acb3-9a59f6999376)
![image](https://github.com/user-attachments/assets/ea1465f5-3954-40aa-a94a-a0285177a0c3)
CLASSIFICATION REPORT:
![image](https://github.com/user-attachments/assets/d6fed8b9-e0a8-4823-acdc-b1b682f1e8ef)
PREDICTION:
![image](https://github.com/user-attachments/assets/64dce08a-905d-45c2-889e-c76825625a6c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
