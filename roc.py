import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
dataset=load_breast_cancer()
df=pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
df['target']=dataset['target']
print(df.head())
print(df.info())
from sklearn.model_selection import train_test_split
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=100)
print('train data of x_train is',x_train.shape)
print('train data of x_test is',x_test.shape)
print('train data of y_train is:',y_train.shape)
print('train data of y_test is:',y_test.shape)
#LOGISTIC REGRESSION
print(y_train.value_counts())
print(y_test.value_counts())
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print(y_pred)

from sklearn import metrics
conf=metrics.confusion_matrix(y_test,y_pred)
print(conf)

classification=metrics.classification_report(y_test,y_pred,digits=3)
print(classification)

import matplotlib.pyplot as plt
y_pred_proba=logreg.predict_proba(x_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test, y_pred_proba)
auc=metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label='Logistic,auc='+str(auc))
plt.legend(loc=4)
plt.show()

y_pred_proba=logreg.predict_proba(x_test)[::,1]
print(y_pred_proba)

print("fpr:",fpr)
print("tpr:",tpr)
print("_",_)

# KNEIGHBORSCLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
k=int(input('Enter number of neighbours'))
model=KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)
predicteddata=model.predict(x_test)
print("enter the number of neighbors",k)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,predicteddata)
print("Accuracy of model is : ",accuracy)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predicteddata)
print(cm)

import matplotlib.pyplot as plt
y_pred_proba_knn = model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba_knn)
auc = metrics.roc_auc_score(y_test, y_pred_proba_knn)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend()
plt.show()


import matplotlib.pyplot as plt
y_pred_proba_knn = model.predict_proba(x_test)[::,1]
fpr_k, tpr_k, _k = metrics.roc_curve(y_test, y_pred_proba_knn)
auc = metrics.roc_auc_score(y_test, y_pred_proba_knn)
plt.plot(fpr_k,tpr_k,label="KNN, auc="+str(auc))
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Logistic, auc="+str(auc))
plt.legend()
plt.show()

#SVM
from sklearn.svm import SVC
main=SVC(probability=True)
main.fit(x_train,y_train)
predict=main.predict(x_test)
print(predict)
print(y_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,predict)
print("Accuracy of model is : ",accuracy)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predict)
print(cm)
from sklearn.metrics import classification_report
cr=classification_report(y_test,predict)
print(cr)

#Dicision Tree
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
predict=model.predict(x_test)
print(predict)
print(y_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predict)
print(cm)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predict)
print(accuracy)
from sklearn.metrics import classification_report
cr=classification_report(y_test,predict)
print(cr)
















































































































































