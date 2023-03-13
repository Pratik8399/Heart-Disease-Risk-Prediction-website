from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

df = df.rename(columns={"cp": "chest_pain", "trestbps": "blood_pressure", "fbs": "blood_sugar", "ca": "vessels", "chol": "cholesterol", "thalach": "Max. Heart Rate", "exang": "Exercise Induced Angina", "thal": "thalassemia"})


features = df[['age', 'sex', 'chest_pain', 'Max. Heart Rate', 'Exercise Induced Angina','oldpeak', 'slope', 'vessels', 'thalassemia']]
target = df['target']

# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
#print("RF's Accuracy is: ", x)

#print(classification_report(Ytest,predicted_values))

from sklearn.model_selection import cross_val_score

# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)


from sklearn.metrics import confusion_matrix

# Lets #print the Confusion matrix first

#plt.rcParams['figure.figsize'] = (20, 10)

cm = confusion_matrix (Ytest, predicted_values)

sns.heatmap(cm, annot = True, cmap = 'Blues')

#plt.title('Confusion Matrix for Random Forest', fontsize  = 15)
#plt.show

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
#print("DecisionTrees's Accuracy is: ", x*100)

#print(classification_report(Ytest,predicted_values))

# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)

#Lets #print the Confusion matrix first

#plt.rcParams['figure.figsize'] = (20, 10)

cm = confusion_matrix (Ytest, predicted_values)

sns.heatmap(cm, annot = True, cmap = 'Blues')

#plt.title('Confusion Matrix for Decision Tree', fontsize  = 15)
#plt.show()

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
#print("Logistic Regression's Accuracy is: ", x)

#print(classification_report(Ytest,predicted_values))

# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv = 5)

from sklearn.metrics import confusion_matrix

# Lets #print the Confusion matrix first

#plt.rcParams['figure.figsize'] = (20, 10)

cm = confusion_matrix (Ytest, predicted_values)

sns.heatmap(cm, annot = True, cmap = 'Blues')

#plt.title('Confusion Matrix for Logistic Regression', fontsize  = 15)


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
#print("Naive Bayes's Accuracy is: ", x)

#print(classification_report(Ytest,predicted_values))

# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)

from sklearn.metrics import confusion_matrix

# Lets #print the Confusion matrix first

#plt.rcParams['figure.figsize'] = (20, 10)

cm = confusion_matrix (Ytest, predicted_values)

sns.heatmap(cm, annot = True, cmap = 'Blues')

#plt.title('Confusion Matrix for Guassian Naive Bayes', fontsize  = 15)

from sklearn.svm import SVC
# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler().fit(Xtrain)
X_train_norm = norm.transform(Xtrain)
# transform testing dataabs
X_test_norm = norm.transform(Xtest)
SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm,Ytrain)
predicted_values = SVM.predict(X_test_norm)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
#print("SVM's Accuracy is: ", x)

#print(classification_report(Ytest,predicted_values))

# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)

from sklearn.metrics import confusion_matrix

# Lets #print the Confusion matrix first

#plt.rcParams['figure.figsize'] = (20, 10)

cm = confusion_matrix (Ytest, predicted_values)

sns.heatmap(cm, annot = True, cmap = 'Blues')

#plt.title('Confusion Matrix for SVM', fontsize  = 15)

accuracy_models = dict(zip(model, acc))
#for k, v in accuracy_models.items():
    #print (k, '-->', v)

#data = np.array([[57,0,0,123,1,0.2,1,0,3]])
#prediction = RF.predict(data)
#print('The health Status is :- ',prediction)
from pydantic import BaseModel
class form (BaseModel):  
    age: int
    chestPain:int
    gender:str
    MaxHeartRate:int
    ExerciseInducedAngina:str
    oldpeak:float
    slope:int
    vessels:int
    thalassemia:int

sex=0
eia=0
def disease(obj:form):
    
    if (obj.gender=='Male'):
        sex=0
    else:
        sex=1

    if (obj.ExerciseInducedAngina=='no'):
        eia=0
    else:
        eia=1
    data= np.array([[int(obj.age),sex,int(obj.chestPain), int(obj.MaxHeartRate), eia, float(obj.oldpeak), int(obj.slope), int(obj.vessels), int(obj.thalassemia)]])
    
    prediction = RF.predict(data)
    res=prediction[0]
    if res==0:
        return {'False'}
    else:
        return {'True'}

