#scaling the data
import  matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import confusion_matrix
cancer = pd.read_csv('cancer_dataset.csv')
#selects the first twelve instances of the data
cancero= cancer.iloc[:, 1:12]
# prints the data shape, i.e the number of cells and rows
print(cancer.shape)
#select features and labels from the data set
#This is done by excluding instance 0-ID and instance 11-Label
features, labels= cancero.iloc[:, 1:10], cancero.loc[:,['diagnosis']]
X, y=features,labels
sc = StandardScaler()
X = sc.fit_transform(X)
#randomly selecting 80% of the data for training purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=42)
#The random forest method consists in creating multiple decision trees
#and taking the average of those to make the prediction.
#This method improves the accuracy of the decision tree model as 
#it prevents the data from overfitting the model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(y_test,y_pred)

#creatig a confusion matrix for "random forests"
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

# check true positve, true negative, false positive, and false negative
#for random forests
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
acc=float(tn+tp)/float(tn+fp+fn+tp)
print("Accuracy=%f" % acc)
