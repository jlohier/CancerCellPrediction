#scaling the data
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
#creating  DECISION TREE with 5 levels 
clf = tree.DecisionTreeClassifier(max_depth = 5)
#Testing the remaining 20%
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(y_test,y_pred)
#creating a confusion matrix
confusion_matrix(y_test, y_pred)
# check true positve, true negative, false positive, and false negative
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
acc=float(tn+tp)/float(tn+fp+fn+tp)
print("Accuracy=%f" % acc)
