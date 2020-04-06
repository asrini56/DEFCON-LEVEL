import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



trXnew = []

# Importing the dataset
dataset = pd.read_csv('train.csv')

dataset["nation"] = abs(dataset["Allied_Nations"] - dataset["Hostile_Nations"])
dataset["threat"] = abs(dataset["Active_Threats"] - dataset["Inactive_Threats"])
dataset["Closest_Threat_Distance(km)"] = dataset["Closest_Threat_Distance(km)"] / 1.6
dataset["Troops_Mobilized(thousands)"] = dataset["Troops_Mobilized(thousands)"] / 1000

X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
X2 = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9]]
y = dataset.iloc[:, 10].values

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X2,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X2.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

from sklearn.decomposition import PCA
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
for i in range(len(X)):
        a = np.mean(X[i])
        b = np.std(X[i])
        trXnew.append([b])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)   


# Fitting classifier to the Training set
# Create your classifier here

constant_value = len(dataset.columns)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 550, criterion = 'entropy',
                                   max_depth = 14, random_state = 43, min_samples_split =2
                                   , min_samples_leaf=constant_value
                                  , min_impurity_decrease=0.0001, min_impurity_split=None ,min_weight_fraction_leaf=0.0
                                   ,oob_score= True)

from sklearn.naive_bayes import GaussianNB
clf3 = GaussianNB()

bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier( class_weight=None,criterion='entropy',
                       max_depth=15, max_leaf_nodes=None,
                       min_impurity_decrease=0.0002, min_impurity_split=None,
                       min_samples_leaf=constant_value, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       random_state=21, splitter='best'), oob_score=True, random_state=42)


model = VotingClassifier(estimators=[('rf', classifier), ('dt', bagging)], voting='soft')
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(model.score(X_test,y_test))
ac = accuracy_score(y_test,preds)
print('Accuracy is: ',ac)