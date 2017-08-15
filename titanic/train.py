# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:45:30 2017

@author: winson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier 
from sklearn.ensemble import VotingClassifier, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
import pickle
import os

def loadData(trainpath, testpath):
    train = pd.read_csv('../input/train.csv', header = 0)
    n_train = train.shape[0]
    test = pd.read_csv('../input/test.csv', header = 0)
    return train,test,n_train

def EDA(train, test):
    train.describe(include='all')
    test.describe(include='all')
    train.boxplot(by = 'Survived')
    train.groupby('Survived').mean().plot(kind = 'barh')
    train.groupby('Survived')[['Pclass','SibSp', 'Parch']].mean().plot(kind = 'barh')
    train.groupby('Sex')['Survived'].value_counts()
    train.groupby('Ticket')['Survived'].value_counts()
    train.groupby('Cabin')['Survived'].value_counts()
    train.groupby('Embarked')['Survived'].value_counts()

def transform_Cat(df, column):
    df_c = pd.get_dummies(df[column], drop_first = False)
    del df[column]
    df = df.join(df_c)
    return df

def featureSelection(Xtrain, Ytrain, Xtest):
    print('Originally feature number:' + str(Xtrain.shape[1]))
    rffs = RandomForestClassifier(max_depth=50, max_features='log2', min_samples_leaf=5,min_samples_split=10,n_estimators=50)
    rffs.fit(Xtrain, Ytrain)
    modelselector = SelectFromModel(rffs, prefit=True, threshold = 0.02)    
    Xtrain = modelselector.transform(Xtrain)
    Xtest = modelselector.transform(Xtest)
    print('After selection, feature number:' + str(Xtrain.shape[1]))
    return Xtrain, Xtest

def CVstats(cvmodel, Xtrain, Ytrain):
    print('\ntrain stats on CV, mean acc:')
    print(cvmodel.cv_results_['mean_train_score'])
    print('\ntest stats on CV, mean acc:')
    print(cvmodel.cv_results_['mean_test_score'])
    print ('\nbest model based on CV: params:')
    print(cvmodel.best_estimator_.get_params())    
    trainscore = [cvmodel.cv_results_[str('split' + str(i) + '_train_score')][cvmodel.best_index_]
                    for i in range(0, cvmodel.cv)]
    testscore = [cvmodel.cv_results_[str('split' + str(i) + '_test_score')][cvmodel.best_index_]
                    for i in range(0, cvmodel.cv)]
    print('\nbest model train stat on different folds:')
    print(trainscore)
    print('\nbest model test stat on different folds:')
    print(testscore)
    print('\nbest model train acc on whole data:' + str(cvmodel.best_estimator_.score(Xtrain, Ytrain)))

def Mypredict(model, Xtest, outputfile):
    Ytest = model.predict(Xtest)
    testoutput = np.stack((testPassengerId.values, Ytest)).T
    testoutputdf = pd.DataFrame(testoutput, columns=['PassengerId', 'Survived'])
    testoutputdf.to_csv(outputfile, index = False)

def Ensemble(fileprefix):
    indexname = ''
    labelname = ''
    index = []
    label = []
    for f in os.listdir("."):
        if f.startswith(fileprefix) and not f.startswith(fileprefix + 'ensemble'):
            data = pd.read_csv(f, header = 0)
            indexname = data.columns[0]
            labelname = data.columns[1]
            index = data[indexname]
            label.append(data[labelname])
    index = np.asarray(index)
    label = np.asarray(label)
    label = sps.mode(label)[0]
    testoutput = np.concatenate((index.reshape(-1, 1), label.reshape(-1,1)), axis=1)
    testoutputdf = pd.DataFrame(testoutput, columns=[indexname, labelname])
    testoutputdf.to_csv(fileprefix + 'ensemble.csv', index = False)
            
def impute_age(traintest):
    agetrain = traintest[pd.notnull(traintest['Age'])]
    agetest = traintest[pd.isnull(traintest['Age'])]
    columns = agetrain.columns.difference(['Age'])
    
    et = ExtraTreesRegressor(n_estimators=50)
    et.fit(agetrain[columns], agetrain.Age)    
    modelselector = SelectFromModel(et, prefit=True, threshold = 0.01)    
    Xtrainage = modelselector.transform(agetrain[columns])
    Xtestage = modelselector.transform(agetest[columns])
    
    knn = KNeighborsRegressor()
    ridge = RidgeCV(cv = 5)
    forest = ExtraTreesRegressor(n_estimators=50)
    
    ridge.fit(Xtrainage, agetrain.Age)
    forest.fit(Xtrainage, agetrain.Age)
    knn.fit(Xtrainage, agetrain.Age)
    
    missingAge1 = ridge.predict(Xtestage)
    missingAge2 = forest.predict(Xtestage)
    missingAge3 = knn.predict(Xtestage)
    missingAge = (missingAge1 + missingAge2 + missingAge3)/3
    return missingAge

def get_family(df):

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    
    # introducing other features based on the family size
    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5<=s else 0)
    
    return df

def get_titles(df):
    
    # we extract the title from each name
    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                        }
    
    # we map each title
    df['Title'] = df.Title.map(Title_Dictionary)
    return df


np.set_printoptions(threshold=np.nan)
njobs = 4

train,test,n_train = loadData('../input/train.csv', '../input/test.csv')
print('load data complete')
#EDA(train, test)
print('explore data complete')

columns = train.columns.difference(['Survived'])
traintest = pd.concat([train[columns], test[columns]],ignore_index = True)
del traintest['PassengerId']
del traintest['Ticket']
del traintest['Cabin']

testPassengerId = test['PassengerId']

mean_age = traintest['Age'].mean()
mean_fare = traintest['Fare'].mean()
mean_Embarked = traintest['Embarked'].mode()[0]
traintest["Embarked"].fillna(mean_Embarked, inplace=True)
traintest['Fare'].fillna(mean_fare, inplace=True)
traintest = get_titles(traintest)
traintest = get_family(traintest)

traintest = transform_Cat(traintest, 'Sex')
traintest = transform_Cat(traintest, 'Embarked')
traintest = transform_Cat(traintest, 'Title')

del traintest['Name']

missingAge = impute_age(traintest)
traintest.loc[pd.isnull(traintest['Age']), 'Age'] = missingAge[:]

print('preprocess data, impute missing value, transform feature complete')

Xtrain = traintest[0:n_train].values
Ytrain = train['Survived'].values
Xtest = traintest[n_train:].values

print('Feature selection')
#Xtrain, Xtest = featureSelection(Xtrain, Ytrain, Xtest)

print('finish preparing data, starting train test')

## random forest
param_grid =  {
                 'max_depth' : [5, 10, 15, 20],
                 'n_estimators': [500],
                 'max_features': ['log2'],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [2, 5, 10],
                 'bootstrap': [True],
                 }
gridrf = GridSearchCV(RandomForestClassifier(), param_grid, cv = 5, n_jobs = njobs)
gridrf.fit(Xtrain, Ytrain)
CVstats(gridrf, Xtrain, Ytrain)
rf = RandomForestClassifier(**gridrf.best_estimator_.get_params())
rf.fit(Xtrain, Ytrain)
Mypredict(rf, Xtest, 'gender_submission_rf.csv')


#extra tree classifier
gridet = GridSearchCV(ExtraTreesClassifier(), param_grid, cv = 5, n_jobs = njobs)
gridet.fit(Xtrain, Ytrain)
CVstats(gridet, Xtrain, Ytrain)
et = ExtraTreesClassifier(**gridrf.best_estimator_.get_params())
et.fit(Xtrain, Ytrain)
Mypredict(et, Xtest, 'gender_submission_et.csv')


############################# standardize data for following models
traintest_std = preprocessing.scale(traintest)
Xtrain = traintest_std[0:n_train]
Xtest = traintest_std[n_train:]

## Logistic regression
param_grid_lr = {'C':np.logspace(-3, 2, 10, endpoint=True),
              'penalty': ['l2', 'l1']}
gridlogreg = GridSearchCV(LogisticRegression(max_iter = 500), param_grid_lr, cv = 5, n_jobs = njobs)
gridlogreg.fit(Xtrain, Ytrain)
CVstats(gridlogreg, Xtrain, Ytrain)
lr = LogisticRegression(**gridlogreg.best_estimator_.get_params())
lr.fit(Xtrain, Ytrain)
Mypredict(lr, Xtest, 'gender_submission_lr.csv')


## SVM with polynomial/RBF kernel
C_range = np.logspace(-4, 2, 10, endpoint=True)
degree_range = [1, 2, 3]
gamma_range = np.logspace(-4, 2, 10, endpoint=True)
param_grid3 = [dict(kernel=['poly'], gamma= gamma_range, 
                    degree=degree_range, C=C_range),
                dict(kernel=['rbf'], gamma= gamma_range,  C=C_range)]
gridsvm = GridSearchCV(SVC(), param_grid3, cv = 5, n_jobs = njobs)
gridsvm.fit(Xtrain, Ytrain)
CVstats(gridsvm, Xtrain, Ytrain)
svm = SVC(**gridsvm.best_estimator_.get_params())
svm.fit(Xtrain, Ytrain)
Mypredict(svm, Xtest, 'gender_submission_svm.csv')

## Neural network, 2 hidden layers 
hl = [(5,5), (10, 5), (50, 10)]
activation =['logistic', 'relu']
alpha = np.logspace(-3, 2, 6, endpoint=True)
param_grid_nn = dict(hidden_layer_sizes = hl, activation = activation, alpha = alpha)
gridnn = GridSearchCV(MLPClassifier(
        solver = 'lbfgs', early_stopping=True, validation_fraction=0.1),
        param_grid_nn, cv = 5, n_jobs = njobs)
gridnn.fit(Xtrain, Ytrain)
CVstats(gridnn, Xtrain, Ytrain)
nn = MLPClassifier(**gridnn.best_estimator_.get_params())
nn.fit(Xtrain, Ytrain)
Mypredict(nn, Xtest, 'gender_submission_nn.csv')

## GBM
param_grid_gbm =  {
                 'learning_rate' : np.logspace(-3, 0, 4, endpoint=True),
                 'max_depth' : [2,3, 5, 10, 15],
                 'n_estimators': [500],
                 'max_features': ['sqrt'],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [2, 5, 10],
                 'subsample': [0.8],
                 }
gridgbm = GridSearchCV(GradientBoostingClassifier(),
                       param_grid_gbm, cv = 5, n_jobs = njobs)
gridgbm.fit(Xtrain, Ytrain)
CVstats(gridgbm, Xtrain, Ytrain)
gbm = GradientBoostingClassifier(**gridgbm.best_estimator_.get_params())
gbm.fit(Xtrain, Ytrain)
Mypredict(gbm, Xtest, 'gender_submission_gbm.csv')

## knn
param_grid_knn =  {
                 'n_neighbors':[5, 10, 20, 50],
                 'weights' : ['uniform', 'distance'],
                 'p' : [1,2]
                 }
gridknn = GridSearchCV(KNeighborsClassifier(),
                       param_grid_knn, cv = 5, n_jobs = njobs)
gridknn.fit(Xtrain, Ytrain)
CVstats(gridknn, Xtrain, Ytrain)
knn = KNeighborsClassifier(**gridknn.best_estimator_.get_params())
knn.fit(Xtrain, Ytrain)
Mypredict(knn, Xtest, 'gender_submission_knn.csv')

################ Result ensemble
label = Ensemble('gender_submission_')
print('done')