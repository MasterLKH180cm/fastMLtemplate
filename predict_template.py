# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
import sys
import traceback
import pdb
def main():
    generater = DataGenerater()
    preprocessor = DataPreprocessor(generater.X, generater.Y)
    preprocessor.preprocess(5)
    metric = Metrics()
    workflow_1 = WorkFlow(preprocessor.x_train, preprocessor.y_train, preprocessor.x_test, preprocessor.y_test)
    workflow_1.addModel(Model(RandomForestClassifier(), metric))
    workflow_1.addModel(Model(DecisionTreeClassifier(), metric))
    workflow_1.addModel(Model(KNeighborsClassifier(n_neighbors=3), metric))
    workflow_1.fitAllModel()
    workflow_1.testAllModel()
    metric.show_results()

class DataGenerater():
    def __init__(self, Xpath = r'./', Ypath = r'./'):
        try:
            self.X = pd.read_csv(Xpath)
            self.Y = pd.read_csv(Ypath)
        except:
            print("loading failure, use default dataset")
            self.X = pd.DataFrame([[random.uniform(0,7) for i in range(10)] for j in range(100)] + [[random.uniform(3,10) for i in range(10)] for j in range(100)])
            self.X.columns = [str(i) for i in range(10)]
            self.Y = pd.DataFrame([0 for i in range(100)]+[1 for i in range(100)])

    def __len__(self):
       
        return len(self.Y)
    # def generateRawData(self):
    #     return self.X, self.Y

class DataPreprocessor():
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
    def fillNans(self, method='mean'):
        self.X = self.X.fillna(self.X.mean())
        # X = X.interpolate(method='linear')
        # X = X.fillna(0)
        # X = X.fillna(method='ffill')
        # X = X.fillna(method='bfill')
        # return X, Y
    def imbalanceProcess(self):
        # ros = RandomOverSampler(random_state=999)
        # X, Y = ros.fit_resample(X, Y)
        # ada = ADASYN(random_state=999)
        # X, Y = ada.fit_resample(X, Y)
        sm = SMOTE(random_state=999)
        self.X, self.Y = sm.fit_resample(self.X, self.Y)
    def normalize(self):
        self.X = normalize(self.X)
    def pca(self, n_components):
        pca = PCA(n_components)
        pca.fit(self.x_train)
        self.pca_x_train = pca.transform(self.x_train)
        self.pca_x_test = pca.transform(self.x_test)
        self.pca_y_train, self.pca_y_test = self.y_train, self.y_test
        
    def split(self):
        np.random.seed(999)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, shuffle = True, test_size = 0.25)
        # np.random.seed(999)
        # self.pca_x_train, self.pca_x_test, self.pca_y_train, self.pca_y_test = train_test_split(self.pca_X, self.Y, shuffle = True, test_size = 0.25)
    def preprocess(self, n_components=5):
        self.fillNans()
        self.imbalanceProcess()
        self.split()
        self.pca(n_components)

class Model():
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric
        self.name = self.model.__class__.__name__
    def fit(self, X, Y):
        self.model.fit(X, Y)
    
    def predict(self, X):
        self.Y_Pred = self.model.predict(X)
        
     
    def confusionMatrix(self, Y):
        self.ConfusionMatrix = confusion_matrix(Y, self.Y_Pred)
        
    def save(self, path=r'./'):
        joblib.dump(self.model, path + self.name + 'pkl')
    
    def predict_proba(self, X):
        self.proba = self.model.predict_proba(X)[:, 1]
        
    def auc(self, Y):
        self.auc_score = roc_auc_score(Y, self.proba)
        
    def plot_roc_curve(self, X, Y):
        self.predict_proba(X)
        self.auc(Y)
        fpr, tpr, thresholds = roc_curve(Y, self.proba)
        self.fpr_tpr = [fpr, tpr]
        self.plot_curve(fpr, tpr, self.auc_score, self.name)
    
    def plot_curve(self, fpr, tpr, auc, model):
        plt.rcParams['savefig.dpi'] = 150 #图片像素
        plt.rcParams['figure.dpi'] = 150 #分辨率
        plt.plot(fpr, tpr, color='orange', label='ROC_'+ model)
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve, AUC: {:3f}'.format(auc))
        plt.legend()
        plt.axis([0, 1, 0, 1])
        plt.show()

    def test(self, X, Y):
        self.predict(X)
        self.confusionMatrix(Y)
        self.plot_roc_curve(X, Y)
        self.metric.calc_metric(self.name, Y, self.Y_Pred, self.ConfusionMatrix, self.auc_score)
        self.save()

class Metrics():
    def __init__(self):
        self.model = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        # self.specificity = []
        self.f1 = []
        self.ConfusionMatrix = []
        self.AUROC =[]
        self.result = {}
    def calc_metric(self, model_name, Y , Y_Pred, ConfusionMatrix, auc):
        self.model += [model_name]
        self.accuracy += [accuracy_score(Y, Y_Pred)]
        self.precision += [precision_score(Y, Y_Pred)]
        self.recall += [recall_score(Y, Y_Pred)]
        # self.specificity += []
        self.f1 += [f1_score(Y, Y_Pred)]
        self.ConfusionMatrix += [ConfusionMatrix]
        self.AUROC += [auc]
    def show_results(self):
        
        self.result = {
            'model':self.model,
            'accuracy':self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            # 'specificity': self.specificity,
            'f1': self.f1,
            'confusion matrix':self.ConfusionMatrix,
            'AUROC': self.AUROC
        }
        pd.DataFrame(self.result)
        
        
class WorkFlow():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.model_list = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def addModel(self, model):
        self.model_list += [model]
    def fitAllModel(self):
        for model in self.model_list:
            model.fit(self.x_train, self.y_train)
    def testAllModel(self):
        for model in self.model_list:
            model.test(self.x_test, self.y_test)
if __name__ == '__main__':
    
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
