# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.impute import KNNImputer
from tabulate import tabulate
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
    preprocessor.preprocess(pca_n_components=5)
    metric = Metrics()
    workflow_1 = WorkFlow(preprocessor.x_train, preprocessor.y_train,
                          preprocessor.x_test, preprocessor.y_test, preprocessor.steps)
    workflow_1.addModel(Model(RandomForestClassifier(), metric))
    workflow_1.addModel(Model(DecisionTreeClassifier(), metric))
    workflow_1.addModel(Model(KNeighborsClassifier(n_neighbors=3), metric))
    workflow_1.fitAllModel()
    workflow_1.testAllModel()
    metric.show_results()
    metric.plotROC()

    preprocessor2 = DataPreprocessor(
        generater.X, generater.Y, fillNansMethod='mean', imbalanceMethod='SMOTE', pcaMethod=True)
    preprocessor2.preprocess(pca_n_components=8)
    metric2 = Metrics()
    workflow_1 = WorkFlow(preprocessor2.x_train, preprocessor2.y_train,
                          preprocessor2.x_test, preprocessor2.y_test, preprocessor2.steps)
    workflow_1.addModel(Model(RandomForestClassifier(), metric2))
    workflow_1.addModel(Model(DecisionTreeClassifier(), metric2))
    workflow_1.addModel(Model(KNeighborsClassifier(n_neighbors=3), metric2))
    workflow_1.fitAllModel()
    workflow_1.testAllModel()

    metric2.show_results()
    metric2.plotROC()


class DataGenerater():
    """Provides raw data

    Length of file from Xpath should equal to the file's from Ypath, or else, errors will happend.
    BE CAREFUL! If files don't exist, or other errors happend, sample data will be auto-generated.

    Attributes:
        Xpath: Input csv files include features.
        Ypath: Input csv files include ground truth.
    """

    def __init__(self, Xpath=r'./', Ypath=r'./'):
        try:
            print('Loading Data')
            self.X = pd.read_csv(Xpath)
            self.Y = pd.read_csv(Ypath)
        except BaseException:
            print("Loading failure, use default dataset")
            self.X = pd.DataFrame([[random.uniform(0, 9) for i in range(10)] for j in range(
                10000)] + [[random.uniform(1, 10) for i in range(10)] for j in range(1000)])
            self.X.columns = [str(i) for i in range(10)]
            self.Y = pd.DataFrame(
                [0 for i in range(10000)] + [1 for i in range(1000)])

    def __len__(self):

        return len(self.Y)


class DataPreprocessor():
    """Preprocesses raw data

    Preprocess the raw data with filling nans, resample the imbalanced dataset and pca.

    Attributes:
        X: Input csv files include features.
        Y: Input csv files include ground truth.
        fillNansMethod(str): used to control which methods of filling nan will be used,
        imbalanceMethod(str): used to control which methods of processing imblanced data will be used,
        pcaMethod(boolean): used to control if Principal Component Analysis (PCA) will be applied on data.
        steps: making a dict with above 3 parameters. It will be passed to a metric Object.
    """

    def __init__(self, X, Y, fillNansMethod='None',
                 imbalanceMethod='None', pcaMethod=False):
        self.X = X
        self.Y = Y
        self.fillNansMethod = fillNansMethod
        self.imbalanceMethod = imbalanceMethod
        self.pcaMethod = pcaMethod
        self.steps = {'fillNansMethod': fillNansMethod,
                      'imbalanceMethod': imbalanceMethod,
                      'pca': pcaMethod}

    """Fills Nans in dataframe

    Fills Nans in pd.DataFrame with some given algoriths. These algos directly
    modified the data without returning variables.

    Args:
        method: a string to control which algo will be used.
        example:'mean', 'linear', 'zero', 'forward', 'backward', 'kNN'
    Returns:
        None
    """

    def fillNans(self, method='None'):
        if method == 'mean':
            self.X = self.X.fillna(self.X.mean())
        elif method == 'linear':
            self.X = self.X.interpolate(method='linear')
        elif method == 'zero':
            self.X = self.X.fillna(0)
        elif method == 'forward':
            self.X = self.X.fillna(method='ffill')
        elif method == 'kNN':
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            tmp = imputer.fit_transform(self.X)
            self.X.data = tmp
        elif method == 'backward':
            self.X = self.X.fillna(method='bfill')

    """Dealing with imbalanced data

    Imbalanced data will lead to accuracy paradox in test state. That is,
    The accuracy of testing will extremely high with low sensitivity or low
    specificity. It will make the model tend to predict input data
    as dominant output. These algos directly
    modified the data without returning variables.

    Args:
        method: a string to control which algo will be used.
        example:'RandomOverSample', 'ADASYN', 'SMOTE'
    Returns:
        None
    """

    def imbalanceProcess(self, method='None'):
        if method == 'RandomOverSample':
            ros = RandomOverSampler(random_state=999)
            self.x_train, self.y_train = ros.fit_resample(
                self.x_train, self.y_train)
        if method == 'ADASYN':
            ada = ADASYN(random_state=999)
            self.x_train, self.y_train = ada.fit_resample(
                self.x_train, self.y_train)
        if method == 'SMOTE':
            sm = SMOTE(random_state=999)
            self.x_train, self.y_train = sm.fit_resample(
                self.x_train, self.y_train)

    def normalize(self):
        self.X = normalize(self.X)
    """Dimension reduction

    Curse of dimensionality happened when input dataset has lots of dimension. It means
    this dataset may be sparse and it will take lots of times for models to fit and predict.
    What's worse is that, the metrics might get lower as the dimension increasing. PCA
    can be used to find new independent features with normalization and coordinate projection.

    Args:
        n_components(int): the top 'n_components' components which used to generate a new dataset.
    Returns:
        None
    """

    def pca(self, n_components):
        pca = PCA(n_components)
        pca.fit(self.x_train)
        self.pca_x_train = pca.transform(self.x_train)
        self.pca_x_test = pca.transform(self.x_test)
        self.pca_y_train, self.pca_y_test = self.y_train, self.y_test

    """Splits training set and testing set
    Generate training features, training ground truth, testing features, and testing ground truth

    Args:
        shuffle(bool): random sort the dataset,
        test_size(float): 0~1, the ratio of training set and testing set
    Returns:
        None
    """

    def split(self):
        np.random.seed(999)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, shuffle=True, test_size=0.25)
        # np.random.seed(999)
        # self.pca_x_train, self.pca_x_test, self.pca_y_train, self.pca_y_test = train_test_split(self.pca_X, self.Y, shuffle = True, test_size = 0.25)

    def preprocess(self, pca_n_components=5):
        print('Preprocessing Data')

        self.fillNans(method=self.fillNansMethod)
        self.split()
        self.imbalanceProcess(method=self.imbalanceMethod)

        if self.pcaMethod:
            self.pca(pca_n_components)


class Model():
    """Provides raw data

    Length of file from Xpath should equal to the file's from Ypath, or else, errors will happend.
    BE CAREFUL! If files don't exist, or other errors happend, sample data will be auto-generated.

    Attributes:
        model: a scikit-learn classification model instance.
        metric: a metric instance to evaluate and visualize the results from model.
        name: name of model
    Raise:
        If input model aren't from scikit-learn or it doesn't have functions like fit, predict,
        predict_proba, it will raise exception. THe solution is to create a new model class inherit
        this class and overwrite those function with adapter pattern.
    """

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
        plt.rcParams['savefig.dpi'] = 150  # 图片像素
        plt.rcParams['figure.dpi'] = 150  # 分辨率
        plt.plot(fpr, tpr, color='orange', label='ROC_' + model)
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve, AUC: {:3f}'.format(auc))
        plt.legend()
        plt.axis([0, 1, 0, 1])
        plt.show()

    def test(self, X, Y, steps):
        self.steps = steps
        self.predict(X)
        self.confusionMatrix(Y)
        self.plot_roc_curve(X, Y)
        self.metric.calc_metric(self.name, Y, self.Y_Pred,
                                self.ConfusionMatrix, self.auc_score, self.steps)
        self.metric.fpr_tprs += [self.fpr_tpr]
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
        self.AUROC = []
        self.fillNansMethod = []
        self.imbalanceMethod = []
        self.pcaMethod = []
        self.fpr_tprs = []
        self.result = {}

    def calc_metric(self, model_name, Y, Y_Pred, ConfusionMatrix, auc, steps):
        self.model += [model_name]
        self.accuracy += [accuracy_score(Y, Y_Pred)]
        self.precision += [precision_score(Y, Y_Pred)]
        self.recall += [recall_score(Y, Y_Pred)]
        # self.specificity += []
        self.f1 += [f1_score(Y, Y_Pred)]
        self.ConfusionMatrix += [ConfusionMatrix]
        self.AUROC += [auc]
        self.fillNansMethod += [steps['fillNansMethod']]
        self.imbalanceMethod += [steps['imbalanceMethod']]
        self.pcaMethod += [steps['pca']]

    def plotROC(self):
        plt.figure()
        for idx, fpr_tpr in enumerate(self.fpr_tprs):
            fpr, tpr = fpr_tpr[0], fpr_tpr[1]
            plt.plot(fpr, tpr, label=self.model[idx])
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.axis([0, 1, 0, 1])
        plt.savefig('ROC_PCA.png')
        plt.show()

    def show_results(self):

        self.result = {
            'model': self.model,
            'fillNansMethod': self.fillNansMethod,
            'imbalanceMethod': self.imbalanceMethod,
            'pcaMethod': self.pcaMethod,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            # 'specificity': self.specificity,
            'f1': self.f1,
            'confusion matrix': self.ConfusionMatrix,
            'AUROC': self.AUROC
        }
        # print(pd.DataFrame(self.result))
        print(
            tabulate(
                pd.DataFrame(
                    self.result),
                headers='keys',
                tablefmt='psql'))


class WorkFlow():
    """Define the functions which need to be applied on each model objects

    Observer pattern, addModel needs to be called before fitAllModel and testAllModel
    addModel->fitAllModel->testAllModel
    Attributes:
        addModel: get models and add them into a list
        fitAllModel: fit data for all model in list.
        testAllModel: test data for all model in list
    """

    def __init__(self, x_train, y_train, x_test, y_test, steps):
        self.model_list = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.steps = steps

    def addModel(self, model):
        self.model_list += [model]

    def fitAllModel(self):
        for model in self.model_list:
            model.fit(self.x_train, self.y_train)

    def testAllModel(self):
        for model in self.model_list:
            model.test(self.x_test, self.y_test, self.steps)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
