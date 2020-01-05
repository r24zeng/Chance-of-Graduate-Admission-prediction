import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from vecstack import stacking, StackingTransformer
from sklearn.svm import SVR
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,mean_squared_error, r2_score
from sklearn import ensemble
import matplotlib.pylab as pyl
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

result={'estimator':[],'MSE':[],'R2':[]}

def cv_result(estimator,dataX, dataY):
    r2 = cross_val_score(estimator, dataX, dataY, cv=10, scoring='r2')
    print("Accuracy: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))
    result['R2'].append(r2.mean())

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=3, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()

def PolyRegression(degree):
    return Pipeline([("poly", PolynomialFeatures(degree=degree)), ("std_scalar", StandardScaler()),("lr", LinearRegression())])


# analysis basic data information
dataset = pd.read_csv('/Users/ZR/Desktop/ECE657/individual project/Admission_Predict_Ver1.1.csv')
pd.set_option('display.width',None)

# analyses basic information of training data
dataset = dataset.rename(columns={'Serial No.':'SerialNo', 'GRE Score':'GREScore', 'TOEFL Score':'TOEFLScore', 'University Rating':'UniversityRating','LOR ':'LOR','Chance of Admit ':'ChanceOfAdmit'})
print(dataset.info())
print(dataset.describe(include='all'))


# remove useless column and formalize data logistic regression algorithm needed
data_train=dataset.drop('SerialNo',axis=1)
data_train = data_train.drop('ChanceOfAdmit', axis=1)
#dataX = dataX.drop('Research', axis=1)
dataY = dataset['ChanceOfAdmit']

# normalize data
sc=MinMaxScaler()
sc.fit(data_train)
dataX=sc.transform(data_train)
dataX=pd.DataFrame(dataX,columns=data_train.columns)

# separate  data to train and verification set
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)

#########################
## start poly regression
for i in range(1, 4):

    poly_model = PolyRegression(i)
    poly_model.fit(x_train, y_train)
    y_predict = poly_model.predict(x_test)
    MSE = mean_squared_error(y_test, y_predict)
    print('Mean squared error of test is :', MSE)
    result['MSE'].append(MSE)
    print("r2 score = ", r2_score(y_test, y_predict))

    cv_result(poly_model,dataX, dataY)
    plot_learning_curve(poly_model, "PolyRegression Learning Curve", dataX, dataY)



result['estimator'].append('power=1')
result['estimator'].append('power=2')
result['estimator'].append('power=3')
# Compare MSE and Score in all models as plot
result=pd.DataFrame(result, columns=['estimator','MSE','R2'])
print(result)
label_x = result['estimator']
num_x1 = result['MSE']*100
num_x2 = result['R2']
x = list(range(len(label_x)))
total_width, n = 0.8, 2
width = total_width / n
plt.bar(label_x, num_x1, width=width, label='MSE',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_x2, width=width, label='R2',tick_label = label_x,fc = 'r')
plt.legend()
plt.show()
