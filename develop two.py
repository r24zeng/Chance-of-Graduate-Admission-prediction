import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.neural_network import MLPRegressor
from vecstack import stacking, StackingTransformer
from sklearn.svm import SVR
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,mean_squared_error, r2_score
from sklearn import ensemble
import matplotlib.pylab as pyl
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings('ignore')

result={'estimator':[],'MSE':[],'R2':[]}


def model(estimator, X_train, Y_train, X_test, Y_test):
    model=estimator.fit(X_train, Y_train)
    print("Parameters of model :\n",model)
#    print("Weights of GRE,  TOEFL,   UniversityRating, SOP, LOR, CGPA, Research:\n",estimator.coef_)
    predict = model.predict(X_test)
    print("Predict:\n", predict)

    train_acc = model.score(X_train, Y_train)
    print('Training data accuracy in model is:', train_acc)
    test_acc = model.score(X_test, Y_test)
    print('Test data accuracy in model is', test_acc)
    MSE = mean_squared_error(Y_test, predict)
    print('Mean squared error of test is :', MSE)

    result['MSE'].append(MSE)

    return predict

def plot_result(weights,data_columns ):
    # plots weights of every features
    weights = pd.Series(weights, index=data_columns)
    weights.nlargest(10).plot(kind='barh')
    plt.show()


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


# analysis basic data information
dataset = pd.read_csv('/Users/ZR/Desktop/ECE657/individual project/Admission_Predict_Ver1.1.csv')
pd.set_option('display.width',None)

# analyses basic information of training data
print(dataset.head())
print(dataset.columns)
dataset = dataset.rename(columns={'Serial No.':'SerialNo', 'GRE Score':'GREScore', 'TOEFL Score':'TOEFLScore', 'University Rating':'UniversityRating','LOR ':'LOR','Chance of Admit ':'ChanceOfAdmit'})
print(dataset.info())
print(dataset.describe(include='all'))



# remove useless column and formalize data logistic regression algorithm needed
data_train=dataset.drop('SerialNo',axis=1)
data_train = data_train.drop('ChanceOfAdmit', axis=1)
#dataX = dataX.drop('Research', axis=1)
dataY = dataset['ChanceOfAdmit']
print(dataset.corr())
print(type(data_train))



# normalize data
sc=MinMaxScaler()
sc.fit(data_train)
dataX=sc.transform(data_train)
dataX=pd.DataFrame(dataX,columns=data_train.columns)
print(type(dataX))
print(dataX.head())

threshold=0.73
class_dataY = np.where(dataY>threshold,1,0)

# separate  data to train and verification set
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)

class_y_train = np.where(y_train>threshold,1,0)
class_y_test = np.where(y_test>threshold,1,0)
#########################################
## Training Model and show result ##
# start linear regression algorithm
print("***************LinearRegression****************")
estimator= LinearRegression()
predict=model(estimator, x_train, y_train, x_test, y_test)
cv_result(estimator,dataX, dataY)
plot_result(estimator.coef_, data_train.columns)
plot_learning_curve(estimator, "Linear Regression Learning Curve", dataX, dataY)

result['estimator'].append('LR')

# # grid search to select the best parameters
# feature_package = [['GREScore', 'TOEFLScore', 'SOP', 'UniversityRating'],
#                    ['GREScore', 'CGPA', 'LOR', 'UniversityRating'],
#                    ['CGPA', 'LOR', 'SOP', 'Research'],
#                    ['GREScore','TOEFLScore', 'SOP', 'LOR', 'CGPA', 'UniversityRating']]
#
# for features in feature_package:
#     print(features)
#     print("*" * 50)
#
#     dataX_set = dataX[features]
#     r2 = cross_val_score(estimator, dataX_set, dataY, cv=10, scoring='r2')
#     print("Accuracy: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))

#########################################
# start Ridge Regression
print("***************RidgeRegression****************")
estimator= Ridge(alpha=0.1, random_state=2)
predict=model(estimator, x_train, y_train, x_test, y_test)
cv_result(estimator,dataX, dataY)
plot_result(estimator.coef_, data_train.columns)
plot_learning_curve(estimator, "Ridge Regression Learning Curve", dataX, dataY)

# grid search to select the best parameters
param_grid={'alpha':np.arange(0.1, 1, 0.2)}
gs = GridSearchCV(estimator, param_grid, cv=10, scoring='r2')
gs.fit(x_train, y_train)
print('best parameters are:', gs.best_params_)
print('best score is:', gs.best_score_)

result['estimator'].append('Ridge')

#########################################
# start Multi-layer Perceptron Regression
print("***************MLPRegression****************")
estimator = MLPRegressor(hidden_layer_sizes=(8,), activation='tanh',solver='lbfgs',random_state=3)
predict=model(estimator, x_train, y_train, x_test, y_test)
cv_result(estimator,dataX, dataY)
result['estimator'].append('MLP')

# grid search to select the best parameters
param_grid={'hidden_layer_sizes':np.arange(1,10)}
gs = GridSearchCV(estimator, param_grid, cv=10, scoring='r2')
gs.fit(x_train, y_train)
print('best parameters are:', gs.best_params_)
print('best score is:', gs.best_score_)

# plot_result(estimator.coefs_, data_train.columns)
plot_learning_curve(estimator, "Ridge Regression Learning Curve", dataX, dataY)

#########################################
# start Random Forest Regression
print("***************RandomForestRegression****************")
estimator = RandomForestRegressor(n_estimators=20,criterion='mse')
predict=model(estimator, x_train, y_train, x_test, y_test)
cv_result(estimator,dataX, dataY)
print(estimator.feature_importances_)
plot_result(estimator.feature_importances_, data_train.columns)
result['estimator'].append('RF')
plot_learning_curve(estimator, "RandomForestRegression Learning Curve", dataX, dataY)

#########################################
## ensemble three models ##
## stacking ##
print("***************Stacking****************")
models = [LinearRegression(),MLPRegressor(hidden_layer_sizes=(8,), activation='tanh',solver='lbfgs'),
          Ridge(alpha=0.1),RandomForestRegressor(n_estimators=20),GradientBoostingRegressor(n_estimators=20),AdaBoostRegressor(Ridge(alpha=0.1))]
S_train, S_test = stacking(models, x_train, y_train, x_test, regression=True, verbose=2)
estimator = Ridge(alpha=0.1)
S_predict=model(estimator, S_train, y_train, S_test, y_test)
S = np.concatenate((S_train,S_test),axis=0)
# cv_result(estimator, S, dataY)
#r2 = cross_val_score(estimator, S_train, dataY , cv=10, scoring='r2')
r2 = r2_score(y_test, S_predict, multioutput='variance_weighted')
result['R2'].append(r2.mean())
result['estimator'].append('ST')


plot_learning_curve(estimator, "Stacking Learning Curve", dataX, class_dataY)



# #####################
# ### weighted average
# print("***************average weight****************")
# model1= LinearRegression()
# model1.fit(x_train,y_train)
# predict1=model1.predict(x_test)
# model2= MLPRegressor(hidden_layer_sizes=(7,), activation='tanh',solver='lbfgs')
# model2.fit(x_train,y_train)
# predict2=model1.predict(x_test)
# model3= RandomForestRegressor(n_estimators=10)
# model3.fit(x_train,y_train)
# predict3=model1.predict(x_test)
#
# finalpred=(predict1*0.6+predict2*0.3+predict3*0.1)
# r2m=r2_score(y_test,finalpred)
# print("r2m=",r2m)

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

