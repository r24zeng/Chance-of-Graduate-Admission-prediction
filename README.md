# Chance of Graduate Admission prediction
ECE657 individual project

### Documents
+ `graphs` collectes all figures of this project
+ `Admission_Predict.csv` is the trainning dataset
+ `Admission_Predict_Ver1.1.csv` is the verification dataset
+ `main_final` is the main coding to implement algorithms and models
+ `polyRegression.py` assists to analyze models' perfermance by plotting
+ `report.pdf` views the whole process and analysis and evaluation objectively


### Database
The [dataset](https://www.kaggle.com/mohansacharya/graduate-admissions) comes from Kaggle which includes 7 variables as predictors such as GRE score, TOEFL score, University Rating( Indian undergraduate school), SOP( Statement of Purpose), LOP( Letter of recommendation strength), CGPA( out of 10), Research experience, and 1 variable as target such as Chance of Admission( ranging from 0 to 1), which were evaluated by the appliers. 

[Admission_Predict.csv](https://github.com/r24zeng/Chance-of-Graduate-Admission-prediction/blob/master/Admission_Predict.csv) containes 400 records which are used to train model, and [Admission_Predict_Ver1.1.csv](https://github.com/r24zeng/Chance-of-Graduate-Admission-prediction/blob/master/Admission_Predict_Ver1.1.csv) contains 500 records which are verification set.

### Purpose of this project
Do regression prediction for the chance of graduate admission by American universities from indian stduetns' perspectivity.

### Process of this project
1. Preprocessing data
2. Linear regression as the base model
3. Ridge regression, Multi-layer Perceptron(MLP) Regression and Random Forest regression to do furthure training
4, Ensemble model to develop the performance by emsemble strategy of stacking
5, Evaluate model perfermance by MSE and r2 score 
6, Conclusion

### Tools of this project
**Language**: Python
**Library**: Scikit-learn 
**Data processing**: Numpy, Pandas
**Plot**: Matplotlib, seaborn

### Key words:
Machine learning, Linear regression, model ensemble, undertraining and overfitting, cross verification
