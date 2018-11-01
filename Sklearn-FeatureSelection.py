import pandas as pd
import numpy as np
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import Ridge

# data = pd.read_csv("data.csv")
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

def SelectBest():
    # Feature extraction
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)

    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)

    features = fit.transform(X)
    # Summarize selected features
    print(features[0:5,:])
    #Output:
    #First Row is the names encoded

def Logistic_Regression():
    # Feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))

def ridgeFunc():
    ridge = Ridge(alpha=1.0)
    ridge.fit(X,Y)
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
    print("Ridge model:", pretty_print_coefs(ridge.coef_))

def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

#SelectBest()
Logistic_Regression()
#ridgeFunc()