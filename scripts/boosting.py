import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

train = pd.read_csv('../data/train_processed.csv')
test = pd.read_csv('../data/test_processed.csv')


class XGBoostRegressor():
    def __init__(self, num_boost_round=100, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        print self.params

    def fit(self, X, y,):
        dtrain = xgb.DMatrix(X, y)
        print self.params
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round)

    def predict(self, X):
        y = self.predict_proba(X)
        return y

    def predict_proba(self, X):
        dX = xgb.DMatrix(X)
        return self.clf.predict(dX)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return r2_score(y,Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def create_features(n_comp = 10):

    # PCA
    pca = PCA(n_components=n_comp, random_state=42)
    pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
    pca2_results_test = pca.transform(test)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=42)
    ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
    ica2_results_test = ica.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]



def tune_boosting():
    y = train["y"]
    y_mean = np.mean(y)
    X = train.drop(["y"], axis=1)
    print X.shape
    print y.shape

    parameters = {
        'max_depth': [4, 5, 6],
    }

    clf = XGBoostRegressor(
        objective ='reg:linear',
        num_boost_round=100,
        eval_metric='rmse',
        eta=0.001,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=1.0,
        silent=0,
    )

    clf = RandomizedSearchCV(clf, parameters, n_jobs=1,cv=3,n_iter=2)
    clf.fit(X,y)
    print clf.best_params_
    print clf.best_score_


# def run_boosting():
#
#     y_train = train["y"]
#     y_mean = np.mean(y_train)
#     xgb_params['base_score'] = y_mean
#     # form DMatrices for Xgboost training
#     dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
#     dtest = xgb.DMatrix(test)
#     cv_folds = 5
#     early_stopping_rounds = 5
#     # xgboost, cross-validation
#     cv_result = xgb.cv(xgb_params,
#                        dtrain,
#                        num_boost_round=700,  # increase to have better results (~700)
#                        early_stopping_rounds=50,
#                        verbose_eval=50,
#                        show_stdv=False,
#                        nfold= cv_folds
#                        )
#     print cv_result
#
#     #num_boost_rounds = int( (len(cv_result) - early_stopping_rounds) / (1- 1/cv_folds))
#     num_boost_rounds=len(cv_result)
#     # train model
#     model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#     return model, dtrain, dtest


create_features(12)
#model, dtrain, dtest = run_boosting()

# print(r2_score(dtrain.get_label(), model.predict(dtrain)))
# y_pred = model.predict(dtest)
# output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
# output.to_csv('../results/xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)

tune_boosting()