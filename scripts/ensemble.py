import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('../data/train_processed.csv')
test = pd.read_csv('../data/test_processed.csv')
usable_columns = list(set(train.columns) - set(['y']))
train,validation = train_test_split(train,test_size=0.2)

class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

def generate_features(n_comp=10):
    df_merged = pd.concat([train.drop(["y"], axis=1), test])

    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd.fit(df_merged)
    tsvd_results_train = tsvd.transform(train.drop(["y"], axis=1))
    tsvd_results_test = tsvd.transform(test)

    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca.fit(df_merged)
    pca2_results_train = pca.transform(train.drop(["y"], axis=1))
    pca2_results_test = pca.transform(test)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=420)
    ica.fit(df_merged)
    ica2_results_train = ica.transform(train.drop(["y"], axis=1))
    ica2_results_test = ica.transform(test)

    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
    grp.fit(df_merged)
    grp_results_train = grp.transform(train.drop(["y"], axis=1))
    grp_results_test = grp.transform(test)

    # SRP
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
    srp.fit(df_merged)
    srp_results_train = srp.transform(train.drop(["y"], axis=1))
    srp_results_test = srp.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]

        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

        train['grp_' + str(i)] = grp_results_train[:, i - 1]
        test['grp_' + str(i)] = grp_results_test[:, i - 1]

        train['srp_' + str(i)] = srp_results_train[:, i - 1]
        test['srp_' + str(i)] = srp_results_test[:, i - 1]

def create_ensemble():
    y_train = train['y'].values
    y_mean = np.mean(y_train)
    id_test = test['ID'].values

    # finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays)
    finaltrainset = train[usable_columns].values
    finaltestset = test[usable_columns].values

    xgb_params = {
        'n_trees': 520,
        'eta': 0.0045,
        'max_depth': 6,
        'subsample': 0.90,
        'objective': 'reg:gamma',
        'eval_metric': 'rmse',
        'base_score': y_mean,  # base prediction = mean(target)
        'silent': 1
    }

    num_boost_rounds = 2000

    dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
    dtest = xgb.DMatrix(test)

    # train model
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
    y_pred = model.predict(dtest)

    '''Train the stacked models then predict the test data'''

    stacked_pipeline = make_pipeline(
        StackingEstimator(estimator=LassoLarsCV(normalize=True)),
        StackingEstimator(
            estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55,
                                                min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
        LassoLarsCV()

    )

    stacked_pipeline.fit(finaltrainset, y_train)
    results = stacked_pipeline.predict(finaltestset)

    '''R2 Score on the entire Train data when averaging'''

    print('R2 score on train data:')
    print(r2_score(y_train, stacked_pipeline.predict(finaltrainset) * 0.2855 + model.predict(dtrain) * 0.7145))

    '''Average the preditionon test data  of both models then save it on a csv file'''

    sub = pd.DataFrame()
    sub['ID'] = id_test
    sub['y'] = y_pred * 0.75 + results * 0.25
    sub.to_csv('../results/akash-ensemble.csv', index=False)



generate_features(12)
create_ensemble()

