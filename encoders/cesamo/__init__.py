""" Implementation of the CESAMO Encoder. For details see 
    "On the encoding of categorical variables for machine learning applications", Chapter 3

    @author github.com/erickgrm
"""
# Libraries providing estimators
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.svm import SVR
from .polynomial_regression import PolynomialRegression, OddDegPolynomialRegression
from sklearn.neural_network import MLPRegressor

dict_estimators = {}
dict_estimators['LinearRegression'] = LinearRegression()
dict_estimators['SGDRegressor'] = SGDRegressor(loss='squared_loss')
dict_estimators['SVR'] = SVR()
dict_estimators['PolynomialRegression'] = PolynomialRegression(max_degree=3)
dict_estimators['Perceptron'] = MLPRegressor(max_iter=150, hidden_layer_sizes=(10,5))
dict_estimators['CESAMORegression'] = OddDegPolynomialRegression(max_degree=11)

from .utilities import *
from .encoder import *
import numpy as np
from numpy.random import uniform
from scipy.stats import shapiro, normaltest
import seaborn as sns
import matplotlib.pyplot as plt
colours = ['blue', 'yellow']
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

class CESAMOEncoder(Encoder):

    def __init__(self, estimator_name='CESAMORegression', plot=False):
        """ Allows any of the keys in dict_estimators as estimator_name
        """
        super(CESAMOEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = 1
        self.codes = {}
        self.plot_flag = plot

    def fit(self, df, _):
        if not(isinstance(df,(pd.DataFrame))):
            try:
                df = pd.DataFrame(df)
            except:
                raise Exception("Cannot convert to pandas.DataFrame")
        #Scale numerical variables to [0,1]
        df = scale_df(df)
         
        df_copy = df.copy()
        cat_cols = categorical_cols(df_copy)
        
        # Find codes variable by variable
        for x in cat_cols:
                df_copy, codes_x = self.encode_var(df_copy, x, self.estimator, self.num_predictors)
                self.codes.update(codes_x)

    def encode_var(self, df, col_num, estimator, num_predictors):
        # Ensure y has the correct type
        y = pd.DataFrame(df[col_num].values)
        X = df.drop(df.columns[col_num], axis=1)

        codes_var = {}
        normality = False
        while not(normality):
            # Propose new set of codes for y
            y_codes = uniform(0,1,len(np.unique(y)))
            y_enc = replace_in_df(y,dict(zip(np.unique(y), y_codes)))
            
            # Choose secondary variable; if categorical, encode with random numbers 
            i = np.random.choice(list(X.columns))
            if is_categorical(X[i]):
                i_codes = dict(zip(np.unique(X[i]), uniform(0,1,len(np.unique(X[i])))))
                X_i = replace_in_df(X[i], i_codes).values.reshape(-1,1).copy()
            else:
                X_i = X[i].values.reshape(-1,1).copy()

            # Fit the estimator and get error 
            estimator.fit(X_i, y_enc)
            error = mean_squared_error(y_enc, estimator.predict(X_i))
            
            # Append error and codes
            codes_var[error] = y_codes
            
            # Update normality flag; at least 19 samples are required by the 
            # D'angostino-Pearson normality test
            if len(codes_var) > 19:
                normality = self.normality_test(list(codes_var.keys())) or (800 <= len(codes_var))
                
        # Plot the distribution of errors if plotting flag=True
        if self.plot_flag:
            if len(codes_var) == 800:
                print('max number of codes (800) sampled for the variable', col_num)
            else: 
                print(len(codes_var), ' codes sampled for the  variable', col_num)
            self.plot_errors(list(codes_var.keys()), col_num)

        # Choose codes corresponding to minimum error and replace in df
        y_final_codes = dict(zip(np.unique(y), codes_var[min(codes_var, key=lambda k:k)]))
        df[col_num] = replace_in_df(y, y_final_codes)

        return df, y_final_codes

    def normality_test(self, observations, alpha=0.05):
        # Perform D’Agostino and Pearson’s normality test
        stat, p = normaltest(observations)

        #stat, p = shapiro(observations) # Another possible test

        if p > alpha:
            return True  # The observations come from a normal variable
        else:
            return False # The observations do not come from a normal variable

    def plot_errors(self, errors, col_num):
        sns.set_style("whitegrid")
        plt.figure(figsize=(6,4))
        sns.distplot(errors, label='Variable '+str(col_num))
        plt.legend()
        plt.show()

