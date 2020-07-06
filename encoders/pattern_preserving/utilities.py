""" Auxiliary functions for the Pattern Preserving Encoders
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class SetOfCodes(object):
    """ Object for an abstract set of codes. Consists of the codes themselves and 
        the attribute "fitness" 
    """
    def __init__(self):
        self.codes = None
        self.fitness = None

    def get_fitness(self):
        return self.fitness

    def get_codes(self):
        return self.codes

def regression(df, target, estimator, num_predictors):
    """" Picks num_predictors predictor variables at random but distinct from target and fits 
        the chosen estimator. Numerical variables must be already scaled to [0,1]
        
         Returns mean squared error
         
         If a predictor is categorical, it is first encoded with random numbers
    """
    # Relabel columns from 0 
    df.columns = range(len(df.columns)) 

    # Choose random indices for the predictor variables
    chosen_cols = []
    while len(chosen_cols) < num_predictors:
        chosen_cols = np.random.randint(0, len(df.columns), num_predictors)
        chosen_cols = np.unique(chosen_cols)

    # Pick the predictor variables
    predictors = df[chosen_cols]
    predictors = random_encoding_of_categories(predictors)

    # Fit the estimator
    estimator.fit(predictors, target)

    return mean_squared_error(target, estimator.predict(predictors))

def evaluate(codes, df, estimator, num_predictors):
    categories = categorical_instances(df)
    try:
        mapping = dict(zip(categories, codes))
    except:
        print("Not the same number of categories and codes")

    X = replace_in_df(df, mapping)
    max_error = 0
    for x in X.columns:
        error = regression(X.drop([x], axis =1), X[x], estimator, num_predictors)
        if max_error < error:
            max_error = error

    return max_error

def ordered_insert(population, soc):
    i = 0
    while i < len(population) and population[i].fitness < soc.fitness:
        i += 1
    population.insert(i,soc)
    return population

def plot(setsOfCodes):
    """ Plots the fitness of the sets of codes 
    """
    plt.figure(figsize=(16,8))
    plt.scatter(range(len(setsOfCodes)), [x.fitness for x in setsOfCodes])

def replace_in_df(df, mapping): 
    """ Replaces categories by numbers according to the mapping
        If a category is not in mapping, it gets a random code
        mapping: dictionary from categories to codes
    """
    # Ensure df has the right type
    if not(isinstance(df,(pd.DataFrame))):
        try: 
            df = pd.DataFrame(df)
        except:
            raise Exception('Cannot convert to pandas.DataFrame')
            
    cat_cols = categorical_cols(df)

    # Updates the mapping with random codes for categories not 
    # previously in the mapping
    for x in cat_cols:
            cats = np.unique(df[x])
            for x in cats:
                if not(x in mapping):
                    mapping[x] = np.random.uniform(0,1)

    return df.replace(mapping)


def scale_df(df):
    """ Scale all numerical variables to [0,1]
    """
    numerical_cols = [x for x in list(df.columns) if x not in categorical_cols(df)]
    sc = MinMaxScaler()

    for x in numerical_cols:
        if min(df[x].values) < 0.0 or 1.0 < max(df[x].values):
            df[x] = sc.fit_transform(df[x].values.reshape(-1,1))
    return df


def codes_to_dictionary(L):
    """ L: list of strings of the form str + ": " + float
        RETURNS dictionary with elements str : float
    """
    dict = {}
    for x in L:
        k, v = split_str(x)
        dict[k] = v
    return dict

def is_categorical(array):
    """ Tests if the column is categorical
    """
    return array.dtype.name == 'category' or array.dtype.name == 'object'

def categorical_cols(df): 
    """ Return the column numbers of the categorical variables in df
    """
    cols = []
    # Rename columns as numbers
    df.columns = range(len(df.columns))
    
    for x in df.columns: 
        if is_categorical(df[x]):
            cols.append(x)
    return cols

def categorical_instances(df):
    """ Returns an array with all the categorical instances in df
    """
    instances = []
    cols = categorical_cols(df)
    for x in cols:
        instances = instances + list(np.unique(df[x]))

    return instances

def num_categorical_instances(df):
    """ Returns the total number of categorical instances in df
    """
    return len(categorical_instances(df))

def random_encoding_of_categories(df):
    """ Encodes the categorical variables with random numbers in [0,1]
    """
    for x in df.columns:
        if is_categorical(df[x]):
            np.random.seed()
            k = len(np.unique(df[x]))
            codes = np.random.uniform(0,1,k)
            dictionary = dict(zip(np.unique(df[x]),codes))
            df[x] = df[x].replace(dictionary)
    return df

def seeded_random_encoding_of_variable(var,seed):
    """ Encodes the target variable with random numbers in [0,1]
        var can be a pandas DataFrame or a numpy array
        returns the encoded target as a pandas DataFrame
    """
    k = len(np.unique(var))
    np.random.seed(seed)
    codes = np.random.uniform(0,1,k)
    dictionary = dict(zip(np.unique(var),codes))

    return pd.DataFrame(var).replace(dictionary)
