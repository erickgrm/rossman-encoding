""" 
    Implementation of several Pattern Preserving Encoders. For details see: 
    "On the encoding of  categorical variables for Machine Learning applications", Ch 3

    author github.com/erickgrm
"""
# Required libraries
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import collections

# Clerical
from .utilities import *
from .encoder import Encoder

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

from multiprocessing import Pool, Process, cpu_count

class SimplePPEncoder(Encoder):
    """ Samples randomly 600 sets of codes (can be changed with self.sampling_size), 
        encodes with best found 
    """

    def __init__(self, estimator_name='PolynomialRegression', num_predictors=2):
        """ estimator_name -> any of the dict_estimators keys
            num_predictors -> how many predictors to use for each regression
        """
        super(SimplePPEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = num_predictors
        self.codes = {}

        self.sample_size = 600
        self.history = []
        self.threshold = 0.01
        self.df = None
        self.num_categories = 0

    def fit(self, df, _):
        self.codes = {}

        # Scale numerical variables to [0,1]
        df = scale_df(df)
        self.df = df
        self.num_categories = num_categorical_instances(df)

        pool = Pool(cpu_count()-1)
        self.history = pool.map(self.new_soc, range(self.sample_size))
        pool.close()
        
        categories = categorical_instances(df)
        self.codes = dict(zip(categories, min(self.history, key=lambda x: x.fitness).codes))

    def new_soc(self, i):
        soc = SetOfCodes()
        soc.codes = np.random.uniform(0,1,self.num_categories)
        soc.fitness = evaluate(soc.codes, self.df, self.estimator, self.num_predictors)
        return soc

    def plot_history(self):
        plot(self.history)


class AgingPPEncoder(Encoder):
    """ Samples sets of codes accorgding to a simplified genetic algorithm, called Aging Evolution
        and that only allows mutation and deletes oldest individual at each iteration.

        Completes 800 iterations (can be changed with self.cycles)
    """

    def __init__(self, estimator_name='LinearRegression', num_predictors=2, cycles=800):
        """ estimator_name -> any of the dict_estimators keys
            num_predictors -> how many predictors to use for each regression
        """
        super(AgingPPEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = num_predictors
        self.codes = {}

        self.cycles = cycles # How many codes will be sampled, instead of generations
        self.threshold = 0.01
        self.prob_mutation = 0.25
        self.size_population = 25
        self.sample_size = int(self.size_population/4)
        self.history = []

    def fit(self, df, _):
        self.codes = {} # cleans up the codes when fitting the same instance for different datasets

        population = self.aging_algorithm(df)
        categories = categorical_instances(df)
        self.codes = dict(zip(categories, min(population, key=lambda x: x.fitness).codes))
        
    def aging_algorithm(self, df):
        k = num_categorical_instances(df)

        population = collections.deque()

        # Initialise population with random individuals
        while len(population) < self.size_population: 
            soc = SetOfCodes()
            soc.codes = np.random.rand(k)
            soc.fitness = evaluate(soc.codes, df, self.estimator, self.num_predictors)
            population.append(soc)
            self.history.append(soc) 

        min_fitness = min(self.history, key=lambda x:x.fitness).fitness

        while len(self.history) < self.cycles and self.threshold < min_fitness:
            sample_inds = np.random.randint(0, self.size_population, self.sample_size)
            sample = [population[i] for i in sample_inds]
            
            parent = min(sample, key=lambda i: i.fitness)

            child = SetOfCodes()
            child.codes = self.mutate(parent.codes)
            child.fitness = evaluate(child.codes, df, self.estimator, self.num_predictors)
            population.append(child)
            self.history.append(child)
            
            population.popleft()

        return population
            
    def mutate(self, codes):
        k = len(codes)
        for i in range(k):
            if np.random.uniform(1) < self.prob_mutation:
                codes[i] = np.random.uniform(1)
        return codes

    def plot_history(self):
        plot(self.history)

class GeneticPPEncoder(Encoder):
    """ Samples sets of codes according to the Eclectic Genetic Algorithm.  
        Completes 80 generations of a population of size 20
    """

    def __init__(self, estimator_name='LinearRegression', num_predictors=1):
        """ estimator_name -> any of the dict_estimators keys
            num_predictors -> how many predictors to use for each regression
        """
        super(GeneticPPEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = num_predictors
        self.codes = {}

        self.generations = 80 # How many generations the GA will run for
        self.threshold = 0.01
        self.size_population = 20
        self.rate_mutation = 0.10
        self.history = []

    def fit(self, df, _):
        # Clean up the codes when fitting the same instance for different datasets
        self.codes = {}        
        
        # Call the genetic algorithm
        population = self.EGA(df)

        # Set the final set of codes
        categories = categorical_instances(df)
        self.codes = dict(zip(categories, population[0].codes))
        
    def EGA(self, df):
        """ Implementation of the Eclectic Genetic Algorithm
        """
        k = num_categorical_instances(df)

        population = []
        self.history = []

        # Initialise population with random individuals
        while len(population) < self.size_population: 
            soc = SetOfCodes()
            soc.codes = np.random.rand(k)
            soc.fitness = evaluate(soc.codes, df, self.estimator, self.num_predictors)
            ordered_insert(population, soc)
            self.history.append(soc)

        min_fitness = population[0].fitness

        G = 0
        # Evolution of the population
        while G  < self.generations and  self.threshold < min_fitness:
            population = self.crossover_population(df, population)
            self.history += population
            population = self.mutate_population(df, population)
            self.history += population
            min_fitness = population[0].fitness
            G += 1

        return population

    def crossover(self, codes1, codes2):
        """ Routine for the (anular) crossover of two individuals
        """
        k = len(codes1)
        while True:
            i = np.random.randint(1,k)
            j = np.random.randint(1,k)
            if i < j:
                break
        new_codes1 = np.concatenate((codes1[:i], codes2[i:j], codes1[j:]), axis=0)
        new_codes2 = np.concatenate((codes2[:i], codes1[i:j], codes2[j:]), axis=0)

        return new_codes1, new_codes2
            
    def crossover_population(self, df, population):
        """ Routine for the crossover of the population, by pairing individuals
        """
        new_population = population
        n = self.size_population
        for i in range(int(n/2)):
            parent1 = population[i]
            parent2 = population[n-i-1]
            child1 = SetOfCodes()
            child2 = SetOfCodes()
            child1.codes, child2.codes = self.crossover(parent1.codes, parent2.codes)
            child1.fitness = evaluate(child1.codes, df, self.estimator, self.num_predictors)
            child2.fitness = evaluate(child2.codes, df, self.estimator, self.num_predictors)

            self.history.append(child1)
            self.history.append(child2)

            ordered_insert(new_population, child1)
            ordered_insert(new_population, child2)

        return new_population[:self.size_population]

    def mutate(self, codes):
        """ Routine to perform mutation on an individual
        """
        k = len(codes)
        for i in range(k):
            if np.random.uniform(1) < self.rate_mutation:
                codes[i] = np.random.uniform(1)
        return codes

    def mutate_population(self, df, population):
        """ Routine to perform mutation on the population
        """
        new_population = population
        m = int(self.rate_mutation*self.size_population)
        inds = np.random.randint(0,self.size_population, m)
        for i in inds:
            child = SetOfCodes()
            child.codes = self.mutate(population[i].codes)
            child.fitness = evaluate(child.codes, df, self.estimator, self.num_predictors)
            self.history.append(child)
            ordered_insert(new_population, child)

        return new_population[:self.size_population]

    def plot_history(self):
        plot(self.history)

class NaivePPEncoder(Encoder):
    """ Samples randomly 500 sets of codes for each categorical variable in a sequential manner 
        Encodes with best found 
    """
    def __init__(self, estimator_name='LinearRegression', num_predictors=1):
        """ Allows any of the estimators names in dict_estimators
        """
        super(NaivePPEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = num_predictors
        self.sampling_size = 500


    def fit(self, df, _):
        # make sure we can take num_predictors variables as predictors
        if len(df.columns)-1 < self.num_predictors:
            self.num_predictors = len(df.columns)-1

        for x in df.columns.values:
            if is_categorical(df[x]):
                errors = {}
                for i in range(self.sampling_size):
                    seed = np.random.randint(0,10000)
                    target = seeded_random_encoding_of_variable(df[x], seed)
                    errors[seed] = regression(df.drop([x],axis=1), target, self.estimator, self.num_predictors)
                best_seed = min(errors, key=lambda k: errors[k])

                np.random.seed(best_seed)
                xcodes = np.random.uniform(0,1,len(np.unique(df[x]))) 

                self.codes.update(zip(np.unique(df[x]), xcodes))
