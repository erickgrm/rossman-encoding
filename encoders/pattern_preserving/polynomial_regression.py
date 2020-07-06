""" Multivariate polynomial regression  

    @author github.com/erickgrm
""" 
try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
except:
    raise Exception('Scikit-Learn 0.22.2+ unavailable')

class PolynomialRegression(LinearRegression):

    def __init__(self, max_degree=1, interaction=False):
        super().__init__()
        self.max_degree = max_degree
        self.interaction = interaction
        self.poly = PolynomialFeatures(self.max_degree, interaction_only=self.interaction)

    def fit(self, X, y):
        return super(PolynomialRegression, self).fit(self.poly.fit_transform(X),y)

    def predict(self, X):
            return super(PolynomialRegression, self).predict(self.poly.fit_transform(X))

         
""" Univariate polynomial regression with non-zero constant term and 
    zero even terms

    @author github.com/erickgrm
"""
import numpy as np

class OddDegPolynomialRegression():

    def __init__(self, max_degree=3):
        self.max_degree = max_degree
        self.terms = [int(2*x+1) for x in range(int(np.ceil(self.max_degree/2)))]
        self.terms.insert(0,0)
        self.model = np.polynomial.polynomial.Polynomial
        self.coef = np.array([])

    def fit(self, X, y):
        # Prepare data for numpy format
        X = np.concatenate(X)
        y = np.concatenate(y.values)

        self.model = np.polynomial.polynomial.Polynomial.fit(X, y, self.terms)
        self.coef = self.model.convert().coef

    def predict(self, X):
        return self.model(X)
