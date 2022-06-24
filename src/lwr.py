from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.utils import check_array
import warnings

__all__ = ['LocalWeightedRegression']


class LocalWeightedRegression(Ridge):
    """
    Implementation of a locally weighted regression,
    """

    def __init__(self,n_neighbours=5,alpha = 1e-2,fit_intercept=True, copy_X=False,
                 n_jobs=None,floor=False,kernal=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.alpha = alpha
        self.floor = floor

        self.n_neighbours = n_neighbours
        self.kneighbours = None
        self._X = None
        self._y = None
       
        self.kernal=kernal
        super().__init__(alpha=alpha)
        

    def fit(self,X,y):
        self._X = X
        self._y = np.asarray(y)
        nrow,ncol=X.shape
        self.kneighbours = LWKNeighborsRegressor(n_neighbors=min(self.n_neighbours,nrow))
        self.kneighbours.fit(X,y)

    def transform(self,X,y=None):
        pass

    def predict(self,X):
        nrow, ncol = X.shape
        preds = []
        distances, indices = self.kneighbours.predict(X)

        for i in range(0,nrow):
            X_fit = self._X[indices[i]]
            y_fit = self._y[indices[i]]
            to_pred = X[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                super().fit(X_fit, y_fit)
                pred = super().predict(to_pred.reshape(1,len(to_pred)))[0]
                if self.floor:
                    pred = max(0,pred)
                if self.kernal:
                    if pred < min(y_fit):
                        pred = min(y_fit)
                    elif pred> max(y_fit):
                        pred = max(y_fit)
                    
            preds.append(pred)
        return np.asarray(preds)

    def score(self,x,y):
        pass

    def from_state(self,state):
        pass
    
    def state(self):
        #state_dict = {'ridge':self.ridge,
        #             'ridge_regression_param':self.ridge_regression_param,
        #             'kneighbours_state':self.kneighbours.state()}
        return {}
        return state_dict

    def reset(self):
        self.kneighbours = LWKNeighborsRegressor(n_neighbors=n_neighbours) 


class LWKNeighborsRegressor(KNeighborsRegressor):
    """
    Helper class for the locally weighted regressiom
    """

    def __init__(self, n_neighbors=5, *, weights='uniform',
               algorithm='auto', leaf_size=30,
               p=2, metric='minkowski', metric_params=None, n_jobs=None,
               **kwargs):
        super().__init__(
          n_neighbors=n_neighbors,
          algorithm=algorithm,
          leaf_size=leaf_size, metric=metric, p=p,
          metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self._X = np.ndarray([])

    def predict(self, X):
        """Predict the target for the provided data
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
              or (n_queries, n_indexed) if metric == 'precomputed'
          Test samples.
        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
          Target values.
        """
        #prexisiting sklearn code to access indices
        X = check_array(X, accept_sparse='csr')
        neigh_dist, neigh_ind = self.kneighbors(X)

        return neigh_dist, neigh_ind
    
    def transform(self,X,y=None):
        pass
    
    def score(X,y):
        pass

    
    def reset(self):
        pass
    
    def state(self):
        state_dict = {
                      'n_neighbors':self.n_neighbors,
                      '_X': self._X.tolist()
    
        }
        return state_dict
    
    def from_state(self,state):
        pass

from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator

class LocallyWeightedRegression(RegressorMixin,BaseEstimator):
    
    def __init__(self,n_neighbours=50,alpha=1.0):
        self.n_neighbours=n_neighbours
        self.alpha = alpha
        self._X=None
        self._y=None
        self.sample=None
        
    def fit(self, X, y, sample_weight=None):
        self._X = X
        self._y = y
        self.sample_weight=None
        
    def predict(self,X):

        preds = []
        for _,x in X.iterrows():
            x = np.asarray(x.tolist())
            #x1 = np.asarray([x for _ in range(len(self._X))])
            distances = np.linalg.norm(x-self._X.values,axis=1) #calculate distances

            
            inds = [i for i in range(len(distances))]
            inds = [j for _, j in sorted(zip(distances, inds))]#sort indices by distances
            subset_X = self._X.iloc[inds[0:self.n_neighbours],:].values #take n_neighbours
            subset_y = self._y.iloc[inds[0:self.n_neighbours],:]
            
            ridge = Ridge(self.alpha)
            ridge.fit(subset_X, subset_y)
            #threshold = sorted(distances)[self.n_neighbours]
            #weights = [1 if i <threshold else 0 for i in distances]
            #ridge.fit(self._X.values,self._y,weights)
            preds.append(ridge.predict([x])[0])
            
        return preds