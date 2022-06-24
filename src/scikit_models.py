# AUTOGENERATED! DO NOT EDIT! File to edit: _notebooks/2022-06-13-Preprocessing.ipynb (unless otherwise specified).

__all__ = ['load_mangoes', 'train_test_split', 'X_y_cat', 'SNV', 'MSC', 'SavGol', 'cross_validate', 'evaluate',
           'TqdmCallback', 'Optimiser', 'PLSRegression', 'load_mangoes', 'train_test_split', 'X_y_cat',
           'cross_validate', 'evaluate', 'TqdmCallback', 'Optimiser', 'PLSRegression', 'SNV', 'MSC', 'SavGol']

# Cell
import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

np.random.seed(123)

import warnings
warnings.filterwarnings('ignore')

# Cell
def load_mangoes():
    mangoes = pd.read_csv("../../datasets/mangoes/mangoes_raw.csv")

    unique_spectra = mangoes['DM'].unique()
    fruit_id_dict = {u:i for i,u in enumerate(unique_spectra)}
    mangoes['Fruit_ID'] = mangoes['DM'].apply(lambda x: fruit_id_dict[x])
    return mangoes

def train_test_split(data):
    train_inds = np.logical_not(data['Set']=='Val Ext')
    test_inds = data['Set']=='Val Ext'

    train_data = data[train_inds]
    test_data = data[test_inds]

    return train_data, test_data


def X_y_cat(data,min_X=285,max_X=1200):
    cat_vars=['Set','Season','Region','Date','Type','Cultivar','Pop','Temp','Fruit_ID']
    y_vars = ['DM']
    X_vars = [i for i in data.columns if (not i in y_vars) and (not i in cat_vars)]
    X_vars = [i for i in X_vars if (int(i)>= min_X) and (int(i)<= max_X)]
    return data[X_vars], data[y_vars], data[cat_vars]

# Cell
from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin, BaseEstimator
class SNV(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X, copy=None):
        return scale(X,axis=1)

# Cell
class MSC(TransformerMixin, BaseEstimator):

    sample_means = None

    def fit(self, X, y=None, sample_weight=None):
        self.sample_means = np.mean(X,axis=0)
        return self

    def transform(self, X, copy=None):
        X_msc = np.zeros_like(X.values)
        for i in range(X.shape[0]):

            fit = np.polyfit(self.sample_means, X.values[i,:], 1, full=True)
            X_msc[i,:] = (X.values[i,:] - fit[0][1]) / fit[0][0]

        return X_msc

# Cell
from scipy.signal import savgol_filter

class SavGol(TransformerMixin, BaseEstimator):

    def __init__(self, window_length=10, polyorder=2, deriv=0,mode='interp'):
        self.window_length=window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.mode=mode

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X, copy=None):
        return savgol_filter(X,self.window_length, self.polyorder,deriv=self.deriv,mode=self.mode)

# Cell

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold

def cross_validate(model,X,y,splitter=GroupKFold(),groups=None,plot=False,save_loc=None):
    preds = None
    ys = None
    for fold, (inds1,inds2) in enumerate(splitter.split(X,y,groups)):
        model.fit(X.iloc[inds1,:],y.iloc[inds1,:])
        pred = model.predict(X.iloc[inds2,:])

        if preds is None:
            preds = pred
            ys = y.iloc[inds2,:]
        else:
            preds = np.concatenate((preds,pred),axis=0)
            ys = np.concatenate((ys,y.iloc[inds2,:]),axis=0)

    r2 = r2_score(ys,preds)
    mse = mean_squared_error(ys,preds)


    if plot:
        ys = ys.flatten()
        preds = preds.flatten()

        m, b = np.polyfit(ys, preds, 1)
        fig, ax = plt.subplots()

        ls = np.linspace(min(ys),max(ys))
        ax.plot(ls,ls*m+b,color = "black", label = r"$\hat{y}$ = "+f"{m:.4f}y + {b:.4f}")
        ax.scatter(x=ys,y=preds,label = r"$R^2$" + f"={r2:.4f}")

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend(bbox_to_anchor=(0.5,1))
        if not save_loc is None:
            fig.savefig(save_loc)
    return mse

def evaluate(model,train_X,train_y,test_X,test_y,plot=False,save_loc=None,log=True):
    test_y=test_y.values.flatten()
    model.fit(train_X,train_y)
    preds = model.predict(test_X)

    r2 = r2_score(test_y,preds)
    mse = mean_squared_error(test_y,preds)

    if log:
        print(f"Test set MSE: {mse:.4f}")

    if plot:
        preds=preds.flatten()

        m, b = np.polyfit(test_y, preds, 1)
        fig, ax = plt.subplots()

        ls = np.linspace(min(test_y),max(test_y))
        ax.plot(ls,ls*m+b,color = "black", label = r"$\hat{y}$ = "+f"{m:.4f}y + {b:.4f}")
        ax.scatter(x=test_y,y=preds,label = r"$R^2$" + f"={r2:.4f}")

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend(bbox_to_anchor=(0.5,1))
        if not save_loc is None:
            fig.savefig(save_loc)
    return model, mse


# Cell
from tqdm.notebook import tqdm
from codetiming import Timer

from skopt import gp_minimize,dump
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback

class TqdmCallback(tqdm):

    def __call__(self, res):
        super().update()

    def __getstate__(self):
        return []
    def __setstate__(self, state):
        pass

class Optimiser():

    def __init__(self,space,model,X,y,splitter=KFold(),groups=None):
        self.space=space
        self.model=model
        self.X=X
        self.y=y
        self.splitter=splitter
        self.groups=groups


    def objective(self,**params):
        self.model.set_params(**params)
        return cross_validate(self.model, self.X, self.y, splitter=self.splitter,groups=self.groups)

    def bayesian_optimise(self,n_calls=50,random_state=0):
        obj = use_named_args(self.space)(self.objective)
        return gp_minimize(obj,self.space,n_calls=n_calls,random_state=random_state,callback=TqdmCallback(total=n_calls))

    def random_optimise(self,n_calls=50, random_state=0):
        obj = use_named_args(self.space)(self.objective)
        return gp_minimize(obj,self.space,n_calls=n_calls,n_initial_points=n_calls,initial_point_generator='random',
                             random_state=random_state,callback=TqdmCallback(total=n_calls))

    def grid_optimise(self,n_calls=50, random_state=0):
        obj = use_named_args(self.space)(self.objective)
        return gp_minimize(obj,self.space,n_calls=n_calls,n_initial_points=n_calls,initial_point_generator='grid',
                             random_state=random_state,callback=TqdmCallback(total=n_calls))


    @Timer()
    def optimise(self,strategy="bayesian",n_calls=50, random_state=0,plot=True,save_file=None,log=True):


        if strategy=="bayesian":
            result = self.bayesian_optimise(n_calls=n_calls, random_state=random_state)
        elif strategy=="grid":
            result = self.grid_optimise(n_calls=n_calls, random_state=random_state)
        elif strategy=="random":
            result = self.random_optimise(n_calls=n_calls, random_state=random_state)

        #set parameters of model
        params = {dim.name:result['x'][i] for i,dim in enumerate(self.space)}
        model = self.model.set_params(**params)

        #save model and search
        if not save_file is None:
            del result.specs['args']['func'] #spaghetti code to not throw an error as the objective function is unserialisable
            dump(result,save_file+'_search.pkl')
            dump(model, save_file+'_model.joblib')

        #plot
        if plot:
            plot_convergence(result)

        # log/print results and include a regression plot:
        if log:
            print(f'Best model had an MSE of {result.fun:.4f}')
            print(f'Setting parameters as: {params}')

            if save_file is None:
                cross_validate(model, self.X,self.y,splitter=self.splitter,groups=self.groups,plot=True,save_loc=None)
            else:
                cross_validate(model, self.X,self.y,splitter=self.splitter,groups=self.groups,plot=True,save_loc=save_file+'_plot.png')

        return model,result



# Cell
from sklearn.cross_decomposition import PLSRegression as PLS_

class PLSRegression(PLS_):

    def transform(self,X,y=None,copy=True):
        X = super().transform(X,copy=copy)
        return X

# Cell
#export
import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

np.random.seed(123)

import warnings
warnings.filterwarnings('ignore')

# Cell
def load_mangoes():
    mangoes = pd.read_csv("../data/mangoes_raw.csv")

    unique_spectra = mangoes['DM'].unique()
    fruit_id_dict = {u:i for i,u in enumerate(unique_spectra)}
    mangoes['Fruit_ID'] = mangoes['DM'].apply(lambda x: fruit_id_dict[x])
    return mangoes

def train_test_split(data):
    train_inds = np.logical_not(data['Set']=='Val Ext')
    test_inds = data['Set']=='Val Ext'

    train_data = data[train_inds]
    test_data = data[test_inds]

    return train_data, test_data


def X_y_cat(data,min_X=285,max_X=1200):
    cat_vars=['Set','Season','Region','Date','Type','Cultivar','Pop','Temp','Fruit_ID']
    y_vars = ['DM']
    X_vars = [i for i in data.columns if (not i in y_vars) and (not i in cat_vars)]
    X_vars = [i for i in X_vars if (int(i)>= min_X) and (int(i)<= max_X)]
    return data[X_vars], data[y_vars], data[cat_vars]

# Cell

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold

def cross_validate(model,X,y,splitter=GroupKFold(),groups=None,plot=False,save_loc=None):
    preds = None
    ys = None
    for fold, (inds1,inds2) in enumerate(splitter.split(X,y,groups)):
        model.fit(X.iloc[inds1,:],y.iloc[inds1,:])
        pred = model.predict(X.iloc[inds2,:])

        if preds is None:
            preds = pred
            ys = y.iloc[inds2,:]
        else:
            preds = np.concatenate((preds,pred),axis=0)
            ys = np.concatenate((ys,y.iloc[inds2,:]),axis=0)

    r2 = r2_score(ys,preds)
    mse = mean_squared_error(ys,preds)


    if plot:
        ys = ys.flatten()
        preds = preds.flatten()

        m, b = np.polyfit(ys, preds, 1)
        fig, ax = plt.subplots()

        ls = np.linspace(min(ys),max(ys))
        ax.plot(ls,ls*m+b,color = "black", label = r"$\hat{y}$ = "+f"{m:.4f}y + {b:.4f}")
        ax.scatter(x=ys,y=preds,label = r"$R^2$" + f"={r2:.4f}")

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend(bbox_to_anchor=(0.5,1))
        if not save_loc is None:
            fig.savefig(save_loc)
    return mse

def evaluate(model,train_X,train_y,test_X,test_y,plot=False,save_loc=None,log=True):
    test_y=test_y.values.flatten()
    model.fit(train_X,train_y)
    preds = model.predict(test_X)

    r2 = r2_score(test_y,preds)
    mse = mean_squared_error(test_y,preds)

    if log:
        print(f"Test set MSE: {mse:.4f}")

    if plot:
        preds=preds.flatten()

        m, b = np.polyfit(test_y, preds, 1)
        fig, ax = plt.subplots()

        ls = np.linspace(min(test_y),max(test_y))
        ax.plot(ls,ls*m+b,color = "black", label = r"$\hat{y}$ = "+f"{m:.4f}y + {b:.4f}")
        ax.scatter(x=test_y,y=preds,label = r"$R^2$" + f"={r2:.4f}")

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend(bbox_to_anchor=(0.5,1))
        if not save_loc is None:
            fig.savefig(save_loc)
    return model, mse


# Cell
from tqdm import tqdm
from codetiming import Timer

from skopt import gp_minimize,dump
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback

class TqdmCallback(tqdm):

    def __call__(self, res):
        super().update()

    def __getstate__(self):
        return []
    def __setstate__(self, state):
        pass

class Optimiser():

    def __init__(self,space,model,X,y,splitter=KFold(),groups=None):
        self.space=space
        self.model=model
        self.X=X
        self.y=y
        self.splitter=splitter
        self.groups=groups


    def objective(self,**params):
        self.model.set_params(**params)
        return cross_validate(self.model, self.X, self.y, splitter=self.splitter,groups=self.groups)

    def bayesian_optimise(self,n_calls=50,random_state=0):
        obj = use_named_args(self.space)(self.objective)
        return gp_minimize(obj,self.space,n_calls=n_calls,random_state=random_state,callback=TqdmCallback(total=n_calls))

    def random_optimise(self,n_calls=50, random_state=0):
        obj = use_named_args(self.space)(self.objective)
        return gp_minimize(obj,self.space,n_calls=n_calls,n_initial_points=n_calls,initial_point_generator='random',
                             random_state=random_state,callback=TqdmCallback(total=n_calls))

    def grid_optimise(self,n_calls=50, random_state=0):
        obj = use_named_args(self.space)(self.objective)
        return gp_minimize(obj,self.space,n_calls=n_calls,n_initial_points=n_calls,initial_point_generator='grid',
                             random_state=random_state,callback=TqdmCallback(total=n_calls))


    @Timer()
    def optimise(self,strategy="bayesian",n_calls=50, random_state=0,plot=True,save_file=None,log=True):


        if strategy=="bayesian":
            result = self.bayesian_optimise(n_calls=n_calls, random_state=random_state)
        elif strategy=="grid":
            result = self.grid_optimise(n_calls=n_calls, random_state=random_state)
        elif strategy=="random":
            result = self.random_optimise(n_calls=n_calls, random_state=random_state)

        #set parameters of model
        params = {dim.name:result['x'][i] for i,dim in enumerate(self.space)}
        model = self.model.set_params(**params)

        #save model and search
        if not save_file is None:
            del result.specs['args']['func'] #spaghetti code to not throw an error as the objective function is unserialisable
            dump(result,save_file+'_search.pkl')
            dump(model, save_file+'_model.joblib')

        #plot
        if plot:
            plot_convergence(result)

        # log/print results and include a regression plot:
        if log:
            print(f'Best model had an MSE of {result.fun:.4f}')
            print(f'Setting parameters as: {params}')

            if save_file is None:
                cross_validate(model, self.X,self.y,splitter=self.splitter,groups=self.groups,plot=True,save_loc=None)
            else:
                cross_validate(model, self.X,self.y,splitter=self.splitter,groups=self.groups,plot=True,save_loc=save_file+'_plot.png')

        return model,result



# Cell
from sklearn.cross_decomposition import PLSRegression as PLS_

class PLSRegression(PLS_):

    def transform(self,X,y=None,copy=True):
        X = super().transform(X,copy=copy)
        return X

# Cell
from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin, BaseEstimator
class SNV(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X, copy=None):
        return scale(X,axis=1)

# Cell
class MSC(TransformerMixin, BaseEstimator):

    sample_means = None

    def fit(self, X, y=None, sample_weight=None):
        self.sample_means = np.mean(X,axis=0)
        return self

    def transform(self, X, copy=None):
        X_msc = np.zeros_like(X.values)
        for i in range(X.shape[0]):

            fit = np.polyfit(self.sample_means, X.values[i,:], 1, full=True)
            X_msc[i,:] = (X.values[i,:] - fit[0][1]) / fit[0][0]

        return X_msc

# Cell
from scipy.signal import savgol_filter

class SavGol(TransformerMixin, BaseEstimator):

    def __init__(self, window_length=10, polyorder=2, deriv=0,mode='interp'):
        self.window_length=window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.mode=mode

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X, copy=None):
        return savgol_filter(X,self.window_length, self.polyorder,deriv=self.deriv,mode=self.mode)