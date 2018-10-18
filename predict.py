import numpy as np
import pandas as pd
import calendar
from sklearn.externals import joblib

# load the model and scaler from disk
_meta = joblib.load('gbr_model.pkl')
_scaler = _meta[0]
_model = _meta[1]

def _catagories():

    '''
    Utility function creates a series object with necessary 
    encoding for catagorical variables.
    '''
    
    months = {'month_'+month.lower(): 0 for month in list( calendar.month_abbr[1:] )}
    days = {'day_'+day.lower(): 0 for day in list( calendar.day_abbr )}

    return pd.Series({**months, **days})

class ForestFire():

    '''
    ForestFire class takes a fire instance as a pandas.Series object and
    predicts the burned area given an input sklearn model and an sklearn
    scaler function. The default scaler is a fitted 
    sklearn.preprocessing.StandardScaler instance. The defualt model is
    a trained sklearn.ensemble.GradientBoostingRegressor instance.
    '''
    
    def __init__(self, fire, scaler = _scaler, model = _model):

        '''
        Inputs:
        
        fire - pandas.Series object containing a forest fire instance.
        scaler - sklearn.preprocessing.StandardScaler or other scaler from sklearn.preprocessing
        model - sklearn trained model with a .predict() method
        '''
        
        self.scaler = scaler
        self.model = model
        catagories = self._encode(fire)
        quantities = self._scale(fire) 
        
        fire = quantities.join(catagories)
        self.fire = fire
        
    def _encode(self, fire):

        '''
        Does the encoding of catagorical variables
        '''

        catagories = _catagories()
        month = fire['month']
        day = fire['day']
        catagories['month_' + month] = 1
        catagories['day_' + day] = 1
        return catagories.to_frame().T

    def _scale(self, fire):

        '''
        scales the input data
        '''

        quantities = fire.drop(['month', 'day'])
        quantities = quantities.to_frame().T
        columns = quantities.columns
        quantities = self.scaler.transform(quantities)
        return pd.DataFrame(data = quantities, columns = columns)

    def predict(self):
        
        '''
        predicts the burned area
        '''
        
        return _model.predict(self.fire).squeeze()
    
if __name__ == "__main__":

    # example usage
    
    data = pd.read_csv('forestfires.csv')

    fire = ForestFire(data.iloc[0].drop('area'))
    print( fire.predict() )
