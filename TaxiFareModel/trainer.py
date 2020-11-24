from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from TaxiFareModel.data import get_data, clean_data



class Trainer():
    
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder(sparse=False, handle_unknown='ignore'))

        
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preproc = ColumnTransformer([('time', pipe_time, time_cols),
                                        ('distance', pipe_distance, dist_cols)])

        self.pipeline = Pipeline([('preproc', preproc),
                    ('regressor', RandomForestRegressor())])
    
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        
        self.pipeline = self.run()
        y_pred = self.pipeline.predict(X_test)
        return np.sqrt(((y_pred - y_test)**2).mean())


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df)

    # set X and y
    X = df.drop(columns=['fare_amount'])
    y = df['fare_amount']

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    
    # train
    train = Trainer(X_train, y_train)
    train.set_pipeline()
    train.run()

    # evaluate
    score = Trainer(X_train, y_train).evaluate(X_test, y_test)