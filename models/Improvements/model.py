import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from math import log10, floor
import numpy as np
import datetime
import seaborn as sns
from scipy import stats
import pandas as pd
import glob
import tqdm
import os
import random
import pickle
import copy 
from xgrid_utils import calc_spatial_integral

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from keras.regularizers import L1L2

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.optimizers import Adam

def plot_timeseries(ypred, ytrue, daterange, marker='x', ax=None):
    """Plot timeseries with red and green fill to show over/under prediction"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(daterange, ytrue, label='Observed', color='k', ls='-', lw=1, marker=marker, markersize=4)
    ax.plot(daterange, ypred, label='Predicted', color='dimgrey', lw=1.5, marker=marker, markersize=4)
    ax.fill_between(daterange, ytrue, ypred, where=(ypred<=ytrue), facecolor='red', alpha=0.3, interpolate=True)
    ax.fill_between(daterange, ytrue, ypred, where=(ypred>=ytrue), facecolor='green', alpha=0.3, interpolate=True)
    plt.legend()
    return ax

def rmspe(y_true, y_pred):
    """Simple percent root mean square error"""
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100

def mape(y_true, y_pred):
    """Simple mean absolute percent error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_trainable_params(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

class Spatial:
    def __init__(self):
        self.extent_scaler = StandardScaler()
        self.input_shape = None
        self.num_timesteps = None

    def process_data(self, 
                     spatial_ds,
                     extents,
                     num_timesteps=3,
                     deseasonalize_type=2,
                     test_size=0.2):

        # Add timesteps to gridded
        ds_timesteps = spatial_ds.rolling(time=num_timesteps).construct('timesteps')
        X_grid = np.array(ds_timesteps.to_array())
        X_grid = X_grid.transpose((1, 4, 2, 3, 0))
        X_grid = np.nan_to_num(X_grid)

        self.input_shape = (num_timesteps, *X_grid.shape[2:])
        self.num_timesteps = num_timesteps

        # Silly 
        y_extent = extents  # np.expand_dims(extents, -1)

        if deseasonalize_type == 'all':
            y_extent, dates = self.deseasonalize(y_extent, ds_timesteps.time)
            y_extent = self.detrend(y_extent, dates)
            raise NotImplementedError('Sry')

        else:
            y_extent, dates, climatologies = self.deseasonalize_recent(y_extent, ds_timesteps.time)

        # NOTE: We must assume deseasonalize_type * 12 > num_timesteps. Otherwise modify this code.
        
        # Find year corresponding to a split at (test_size)% 
        split_year = dates[int(len(y_extent) * (1 - test_size))].dt.year.values.item()

        # Find index of the first month of that year, at which we will split
        split_index = np.argwhere(np.array(dates.dt.year).flatten() > split_year)[0][0]

        # Split the dates using the split_index
        dates_train = dates[:split_index]
        dates_test = dates[split_index:]
        climatologies_train = climatologies[:split_index]
        climatologies_test = climatologies[split_index:]

        # Split y extents using the same index
        y_extent_train = y_extent[:split_index]
        y_extent_test = y_extent[split_index:]

        # Standardize y_extent
        y_extent_train = self.extent_scaler.fit_transform(y_extent_train.reshape(-1, 1)).flatten()
        y_extent_test = self.extent_scaler.transform(y_extent_test.reshape(-1, 1)).flatten()

        # Create X extents from the standardized y_extents
        X_extent_train = self.create_timesteps(y_extent_train, num_timesteps).T[:-1]
        X_extent_test = self.create_timesteps(y_extent_test, num_timesteps).T[:-1]

        # Split X gridded
        # Remove first few years of X_grid data coinciding with the number of timesteps (assumes data starts on January of first year)
        if deseasonalize_type != 'all':
            # Train:
            # Start index is the number of years skipped due to deseasonalizing
            # End index is the split index PLUS the number of years skipped MINUS the number of timesteps in X 
            X_grid_train = X_grid[deseasonalize_type * 12:]
            X_grid_train = X_grid_train[: split_index - num_timesteps]

            # Test:
            # Start index is the split index PLUS the number of years skipped due to deseasonalizing
            # End index is the number of timesteps in X (from the end)
            X_grid_test = X_grid[split_index + deseasonalize_type * 12: -num_timesteps]
        else:
            raise NotImplementedError('TODO')
            # X_grid_train = X_grid[: num_timesteps + split_index - 1]
            # X_grid_test = X_grid[num_timesteps + split_index: -1]

        # Since we are predicting y_extent from (num_timesteps) on
        y_extent_train = y_extent_train[num_timesteps:]
        y_extent_test = y_extent_test[num_timesteps:]
        dates_train = dates_train[num_timesteps:]
        dates_test = dates_test[num_timesteps:]
        climatologies_train = climatologies_train[num_timesteps:]
        climatologies_test = climatologies_test[num_timesteps:]

        data = dict(
            X_grid_test=X_grid_test,
            X_grid_train=X_grid_train,
            X_extent_train=X_extent_train,
            X_extent_test=X_extent_test,
            y_extent_train=y_extent_train,
            y_extent_test=y_extent_test,
            dates_train=dates_train,
            dates_test=dates_test,
            climatologies_test=climatologies_test,
            climatologies_train=climatologies_train,
        )

        return data

    @staticmethod
    def create_timesteps(arr, num_timesteps=3):
        timesteps = [arr[:-(num_timesteps - 1)]]
        
        for i in range(1, num_timesteps - 1):
            timesteps.append(arr[i:-((num_timesteps-1)-i)])
                    
        timesteps.append(arr[(num_timesteps - 1):])
        return np.array(timesteps)

    @staticmethod
    def deseasonalize(y, dates):
        """Deseasonalize y by subtracting the monthly means"""
        y = xr.DataArray(y, dims=['time'], coords={'time': dates})  # Create time-aware array
        climatologies = y.groupby('time.month').mean()  # Get monthly means
        y = (y.groupby('time.month') - climatologies).values  # Subtract monthly means
        return y

    @staticmethod
    def deseasonalize_recent(y, dates, past_n_years=1, return_climatologies=True):
        """Deseasonalize y by subtracting the monthly means of the previous n years"""
        y = xr.DataArray(y, dims=['time'], coords={'time': dates})  # Create time-aware array

        # Group y by year and store years
        y_years = y.groupby('time.year')
        years = sorted(y_years.groups.keys())

        # Loop over each year, starting at (past_n_years), deseasonalize, and append
        # i.e. for part_n_years, use the first two to deseasonalize the third, and so on
        #
        # [xx+oooooooo]  
        # [oxx+ooooooo]
        # [ooxx+oooooo]
        # where xx are used to find climatology, and + is being deseasonalized

        y_deseasonalized = []  # List of deseasonalized yearly data
        climatologies = []  # List of climatologies to store for later inversing
        for i in range(past_n_years, len(years)):
            clim_year = xr.concat(
                [y_years[year] for year in years[i - past_n_years: i]],
                dim='time').groupby('time.month').mean()
            climatologies.append(clim_year)
            y_deseasonalized.append(y_years[years[i]].groupby('time.month') - clim_year)

        climatologies = np.hstack([clim.values for clim in climatologies])

        y_deseasonalized = xr.concat(y_deseasonalized, dim='time')
        dates = y_deseasonalized.time
        y = np.array(y_deseasonalized)

        if return_climatologies:
            return y, dates, climatologies
        else:
            return y, dates

    @staticmethod
    def detrend(y, dates):
        """Remove a linear trend from y"""
        slope, intercept, r, p, se = stats.linregress(dates.astype(int), list(y))
        y = np.array(y - (intercept + slope * dates.astype(int)))
        return y

    def create_model(self,
                     num_convlstm,
                     convlstm_filters,
                     convlstm_kernels,
                     convlstm_dropout,
                     convlstm_rec_dropout,
                     convlstm_kernel_reg,
                     batchnorm,
                     num_conv2d,
                     conv2d_filters,
                     conv2d_kernels,
                     max_pooling,
                     num_lstm,
                     lstm_units,
                     lstm_dropout,
                     lstm_kernel_reg,
                     learning_rate,
                     GAP
                     ):
        inp = layers.Input(shape=self.input_shape)
        extent_inp = layers.Input(shape=(self.num_timesteps, 1))

        x_spatial = inp
        for i in range(num_convlstm):
            x_spatial = layers.ConvLSTM2D(
                filters=convlstm_filters[i],
                kernel_size=convlstm_kernels[i],
                padding="valid",
                return_sequences=True if i < num_convlstm - 1 else False,
                activation="relu",
                dropout=convlstm_dropout,
                recurrent_dropout=convlstm_rec_dropout,
                kernel_regularizer=convlstm_kernel_reg,
            )(x_spatial)
            if batchnorm:
                x_spatial = layers.BatchNormalization()(x_spatial)

        for i in range(num_conv2d):
            x_spatial = layers.Conv2D(filters=conv2d_filters[i],
                                kernel_size=conv2d_kernels[i],
                                activation="relu",
                                padding="valid",
                                data_format='channels_last')(x_spatial)
            
            if max_pooling and i == 2:
                x_spatial = layers.MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last')(x_spatial)

        if GAP:
            x_spatial = layers.GlobalAveragePooling2D()(x_spatial)
        else:
            shape = [tf.shape(x_spatial)[k] for k in range(4)]
            x_spatial = tf.reshape(x_spatial, [shape[0], shape[1]*shape[2]*shape[3]])  # Flattens

        # LSTMify the input extents
        x_extent = extent_inp
        for i in range(num_lstm):
            x_extent = layers.LSTM(
                units=lstm_units[i],
                return_sequences=True if i < num_lstm - 1 else False,
                dropout=lstm_dropout,
                kernel_regularizer=lstm_kernel_reg,
            )(x_extent)

        x = layers.Concatenate(axis=-1)([x_spatial, x_extent])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(1)(x)

        # Next, we will build the complete model and compile it.
        self.model = keras.models.Model([inp, extent_inp], x)
        # print(self.model.summary())
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
        
        
    def train(self, data, save_dir, model_name, epochs=100, batch_size=10, verbose=0):
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

        self.checkpoint_filepath = save_dir + model_name + '.checkpoint'

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        # Hyperparameters
        history = self.model.fit(
            [data['X_grid_train'], data['X_extent_train']],
            data['y_extent_train'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[[data['X_grid_test'], data['X_extent_test']], data['y_extent_test']],
            verbose=verbose,
            callbacks=[early_stopping, reduce_lr, model_checkpoint_callback]
        )

        self.history = history.history

    def reseasonalize(self, y, climatologies):
        return self.extent_scaler.inverse_transform(y.flatten().reshape(-1, 1)).flatten() + climatologies

    def save(self, model_name, save_dir):
        filename = save_dir + model_name

        # Save model separately 
        self.model.save(filename + '_model.p')

        del self.model

        # Save without model
        with open(filename + '.p', 'wb') as f:
            pickle.dump(self, f)

    def load(self, model_name, save_dir, load_model=False):
        filename = save_dir + model_name

        # Load object without model
        with open(filename + '.p', 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)

        # Load model and add as attribute to self
        if load_model:
            self.model = keras.models.load_model(filename + '_model.p')