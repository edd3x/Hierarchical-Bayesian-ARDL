"""
@author: Edward Salakpi
Function for training HBMs
"""

import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import HBM_Factory_AEZ
import pymc3.sampling_jax
import warnings
warnings.filterwarnings('ignore')
print(f"Running on PyMC3 v{pm.__version__}")

county100 = [
'Marsabit',
'Garissa',
'Baringo',
'Kajiado',
'Machakos',
'Nandi',
'Mandera',
'Kericho',
'Makueni',
'Samburu',
'Nyeri'
]

county103 = [
'Marsabit',
'Kitui',
'Baringo',
'Tana-River',
'Turkana',
'Narok'
]

county101 = [
'Marsabit',
'Kitui',
'Baringo',
'Tana-River',
'Turkana',
'Narok',
'West-Pokot',
'Wajir',
'Mandera',
'Samburu',
'Laikipia',
'Kajiado',
'Isiolo',
'Taita-Taveta',
'Garissa',
'Nyeri'
]

target = 'VCI'
groups = 'AEZs'
level = '3M'


path = '/mnt/lustre/projects/DISCUS/ASTROCAST/Research_Data'
azdataFiles = glob.glob(path+f'/LC_Extracts/AEZ_{level}/s2*pft*csv')

dataFiles_Sub = []
for cs in county101:
    [dataFiles_Sub.append(m) for m in azdataFiles if cs in m]

dataFiles_Up = []
for cs in county103:
    [dataFiles_Up.append(m) for m in azdataFiles if cs in m]

def runPostSample(type=None, horizons=None):
    """Model params:
    type: str
        'PartPooled', 'Pooled', 'Unpooled'
    """
    ############################### PartialPooled Model #####################################
    if type == 'PartPooled':
        for h in horizons:
            print(f'{type} for {h} Weeks ahead')
            print(path+f'/LC_Extracts/h_models/f_horizon_{groups}_{type}_VCI_{h}_{level}_AEZNew.nc')
            model_trace = az.from_netcdf(path+f'/LC_Extracts/h_models/f_horizon_{groups}_{type}_VCI_{h}_{level}_AEZNew.nc')


            subpredsAnom0 = []
            subPredsArray = []
            for testfile in dataFiles_Sub:
                test_df1, county_name = HBM_Factory_AEZ.testSet(testfile)
                preds, predArray = HBM_Factory_AEZ.testModel(testDF=test_df1, county=county_name, trace=model_trace, horizon=h,
                                              target='VCI', hgroups='AEZs', anom=True, sampler='JAX1', detrend=True)


                subpredsAnom0.append(preds)
                subPredsArray.append(predArray)

            pd.concat(subpredsAnom0).to_csv(path+f'/LC_Extracts/Forecast_Output/AEZJAX0_{type}_{h}_weeks_New.csv')
            np.save(path+f'/LC_Extracts/Forecast_Output/AEZJAX0_{type}_{h}_weeks_New.npy', np.concatenate((subPredsArray), axis=1))


    ############################### Pooled Model #####################################
    if type == 'Pooled':
        for h in horizons:
            print(f'{type} for {h} Weeks ahead')
            print(path+f'/LC_Extracts/h_models/f_horizon_{groups}_{type}_VCI_{h}_{level}_AEZNew.nc')
            model_trace = az.from_netcdf(path+f'/LC_Extracts/h_models/f_horizon_{groups}_{type}_VCI_{h}_{level}_AEZNew.nc')


            subpredsAnom0 = []
            subPredsArray = []
            for testfile in dataFiles_Sub:
                test_df1, county_name = HBM_Factory_AEZ.testSet(testfile)
                preds, predArray = HBM_Factory_AEZ.testModel(testDF=test_df1, county=county_name, trace=model_trace, horizon=h,
                                              target='VCI', hgroups='AEZs', anom=True, sampler='JAX0', detrend=True)

                subpredsAnom0.append(preds)
                subPredsArray.append(predArray)

            pd.concat(subpredsAnom0).to_csv(path+f'/LC_Extracts/Forecast_Output/AEZJAX0_{type}_{h}_weeks_POnly.csv')
            np.save(path+f'/LC_Extracts/Forecast_Output/LCJAX_{type}_{h}_weeks.npy', np.concatenate((subPredsArray), axis=1))


    ############################### UnPooled Model #####################################
    if type == 'Unpooled':

        for testfile in dataFiles_Up:
            test_df1, county_name = HBM_Factory_AEZ.testSet(testfile)
            subpredsAnom0 = []
            subPredsArray = []
            for h in horizons:
                print(f'{county_name} {type} for {h} Weeks ahead')
                print(path+f'/LC_Extracts/h_models/f_horizon_{county_name}_{groups}_{type}_VCI_{h}_{level}_AEZNew.nc')
                model_trace = az.from_netcdf(path+f'/LC_Extracts/h_models/f_horizon_{county_name}_{groups}_{type}_VCI_{h}_{level}_AEZNew.nc')


                preds, predArray = HBM_Factory_AEZ.testModel(testDF=test_df1, county=county_name, trace=model_trace, horizon=h,
                                                  target='VCI', hgroups='AEZs', anom=True, sampler='JAX0', detrend=True)

                subpredsAnom0.append(preds)
                subPredsArray.append(predArray)

            pd.concat(subpredsAnom0).to_csv(path+f'/LC_Extracts/Forecast_Output/AEZJAX_{type}_{county_name}_weeks_AEZNew.csv')
            np.save(path+f'/LC_Extracts/Forecast_Output/AEZJAX_{type}_{county_name}_weeks_AEZNew.npy', np.concatenate((subPredsArray), axis=1))
