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
'Turkana',
'West-Pokot',
'Wajir',
'Mandera',
'Marsabit',
'Samburu',
'Tana-River',
'Baringo',
'Laikipia',
'Narok',
'Kitui',
'Kajiado',
'Machakos',
'Kakamega',
'Isiolo',
'Taita-Taveta',
'Nandi',
'Garissa',
'Kericho',
'Makueni',
'Nyeri'
]

target = 'VCI'
groups = 'AEZs'
level = '3M'


path = '/'

aztrainDataFile = glob.glob(path+f'/AEZ_{level}/HBM_train_df_{level}_Sub_pft.csv')
azdataFiles = glob.glob(path+f'/AEZ_{level}/s2*pft*csv')

aztrainDFX = pd.read_csv(path+f'/AEZ_{level}/HBM_train_df_{level}_Sub_pft.csv')
sub_train = aztrainDFX.loc[aztrainDFX.County.isin(county103)]
print(sub_train.shape)

def runModel(type=None, samples=None, horizons=None):
    """Model types:
    type: str
        'PartPooled', 'Pooled', 'Unpooled'
    """
    ############################### PartialPooled Model #####################################
    if type == 'PartPooled':
        for h in horizons:
            print(f'AEZ {type} for {h} Weeks ahead')
            X_train, y_train, ssn_gp, cty_gp, aez_gp, means = HBM_Factory_AEZ.PrepData(sub_train, lst_p0=0,precip_p1=6,soil_p2=6,targ_q=6,
                                                                                              target=target, anom=True, growing_ssn=None,
                                                                                              f_horizon=h)

            X_sub = X_train
            print(X_sub.shape)
            X_data = X_sub.drop(['AEZ_Code_lag_0', 'Season_Code_lag_0','County_Code_lag_0', y_train.name], axis=1)
            y_data = X_sub[y_train.name]
            print(X_data.columns)

            #County Index
            aez_idx = X_sub['AEZ_Code_lag_0'].values.astype(int)
            ssn_idx = X_sub['Season_Code_lag_0'].values.astype(int)
            cty_idx = X_sub['County_Code_lag_0'].values.astype(int)

            with HBM_Factory_AEZ.HBARDL_factoryE(X_data=X_data, y_data=y_data, idx=aez_idx, hgroup=groups, sampler='JAX') as HB_Model_A:
                trace_h = pm.sampling_jax.sample_numpyro_nuts(samples, tune=samples,target_accept=0.95)
                # trace_h = pm.sample(2000, tune=5000, target_accept=0.9)

                trace_h.to_netcdf(filename=path+f'/h_models/f_horizon_{groups}_{type}_{y_train.name}_{level}_AEZNew.nc')
                az.summary(trace_h).to_csv(path+f'/traces_info/trace_summary_lc_{groups}_{type}_{y_train.name}_{level}_AEZNew.csv')
                _ = az.plot_trace(trace_h, compact=True)
                plt.savefig(path+f'/traces_info/tracePlot_{groups}_{type}_{y_train.name}_{level}_AEZNew.png')


    ############################### Pooled Model #####################################
    if type == 'Pooled':
        for h in horizons:
            print(f'AEZ {type} for {h} Weeks ahead')
            X_train, y_train, ssn_gp, cty_gp, aez_gp, means = HBM_Factory_AEZ.PrepData(sub_train, lst_p0=0,precip_p1=6,soil_p2=6,targ_q=6,
                                                                                              target=target, anom=True, growing_ssn=None,
                                                                                              f_horizon=h)

            X_sub = X_train
            print(X_sub.shape)
            X_data = X_sub.drop(['AEZ_Code_lag_0', 'Season_Code_lag_0','County_Code_lag_0', y_train.name], axis=1)
            y_data = X_sub[y_train.name]
            print(X_data.columns)

            #County Index
            aez_idx = X_sub['AEZ_Code_lag_0'].values.astype(int)
            ssn_idx = X_sub['Season_Code_lag_0'].values.astype(int)
            cty_idx = X_sub['County_Code_lag_0'].values.astype(int)

            with HBM_Factory_AEZ.HBARDL_factoryC(X_data=X_data, y_data=y_data, sampler='JAX') as HB_Model_A:
                trace_h = pm.sampling_jax.sample_numpyro_nuts(samples, tune=samples, target_accept=0.95)
                # trace_h = pm.sample(2000, tune=5000, target_accept=0.9)

                trace_h.to_netcdf(filename=path+f'/h_models/f_horizon_{groups}_{type}_{y_train.name}_{level}_AEZNew.nc')
                az.summary(trace_h).to_csv(path+f'/traces_info/trace_summary_lc_{groups}_{type}_{y_train.name}_{level}_AEZNew.csv')
                _ = az.plot_trace(trace_h, compact=True)
                plt.savefig(path+f'/traces_info/tracePlot_{groups}_{type}_{y_train.name}_{level}_AEZNew.png')


    ############################### UnPooled Model #####################################
    if type == 'Unpooled':
        for c in county103:
            train_df1, county_name = HBM_Factory_AEZ.trainSet(path+f'/LC_Extracts/AEZ_{level}/s2_{c}_aez_data_SM_3M_pft.csv')
            print(train_df1)
            for h in horizons:
                print(f'AEZ {type} - {c} for {h} Weeks ahead')
                X_train, y_train, ssn_gp, cty_gp, aez_gp, means = HBM_Factory_AEZ.PrepData(train_df1, lst_p0=0,precip_p1=6,soil_p2=6,targ_q=6,
                                                                                                  target=target, anom=True, growing_ssn=None,
                                                                                                  f_horizon=h)

                X_sub = X_train
                print(X_sub.shape)
                X_data = X_sub.drop(['AEZ_Code_lag_0', 'Season_Code_lag_0','County_Code_lag_0', y_train.name], axis=1)
                y_data = X_sub[y_train.name]
                print(X_data.columns)

                #County Index
                aez_idx = X_sub['AEZ_Code_lag_0'].values.astype(int)
                ssn_idx = X_sub['Season_Code_lag_0'].values.astype(int)
                cty_idx = X_sub['County_Code_lag_0'].values.astype(int)

                with HBM_Factory_AEZ.HBARDL_factoryC(X_data=X_data, y_data=y_data, sampler='JAX') as HB_Model_A:
                    trace_h = pm.sampling_jax.sample_numpyro_nuts(samples, tune=samples, target_accept=0.95)
                    # trace_h = pm.sample(2000, tune=5000, target_accept=0.9)

                    trace_h.to_netcdf(filename=path+f'/h_models/f_horizon_{c}_{groups}_{type}_{y_train.name}_{level}_AEZNew.nc')
                    az.summary(trace_h).to_csv(path+f'/traces_info/trace_summary_lc_{c}_{groups}_{type}_{y_train.name}_{level}_AEZNew.csv')
                    _ = az.plot_trace(trace_h, compact=True)
                    plt.savefig(path+f'/traces_info/tracePlot_{c}_{groups}_{type}_{y_train.name}_{level}_AEZNew.png')
