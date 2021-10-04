"""
@author: Edward Salakpi
"""

import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pymc3 as pm
import arviz as az
import HBM_Factory
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
groups = 'LCs'
level = '3M'


path = '/'
trainDataFile = glob.glob(path+f'/LC_{level}/HBM_train_df_{level}_Sub_pft.csv')
dataFiles = glob.glob(path+f'/LC_{level}/s2*pft*csv')

dataFiles_Sub = []
for cs in county101:
    [dataFiles_Sub.append(m) for m in dataFiles if cs in m]

# print(trainDataFile)
trainDF = pd.read_csv(path+f'/LC_{level}/HBM_train_df_{level}_Sub_pft.csv')
sub_train = trainDF.loc[trainDF.County.isin(county103)]
print(sub_train.shape)

def runModel(type=None, samples=None, horizons=None):
    """Model types:
    type: str
        'PartPooled', 'Pooled', 'Unpooled'
    """
    ############################### PartialPooled Model #####################################
    if type == 'PartPooled':
        print(type)
        for h in horizons:
            print(f'LC {type} Model for {h} Weeks ahead')
            X_train, y_train, lcz_gp, ssn_gp, cty_gp, aez_gp, means = HBM_Factory.PrepData(sub_train, lst_p0=0,precip_p1=6,soil_p2=6,targ_q=6,
                                                                                              target=target, anom=True,
                                                                                              f_horizon=h)

            X_sub = X_train
            # print(X_sub.shape)
            X_data = X_sub.drop(['AEZ_Code_lag_0','LCover_Code_lag_0', 'Season_Code_lag_0','County_Code_lag_0','Date_lag_0', y_train.name], axis=1)
            print(f'Input:{X_data.shape}')
            y_data = X_sub[y_train.name]



            #County Index
            lc_idx = X_sub['LCover_Code_lag_0'].values.astype(int)
            ssn_idx = X_sub['Season_Code_lag_0'].values.astype(int)
            cty_idx = X_sub['County_Code_lag_0'].values.astype(int)

            with HBM_Factory.HBARDL_factoryE(X_data=X_data, y_data=y_data, idx=lc_idx, hgroup=groups, sampler='JAX') as HB_Model_A:
                trace_h = pm.sampling_jax.sample_numpyro_nuts(samples, tune=samples, target_accept=0.95)
                # trace_h = pm.sample(2000, tune=5000, target_accept=0.9)

                trace_h.to_netcdf(filename=path+f'/h_models/f_horizon_{groups}_{type}_{y_train.name}_{level}_LCMAM.nc')
                az.summary(trace_h).to_csv(path+f'/traces_info/trace_summary_lc_{groups}_{type}_{y_train.name}_{level}_LCMAM.csv')
                _ = az.plot_trace(trace_h, compact=True)
                plt.savefig(path+f'/traces_info/tracePlot_{groups}_{type}_{y_train.name}_{level}_LCMAM.png')



    ###############################Pooled Model#####################################
    if type == 'Pooled':
        for h in horizons:
            print(f'LC {type} Model for {h} Weeks ahead')
            X_train, y_train, lcz_gp, ssn_gp, cty_gp, aez_gp, means = HBM_Factory.PrepData(sub_train, lst_p0=0,precip_p1=6,soil_p2=0,targ_q=6,
                                                                                              target=target, anom=True, growing_ssn=None,
                                                                                              f_horizon=h)

            X_sub = X_train
            # print(X_sub.shape)
            X_data = X_sub.drop(['AEZ_Code_lag_0','LCover_Code_lag_0', 'Season_Code_lag_0','County_Code_lag_0', 'Date_lag_0', y_train.name], axis=1)
            print(f'Input:{X_data.shape}')
            y_data = X_sub[y_train.name]


            #County Index
            lc_idx = X_sub['LCover_Code_lag_0'].values.astype(int)
            ssn_idx = X_sub['Season_Code_lag_0'].values.astype(int)
            cty_idx = X_sub['County_Code_lag_0'].values.astype(int)

            with HBM_Factory.HBARDL_factoryC(X_data=X_data, y_data=y_data, sampler='JAX') as HB_Model_A:
                trace_h = pm.sampling_jax.sample_numpyro_nuts(samples, tune=samples, target_accept=0.95)
                # trace_h = pm.sample(2000, tune=5000, target_accept=0.9)

                trace_h.to_netcdf(filename=path+f'/h_models/f_horizon_{groups}_{type}_{y_train.name}_{level}_LCPOnly.nc')
                az.summary(trace_h).to_csv(path+f'/traces_info/trace_summary_lc_{groups}_{type}_{y_train.name}_{level}_LCPOnly.csv')
                _ = az.plot_trace(trace_h, compact=True)
                plt.savefig(path+f'/traces_info/tracePlot_{groups}_{type}_{y_train.name}_{level}_LCPOnly.png')



    ############################### Unpooled Model ###################################
    if type == 'Unpooled':
        for c in county103:
            train_df1, county_name = HBM_Factory.trainSet(path+f'/LC_Extracts/LC_{level}/s2_{c}_lc_data_SM_3M_pft.csv')
            for h in horizons:
                print(f'LC {type} - {c} for {h} Weeks ahead')
                X_train, y_train, lcz_gp, ssn_gp, cty_gp, aez_gp, means = HBM_Factory.PrepData(train_df1, lst_p0=0,precip_p1=6,soil_p2=6,targ_q=6,
                                                                                                  target=target, anom=True, growing_ssn='MAM',
                                                                                                  f_horizon=h)

                X_sub = X_train
                X_data = X_sub.drop(['AEZ_Code_lag_0','LCover_Code_lag_0', 'Season_Code_lag_0','County_Code_lag_0', 'Date_lag_0', y_train.name], axis=1)
                print(f'Input:{X_data.shape}')
                y_data = X_sub[y_train.name]


                #County Index
                lc_idx = X_sub['LCover_Code_lag_0'].values.astype(int)
                ssn_idx = X_sub['Season_Code_lag_0'].values.astype(int)
                cty_idx = X_sub['County_Code_lag_0'].values.astype(int)

                with HBM_Factory.HBARDL_factoryC(X_data=X_data, y_data=y_data, sampler='JAX') as HB_Model_A:
                    trace_h = pm.sampling_jax.sample_numpyro_nuts(samples, tune=samples, target_accept=0.95)
                    # trace_h = pm.sample(2000, tune=5000, target_accept=0.9)

                    trace_h.to_netcdf(filename=path+f'/h_models/f_horizon_{c}_{groups}_{type}_{y_train.name}_{level}_LCMAM.nc')
                    az.summary(trace_h).to_csv(path+f'/traces_info/trace_summary_lc_{c}_{groups}_{type}_{y_train.name}_{level}_LCMAM.csv')
                    _ = az.plot_trace(trace_h, compact=True)
                    plt.savefig(path+f'/traces_info/tracePlot_{c}_{groups}_{type}_{y_train.name}_{level}_LCMAM.png')
