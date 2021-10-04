"""
@author: Edward Salakpi
"""

import glob
import pandas as pd
import numpy as np
import arviz as az
import pymc3 as pm
import pymc3.sampling_jax
import warnings
warnings.filterwarnings('ignore')

#Function for creating direct forecast DataFrame
def makeDirFcastDf2(df, p_lags0,p_lags1,p_lags2, q_lags, s_lags, z_lags, c_lags, date, target):
    """
    df:  data (DataFrame)
    p_lags0,p_lags1,p_lags2, q_lags, s_lags, z_lags, c_lags, date: lag order for input variables in this order
    'LST','Rainfall','SoilMoist',Target Variable(VCI etc)','Season','Zone', 'County', 'Date', Int
    date:
    target: target variable to forecast, string
    """
    new_df = pd.DataFrame()
    col = df.columns
    for h in range(1,21): # Number of lead times
        new_df[f'{target}_0']=df[target]
        new_df[f'{target}_{h}']=df[target].shift(periods=-h)

        if q_lags == 0:
            pass
        elif q_lags == 1:
            new_df[f'{target}_lag_0']=df[target]
        elif q_lags >= 2:
            for q in range(1,q_lags):
                new_df[f'{target}_lag_0']=df[target]
                new_df[f'{target}_lag_{q}']=df[target].shift(periods=q)

        if p_lags0 == 0:
            pass
        elif p_lags0 == 1:
            new_df[f'{col[0]}_lag_0']=df[col[0]]
        elif p_lags0 >= 2:
            for p0 in range(1,p_lags0):
                new_df[f'{col[0]}_lag_0']=df[col[0]]
                new_df[f'{col[0]}_lag_{p0}']=df[col[0]].shift(periods=p0)

        if p_lags1 == 0:
            pass
        elif p_lags1 == 1:
            new_df[f'{col[1]}_lag_0']=df[col[1]]
        elif p_lags1 >= 2:
            for p1 in range(1,p_lags1):
                new_df[f'{col[1]}_lag_0']=df[col[1]]
                new_df[f'{col[1]}_lag_{p1}']=df[col[1]].shift(periods=p1)

        if p_lags2 == 0:
            pass
        elif p_lags2 == 1:
            new_df[f'{col[2]}_lag_0']=df[col[2]]
        elif p_lags2 >= 2:
            for p2 in range(1,p_lags2):
                new_df[f'{col[2]}_lag_0']=df[col[2]]
                new_df[f'{col[2]}_lag_{p2}']=df[col[2]].shift(periods=p2)
#

        if s_lags == 1:
            new_df['Season_Code_lag_0']=df['Season_Code']

        if z_lags == 1:
            new_df['AEZ_Code_lag_0']=df['AEZ_Code']

        if c_lags == 1:
            new_df['County_Code_lag_0']=df['County_Code']

        if date == 1:
            new_df['Date_lag_0']=df['Date']

    return new_df


#Function for spliting data in to training and test set DataFrame
def fcast_train_testDF1(df, p_order0,p_order1,p_order2,q_order, target_var, s_lags, z_lags, c_lags,date, f_horizon=None):
    """
    df:  data (DataFrame)
    p_order0,p_order1,p_order2,q_order, target_var, s_lags, z_lags, c_lags,date: lag order for input variables in this order
    'LST','Rainfall','SoilMoist',Target Variable(VCI etc)','Season','Zone', 'County', 'Date', Int
    target: target variable to forecast, string
    """

    newdf_ls = []
    X_train2 = pd.DataFrame()
    ssn_ = df.Season.unique().tolist()
    cty_ = df.County.unique().tolist()
    aez_ = df.AEZ.unique().tolist()

    for c in cty_:
        df2 = df[df.County == c]
        for z in df2.AEZ.unique().tolist():
            df3 = df2[df2.AEZ == z]
            newdf1 = makeDirFcastDf2(df3,p_order0,p_order1, p_order2, q_order, s_lags, z_lags, c_lags,date, target_var)
            useDF1 = newdf1.loc[:,[g for g in newdf1.columns if 'lag' in g]]

            useDF1[f'{target_var}_{f_horizon}'] = newdf1[f'{target_var}_{f_horizon}']
            X_train = useDF1.dropna()
            newdf_ls.append(X_train)

    X_train2 = pd.concat(newdf_ls, axis=0)
    y_train = X_train2.loc[:,f'{target_var}_{f_horizon}']
    return X_train2, y_train, ssn_, cty_, aez_

#Function for data standardization
def scaleValues(df, target):
    sdf = df.iloc[:,:-7]
    print(sdf.columns)
    sdf1 = (sdf-sdf.mean())/sdf.std()
    sdf1[target] = df[target]/100
    sdf1['Season'] = df.Season
    sdf1['AEZ'] = df.Zone
    sdf1['County'] = df.County
    sdf1['County_Code'] = df.County_Code
    sdf1['AEZ_Code'] = df.Zone_Code
    sdf1['Season_Code'] = df.Season_Code
    sdf1['Date'] = df.Date

    return sdf1

#Function for detrending VCI or making VCI anomalies
def detrend(df, target):
    df_ls = []
    means0 = {'county':[],'AEZs':[], 'means':[]}
    for c in df.County.unique().tolist():
        df2 = df[df.County == c]
        for z in df2.Zone.unique().tolist():
            df4 = df2[df2.Zone == z]
            tgmeans = df4[[target]].mean()
            df4[target] = df4[[target]] - tgmeans
            means0['county'].append(c)
            means0['AEZs'].append(z)
            means0['means'].append(tgmeans.values[0]/100)
            df_ls.append(df4)
    return pd.concat(df_ls), means0

#Function for adding rainig seasons to data
def addSeason(_df):
    _df['Date'] = pd.to_datetime(_df['Date'])
    df2 = _df.set_index('Date')

    df2['Month'] = df2.index.month

    jf = df2[(df2.Month>=1) & (df2.Month<=2)]
    jf['Season']  = 'jf'
    mam = df2[(df2.Month>=3) & (df2.Month<=5)]
    mam['Season']  = 'mam'
    jja = df2[(df2.Month>=6) & (df2.Month<=9)]
    jja['Season']  = 'jja'
    ond = df2[(df2.Month>=10) & (df2.Month<=12)]
    ond['Season']  = 'ond'
    new_sdf = pd.concat([jf, mam, jja, ond])
    new_sdf = new_sdf.sort_values(by='Date')
    return new_sdf.reset_index()

#Function for making training set for No-Pooling Model
def trainSet(f):
    """
    f: Files
    """
    sub_ls=[]
    cty = f.split('/')[-1].split('_')[1]
    ndf = pd.read_csv(f)
    aezone = ndf.Zone.unique()
    for z in aezone:
        subndf1 = ndf[ndf.Zone == z]
        subndf1['Date'] = pd.to_datetime(subndf1['Date'])
        subndf1 = subndf1.set_index('Date')

        subndf2 = subndf1[(subndf1.index.year >= 2001) & (subndf1.index.year <= 2015)]
        subndf2['County'] = cty
        subndf2['County'] = subndf2['County'].astype('category')
        subndf2['County_Code'] = subndf2['County'].cat.codes

        subndf2['Date'] = subndf2.index
        subndf2.index = np.arange(len(subndf2))
        sub_ls.append(subndf2)
    trainDf = pd.DataFrame()
    trainDf = pd.concat(sub_ls)
    return trainDf, cty

#Function for making test set for forecasting/prediction
def testSet(f):
    """
    f: Files
    """
    sub_ls=[]
    cty = f.split('/')[-1].split('_')[1]
    ndf = pd.read_csv(f)
    aezone = ndf.Zone.unique()
    for z in aezone:
        subndf1 = ndf[ndf.Zone == z]

        subndf1['Date'] = pd.to_datetime(subndf1['Date'])
        subndf1 = subndf1.set_index('Date')
        subndf2 = subndf1[(subndf1.index.year > 2015)]

        subndf2['County'] = cty
        subndf2['County'] = subndf2['County'].astype('category')
        subndf2['County_Code'] = subndf2['County'].cat.codes

        subndf2['Date'] = subndf2.index
        subndf2.index = np.arange(len(subndf2))
        sub_ls.append(subndf2)
    testDf = pd.DataFrame()
    testDf = pd.concat(sub_ls)
    return testDf, cty

#Function for preparing training data for HBMs
def PrepData(tr_df, lst_p0,precip_p1,soil_p2,targ_q, target, f_horizon=None, anom=None, growing_ssn=None):
    """
    tr_df: training data (lagged variables) the target variables (DataFrame)
    lst_p0,precip_p1,soil_p2,targ_q: lag order for input variables, Int
    target: target variable to forecast, string
    f_horizon: forecast lead time, int
    anom: Use anomaly inputs (Boolean)
    model_factory: instance of model used to train the model
    growing_ssn: use seasons (Only used when working with MAM or OND seasons)
    sampler: MCMC sampler used in training model for Hamiltonian Monte Carlo (HMC) or JAX Please run on PyMC3 v3.11.2 if using JAX
    """

    if anom == True:
        vars_abs = ['LST_Anom','Rainfall_Anom', 'SoilMoist_Anom',f'{target}','Zone', 'Zone_Code','County', 'County_Code', 'Season', 'Season_Code','Date']
    elif anom ==False:
        vars_abs = ['LST','Rainfall', 'SoilMoist',f'{target}','Zone', 'Zone_Code','County', 'County_Code', 'Season', 'Season_Code','Date']

    pq_order = [lst_p0,precip_p1,soil_p2,targ_q]

    tr_df['Rainfall'] = tr_df['Rainfall'].ewm(com=5).mean()

    if growing_ssn == 'MAM':
        tr_df = tr_df.loc[tr_df['Season'].isin(['mam'])]
    elif growing_ssn == 'OND':
        tr_df = tr_df.loc[tr_df['Season'].isin(['ond'])]
    else:
        pass


    tr_df2, target_means = detrend(tr_df, target)

    scale_df = scaleValues(tr_df2.loc[:,vars_abs], target)

    X_trainX, y_trainX, ssn_grp, cty_grp, aez_grp = fcast_train_testDF1(scale_df, p_order0=pq_order[0] ,
                                                                    p_order1=pq_order[1], p_order2=pq_order[2],
                                                                    q_order=pq_order[3],s_lags=1, z_lags=1, c_lags=1,date=1,
                                                                    target_var=target,
                                                                    f_horizon=f_horizon)
#     print(X_trainX.columns)
    return X_trainX, y_trainX, ssn_grp, cty_grp, aez_grp, target_means



#Function for deriving forecast probabilities
def NewcatProbs(farr0):
    """
    farr0: Forecast distribution for target variables from sample_posterior_predictive, Array
    n_samples: Number of forecast samples drawn
    """
    n_samples = farr0.shape[0]
    cats = {'FNo-Drought':[],'FDrought':[]}
    for i in np.arange(farr0.shape[1]):
        # cats['FNo-Drought'].append(np.mean(farr0[:,i][(farr0[:,i] > 0.35)]))
        # cats['FDrought'].append(np.mean(farr0[:,i][(farr0[:,i] < 0.35)]))
        cats['FNo-Drought'].append(len(farr0[:,i][(farr0[:,i] > 0.35)])/n_samples)
        cats['FDrought'].append(len(farr0[:,i][(farr0[:,i] < 0.35)])/n_samples)
    return pd.DataFrame(cats)


# Function for Hierarchical Bayesian Model (Please run on PyMC3 v3.11.2)
def HBARDL_factoryE(X_data=None, y_data=None, idx=None, hgroup=None, sampler=None):
    """
    X_data: input data (lagged variables), ND Array
    y_data: taget variable Data, 1D Array
    idx: Categorical Index values for data sub-groups 1D Array
    hgroup: Sub-group lables, group_dict key (AEZ) or SSN
            group_dict = {'AEZs':['Humid','Semi-Humid','Semi-Arid','Arid','Very-Arid'],
                'SSNs':['jf','mam', 'jja', 'ond']}
    sampler: MCMC sampler choose MCMC for Hamiltonian Monte Carlo (HMC) or JAX Please run on PyMC3 v3.11.2 if using JAX
    """
    group_dict = {'AEZs':['Humid','Semi-Humid','Semi-Arid','Arid','Very-Arid'],
                'SSNs':['jf','mam', 'jja', 'ond']}


    coords = {'Var0':X_data.columns.to_list(), 'Var1':['var_intercept']+X_data.columns.to_list(), hgroup:group_dict[hgroup], 'Obs':np.arange(X_data.shape[0])}

    with pm.Model(coords=coords) as new_hbm:
        if sampler == 'MCMC':
            x_input = pm.Data('x_input1', X_data)
            y_input = pm.Data('y_input', y_data)
            group_idx = pm.Data(f'{hgroup}_idx', idx, dims='Obs')

        elif sampler == 'JAX':
            x_input = X_data.values
            y_input = y_data
            group_idx = idx


        sd_dist = pm.HalfNormal.dist(1.0)

    # get back standard deviations and rho:
        chol, corr, stds = pm.LKJCholeskyCov("chol", n=X_data.shape[1]+1, eta=2.0, sd_dist=sd_dist, compute_corr=True)

        # Model global priors (Hyperpriors)
        alpha = pm.Normal('globl_alpha', mu=0, sigma=1.0)
        mu_beta1 = pm.Normal('global_beta',  mu=0, sigma=1.0, dims='Var0')

        # Model group level priors
        mu_z = pm.Normal("mu_z", mu=0, sigma=1.0, dims=(hgroup,'Var1')) #Offset
        alpha_beta_AEZ = pm.Deterministic(f'group_beta_{hgroup}', pm.math.dot(chol,  mu_z.T).T, dims=(hgroup, 'Var1'))

        # Model
        mean = alpha + alpha_beta_AEZ[group_idx,0] + (mu_beta1 + alpha_beta_AEZ[group_idx,1:]*x_input).sum(axis=1)

        #Model Error
        sigma = pm.HalfNormal('sigma', 1.0)

        # Model likelihood
        y_pred = pm.Normal('y_pred', mu=mean, sigma=sigma, observed=y_input, testval=1, dims="Obs")

    return new_hbm


# Function for Bayesian ARDL model (Please run on PyMC3 v3.11.2)
def HBARDL_factoryC(X_data=None, y_data=None, sampler=None):
    """
    X_data: input data (lagged variables), ND Array
    y_data: taget variable Data, 1D Array
    sampler: MCMC sampler choose 'MCMC' for Hamiltonian Monte Carlo (HMC) or 'JAX' Please run on PyMC3 v3.11.2 if using JAX
    """
    coords = {'Var':X_data.columns, 'Obs':np.arange(X_data.shape[0])}
    with pm.Model(coords=coords) as hadl_model_h3A:
        #Get DataFrame
        if sampler == 'MCMC':
            x_input = pm.Data('x_input', X_data)
            y_input = pm.Data('y_input', y_data)

        elif sampler == 'JAX':
            x_input = X_data.values
            y_input = y_data

        # Priors
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1.0)
        mu_beta = pm.Normal('mu_beta', mu=0, sigma=0.5, dims='Var')

        # Model
        mean = mu_alpha + (mu_beta*x_input).sum(axis=1)

        #Model Error
        sigma = pm.HalfNormal('sigma', 1.0)

        # Model likelihood
        y_pred = pm.Normal('y_pred', mu=mean, sigma=sigma, observed=y_input, testval=1, dims="Obs")

    return hadl_model_h3A

# Function for forecasting VCI 'N' steps ahead (Please run on PyMC3 v3.11.2)
def testModel(testDF, county, trace, horizon, target, hgroups, anom=None, detrend=None, sampler=None, model_factory=None, growing_ssn=None):
    """
    testDF: test data (lagged variables) without the target variables (DataFrame)
    county: County name, string
    trace: Posterior distribution (model parameter) from HMC or Jax sampler
    horizon: forecast lead time, int
    target: target variable to forecast, string
    hgroups: Sub-group lables, group_dict key (AEZ) or SSN
            level_dict = {'AEZs':['Humid','Semi-Humid','Semi-Arid','Arid','Very-Arid'],
                'SSNs':['jf','mam', 'jja', 'ond']}
    anom: Use anomaly inputs (Boolean)
    detrend: create VCI anomaly
    model_factory: instance of model used to train the model
    growing_ssn: use seasons (Only used when working with MAM or OND seasons)
    sampler: MCMC sampler used in training model for Hamiltonian Monte Carlo (HMC) or JAX Please run on PyMC3 v3.11.2 if using JAX
    """

    level_dict = {'AEZs':['Humid','Semi-Humid','Semi-Arid','Arid','Very-Arid'],
                'SSNs':['jf','mam', 'jja', 'ond']}



    X_test, y_test, ssn_grp_test, cty_grp_test, aez_grp_test, test_means = PrepData(testDF,lst_p0=0,precip_p1=6,soil_p2=6,
                                                                                                          targ_q=6,
                                                                                                          target=target,
                                                                                                          anom=anom, growing_ssn=growing_ssn,
                                                                                                          f_horizon=horizon)

    print(aez_grp_test)
    test_aez_idx = X_test['AEZ_Code_lag_0'].values.astype(int)
    test_ssn_idx = X_test['Season_Code_lag_0'].values.astype(int)
    test_cty_idx = X_test['County_Code_lag_0'].values.astype(int)
    Date = X_test['Date_lag_0'].values

    group_dict = {'AEZs':test_aez_idx,
                'SSNs':test_ssn_idx}

    X_test0 = X_test.drop(['AEZ_Code_lag_0','Season_Code_lag_0', 'County_Code_lag_0','Date_lag_0', y_test.name], axis=1)

    meandf = pd.DataFrame(test_means)
    meandf2 = meandf[meandf.county==county]
#     print(meandf2.iloc[:,1:]).set_index(hgroups)
    mean_dicts = meandf2.iloc[:,1:].set_index(hgroups).to_dict('index')
    if detrend == True:
        lcmeans = np.array([mean_dicts[level_dict[hgroups][l]]['means'] for l in group_dict[hgroups]])
    elif detrend == False:
        lcmeans = 0

    county = np.repeat(county, len(test_aez_idx))
    h = np.repeat(horizon, len(test_aez_idx))
    aez = [level_dict['AEZs'][z] for z in test_aez_idx]
    y_empty = np.empty_like(y_test.values)

    if sampler == 'JAX0':
        print('Samplling from UnPooled')
        with HBARDL_factoryC(X_data=X_test0, y_data=y_empty, sampler='JAX') as HB_Model:
            pred2_ = pm.sample_posterior_predictive(trace, random_seed=100)
            new_pred = pred2_['y_pred']+lcmeans
            probs= NewcatProbs(new_pred, new_pred.shape[0])
            print(new_pred.shape)
            print(len(aez))
            print(y_test.values.shape)
            if horizon >=10:
                v = y_test.name[:-3]
            else:
                v = y_test.name[:-2]
            forecastDf = pd.DataFrame({'County':county,
                                    'AEZ':aez,
                                    'Horizon':h,
                                    'Date':Date,
                                    f'{v}_Forecast':new_pred.mean(axis=0),
                                    f'{v}_Upper1':np.percentile(new_pred, 97.5, axis=0),
                                    f'{v}_Upper0':np.percentile(new_pred, 75, axis=0),
                                    f'{v}_Lower1':np.percentile(new_pred, 25, axis=0),
                                    f'{v}_Lower0':np.percentile(new_pred, 2.5, axis=0),
                                    f'{v}_Observed':y_test.values+lcmeans})

            forecastDf['Obs_No-Drought'] = np.where((forecastDf[f'{v}_Observed'] > 0.35), 1, 0)
            forecastDf['Obs_Drought'] = np.where((forecastDf[f'{v}_Observed'] < 0.35), 1, 0)

            forecastDf[['FNo-Drought','FDrought']] = probs

    if sampler == 'JAX1':
        print('Samplling from PartPooled')
        with HBARDL_factoryE(X_data=X_test0, y_data=y_empty, idx=group_dict[hgroups], hgroup=hgroups, sampler='JAX') as HB_Model:
            pred2_ = pm.sample_posterior_predictive(trace, random_seed=100)
            new_pred = pred2_['y_pred']+lcmeans
            probs= NewcatProbs(new_pred, new_pred.shape[0])
            print(new_pred.shape)
            print(len(aez))
            print(y_test.values.shape)
            if horizon >=10:
                v = y_test.name[:-3]
            else:
                v = y_test.name[:-2]
            forecastDf = pd.DataFrame({'County':county,
                                    'AEZ':aez,
                                    'Horizon':h,
                                    'Date':Date,
                                    f'{v}_Forecast':new_pred.mean(axis=0),
                                    f'{v}_Upper1':np.percentile(new_pred, 97.5, axis=0),
                                    f'{v}_Upper0':np.percentile(new_pred, 75, axis=0),
                                    f'{v}_Lower1':np.percentile(new_pred, 25, axis=0),
                                    f'{v}_Lower0':np.percentile(new_pred, 2.5, axis=0),
                                    f'{v}_Observed':y_test.values+lcmeans})

            forecastDf['Obs_No-Drought'] = np.where((forecastDf[f'{v}_Observed'] > 0.35), 1, 0)
            forecastDf['Obs_Drought'] = np.where((forecastDf[f'{v}_Observed'] < 0.35), 1, 0)

            forecastDf[['FNo-Drought','FDrought']] = probs

    return forecastDf, new_pred
