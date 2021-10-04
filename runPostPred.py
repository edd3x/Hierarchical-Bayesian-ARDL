# import subprocess
# import glob
# import runHBM_EJAX as LCModels
import sample_LC
import sample_LC_MAM
import sample_AEZ

# tune = 6000
sample_AEZ.runPostSample(type='PartPooled', horizons=[8])

# models = ['Unpooled','PartPooled']
# for m in models:
#         sample_LC.runPostSample(type=m)

# for m in models[:2]:
#     sample_AEZ_LC.runPostSample(type=m)

# sample_AEZ.runPostSample(type='Unpooled')
