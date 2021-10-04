"""
@author: Edward Salakpi
Function for training HBMs
"""
import runHBM_EJAX as LCModels
import runHBM_AEZ_EJAX as AEZModels

# tune = 6000
# models = ['Unpooled', 'Pooled','PartPooled']
models = [('Unpooled',3000), ('Pooled',3000), ('PartPooled',10000)]

# models = ('PartPooled',10000)
horizons = [4,6,8,10,10]
for m in models:
    LCModels.runModel(type=models[0], samples=models[1], horizons=horizons)
    AEZModels.runModel(type=models[0], samples=models[1], horizons=horizons)
