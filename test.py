
# coding: utf-8

# Project Choice #2

#Step 1
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

h2o.init()
cacao = h2o.import_file("http://coursera.h2o.ai/cacao.882.csv")
train, test = cacao.split_frame([0.9], seed=60)

# Step 2
y = "Maker Location"
x = [a for a in cacao.col_names if a not in [y]]

# Step 3
m1 = H2ODeepLearningEstimator(model_id = "m1", nfolds = 5)
get_ipython().run_line_magic('time', 'm1.train(x, y, train)')

# Step 4
m2 = H2ODeepLearningEstimator(model_id = "m2",
                              epochs=50,
                              l1=1e-6,
                              l2=0,
                              hidden = [300,200,300],
                              nfolds = 5,
                              hidden_dropout_ratios = [0.1, 0.1, 0.1],
                              activation = "TanhWithDropout"
                             )
get_ipython().run_line_magic('time', 'm2.train(x, y, train)')

m1.plot()
m2.plot()
print(f' valid m1 => m2')
print(f' MSE: {m1.mse(xval=True):.3f} => {m2.mse(xval=True):.3f}') # 0.176 => 0.116
print(f'RMSE: {m1.rmse(xval=True):.3f} => {m2.rmse(xval=True):.3f}') # 0.420 => 0.341
print(f'')
print(f' test m1 => m2')
print(f' MSE: {m1.model_performance(test).mse():.3f} => {m2.model_performance(test).mse():.3f}') # 0.138 => 0.095
print(f'RMSE: {m1.model_performance(test).rmse():.3f} => {m2.model_performance(test).rmse():.3f}') # 0.371 => 0.308

# Step 5
h2o.save_model(model=m1, force=True)
h2o.save_model(model=m2, force=True)
