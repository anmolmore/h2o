
# coding: utf-8

# In[1]:


import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import time

# In[2]:


#initialize h2o
#Running on 4 cores, 8 GB, Macbook Air
h2o.init()


# In[3]:


#import data from url
print("Loading data from endpoint - http://coursera.h2o.ai/cacao.882.csv")
url = "http://coursera.h2o.ai/cacao.882.csv"
cacao = h2o.import_file(url)
cacao.tail()


# In[4]:


#dplit data in train, valid, test sets
train,valid,test = cacao.split_frame([0.8,0.1])


# In[5]:


train.summary()


# In[6]:


#Identify input and target columns
x = ['Maker','REF','Review Date','Cocoa Percent','Rating','Bean Type','Bean Origin']
y = "Maker Location"


# In[7]:


#Run Deeplearning with default parameters, takes around 10 seconds
print("Creating base deep learning model, please wait .. may take upto 15 sec")
mDL = H2ODeepLearningEstimator(model_id='base_deep_learning_model')
start = time.time()
mDL.train(x,y,train,validation_frame=valid)
end = time.time()
print("Time taken to build baseline model : ",(end-start))
print(mDL.model_performance(test))

# In[8]:


#Takes around 8 times more than baseline model on my machine
#Also, trying to create model with early stopping

#Trying with three hidden layers, takes around 9 times the running time of baseline model
#Also, I tried with increasing number of hidden layers and number of hidden units. But performance didn't improved

#Ultimately, only early stopping helped with stopping rounds as 2 only
# ** Reported on test data. **
# 
# MSE: 0.1010197462659099
# RMSE: 0.31783603676409933
# LogLoss: 0.47930617731849545
# Mean Per-Class Error: 0.10960342865272445
#Hit Ratio 97%

mDL_200_epoch = H2ODeepLearningEstimator(epochs=200,stopping_rounds=2,stopping_tolerance=0,stopping_metric="logloss")
start = time.time()
mDL_200_epoch.train(x,y, train,validation_frame=valid)
end = time.time()
print("Time taken to build enhanced deep learning model : ",(end-start))
print(mDL_200_epoch.model_performance(test))


# In[9]:


#save base models
model_path = h2o.save_model(model=mDL, path="/tmp/base_deep_learning_model", force=True)
print(model_path)
# load the model
saved_model = h2o.load_model(model_path)


# In[10]:


#save other models
model_path = h2o.save_model(model=mDL_200_epoch, path="/tmp/enhanced_deep_model_early_stopping_hidden_layers", force=True)
print(model_path)
# load the model
saved_model = h2o.load_model(model_path)


# In[11]:




