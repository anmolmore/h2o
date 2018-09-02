
# coding: utf-8

# In[19]:


import h2o
import pandas as pd
import random
from random import randint
#import matplotlib.pyplot as tk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


#Generate random dataset for housing prices

def list_integer(no_of_rows, min, max):
    x_list = []
    for i in range(no_of_rows) :
        x_list.append(randint(min, max))
    return(x_list)

def list_float(no_of_rows, min, max):
    x_list = []
    for i in range(no_of_rows) :
        x_list.append(round(random.uniform(min,max),2))
    return(x_list)

def list_random_pick(no_of_rows, pick_list) :
    x_list = []
    for i in range(no_of_rows) :
        x_list.append(random.choice(pick_list))
    return(x_list)

def list_random_pick(no_of_rows, pick_list) :
    x_list = []
    for i in range(no_of_rows) :
        x_list.append(random.choice(pick_list))
    return(x_list)
    
no_of_rows = 1000
#feature and target columns
train_columns = ['No of BedRooms','Size','No of Floors','Area Code','State','Area Type','Age of Contruction']
df = pd.DataFrame(columns=train_columns)

#create arrays for each column with randomness in price
df['No of BedRooms'] = list_integer(no_of_rows,1,8)
df['Size'] = list_float(no_of_rows,600,2800)
df['No of Floors'] = list_integer(no_of_rows,0,3)
df['Area Code'] = list_integer(no_of_rows,10000,50000)
state_list = ["Arizona","California","Connecticut","Columbia", "Florida", "Georgia","Kansas","New York","Texas","Washington"]
df['State'] = list_random_pick(no_of_rows,state_list)
area_type_list = ["Upscale","Midscale","Downscale","Commercial"]
df['Area Type'] = list_random_pick(no_of_rows,area_type_list)
df['Age of Contruction'] = list_integer(no_of_rows,1,15)

#Calculate housing price, add some randomness
df['Price'] = randint(100000, 500000) + df['No of BedRooms']*100 + df['No of Floors']*100 + df['Size']*30 + df['Area Code']*1 - df['Age of Contruction']*300
df.head()


# In[21]:


#print a correlation matrix to validate our data has good amount of relation
print(df.corr())


# In[22]:


#initialize h2o session
h2o.init()


# In[23]:


#convert dataframe to h2o frame
data = h2o.H2OFrame(df)


# In[24]:


data.summary()
#check data summary, we can see min, max, mean prices


# In[25]:


#split data in train, valid and testset
train,valid,test = data.split_frame([0.8,0.1])


# In[26]:


from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[27]:


mRF = H2ORandomForestEstimator()
mRF.train(train_columns,"Price",train)


# In[28]:


#Get Random Forest model summary
mRF


# In[29]:


#Test initial model performance on test data
mRF.model_performance(test)

# plot training logloss and auc
sh = mRF.score_history()
sh = pd.DataFrame(sh)
sh.plot(x='number_of_trees', y = ['training_deviance'])


# In[30]:


#Lets create the model again with validation frame
#We can see we got a better RMSE

mRF_validation_frame = H2ORandomForestEstimator(model_id="mrf_validation_model")
mRF_validation_frame.train(train_columns,"Price",train, validation_frame=valid)
mRF_validation_frame.model_performance(test)


# In[31]:


# plot rmse against no of trees, its much more smoother now
sh = mRF_validation_frame.score_history()
sh = pd.DataFrame(sh)
#print(sh)
sh.plot(x='number_of_trees', y = ['training_rmse','validation_rmse'])


# In[32]:


#Now lets try to overfit the data with high number of trees
mRF_overfit = H2ORandomForestEstimator(model_id="mrf_overfit",ntrees=1000)
mRF_overfit.train(train_columns,"Price",train, validation_frame=valid)
mRF_overfit.model_performance(test)


# In[33]:


#Highly overfitted with no improvement after 250 trees
sh = mRF_overfit.score_history()
sh = pd.DataFrame(sh)
#print(sh)
sh.plot(x='number_of_trees', y = ['training_rmse','validation_rmse'])


# In[34]:


#Now lets try to train with high number of trees and max_depth
mRF_overfit = H2ORandomForestEstimator(model_id="mrf_overfit",ntrees=1000,max_depth=100)
mRF_overfit.train(train_columns,"Price",train, validation_frame=valid)
mRF_overfit.model_performance(test)


# In[35]:


#Even more overfitted data
sh = mRF_overfit.score_history()
sh = pd.DataFrame(sh)
sh.plot(x='number_of_trees', y = ['training_rmse','validation_rmse'])

