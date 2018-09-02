import h2o
import pandas as pd
#Complete run to make 20 base models and around 20 CV models, takes around 30 minutes on single cluster, 1.7 GB, 4 cores

h2o.init()
#import data from url and store as house_data
url = "http://coursera.h2o.ai/house_data.3487.csv"
print("Getting data file from h2o server, please wait ...")
house_data = h2o.import_file(url)
house_data.tail()

#Extract Day, Month and Year
def refine_date_col(data, col, pattern):
    data[col]         = data[col].as_date(pattern)
    data["Month"]     = data[col].month()
    data["Year"]      = data[col].year()
    
refine_date_col(house_data, "date", "%Y%m%dT%H%M%S")

#Check data quality
print(house_data.summary())

#split data in train, and test sets, and check number of rows
#splitting further in validation set is avoided as we have to build a ensemble model
train,test = house_data.split_frame([0.9],seed=123)
print("No of rows in train set : ", train.nrows)
print("No of rows in test set : ", test.nrows)

#We can see that except date and zipcode, all features have good coorelation to price
house_data.cor()

columns = house_data.columns
X = [
 'Month','Year','bedrooms','bathrooms','sqft_living', 'sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode', 'lat','long','sqft_living15','sqft_lot15']
y = 'price'

#Fix the number of folds for cross validation
folds = 5

#Build a generalized linear estimator model
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
glm = H2OGeneralizedLinearEstimator(family='poisson',nfolds=folds,fold_assignment="Modulo",keep_cross_validation_predictions=True)
glm.train(X,y,train)
print(glm.model_performance(test))

#Build Random Forest model.
#Tried with nTrees=200 and max_depth upto 50 but there was not much performance improvement
#Code removed to fit in 200 lines
from h2o.estimators.random_forest import H2ORandomForestEstimator
rFm = H2ORandomForestEstimator(nfolds=folds,fold_assignment="Modulo",keep_cross_validation_predictions=True)
rFm.train(X,y,train)
print(rFm.model_performance(test))

#Gradient Boosting Machine
from h2o.estimators.gbm import H2OGradientBoostingEstimator
mGBM = H2OGradientBoostingEstimator(nfolds=folds,fold_assignment="Modulo",keep_cross_validation_predictions=True)
mGBM.train(X,y,train)
print(mGBM.model_performance(test))

#XGBoost model, This showed the most optimal result and also achieves below $123000 RMSE
from h2o.estimators import H2OXGBoostEstimator
xgb = H2OXGBoostEstimator(nfolds=folds,ntrees=60,learn_rate=0.2,fold_assignment="Modulo",keep_cross_validation_predictions=True)
xgb.train(X,y,train)
print(xgb.model_performance(test))


print("##################################### Check Ensemble Model #####################")
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
# Train a stacked ensemble using above models
ensemble = H2OStackedEnsembleEstimator(base_models=[glm,rFm,mGBM, xgb])
ensemble.train(X,y,train)

# Eval ensemble performance on the test data
#performance is worst among all
print(ensemble.model_performance(test))


print("##################################### Start Training Deep Learning Model #####################")
#split train set in train and validation
#Running Deep learning using cross validation is extremely slow, so just use validation set
train,valid = train.split_frame([0.9])
print("No of rows in train, after validation set split : ", train.nrows)
print("No of rows in validation : ", valid.nrows)


from h2o.estimators.deeplearning import H2ODeepLearningEstimator

#Identify optimal no of epochs, which is not overfitting. Also tried with 40,60,80
dl_50 = H2ODeepLearningEstimator(epochs=50)
dl_50.train(X, y, train, validation_frame=valid)

dl_100 = H2ODeepLearningEstimator(checkpoint=dl_50, epochs=100)
dl_100.train(X, y, train, validation_frame=valid)

dl_200 = H2ODeepLearningEstimator(checkpoint=dl_100, epochs=200)
dl_200.train(X, y, train, validation_frame=valid)


#Compare validation set performance with various epochs
models_dl = [dl_50, dl_100, dl_200]
for model in models_dl :
    print("\nTrain set RMSE : ", model.model_performance()['RMSE'])
    print("Validation set RMSE : ", model.model_performance(valid=True)['RMSE'])
print("There is not much performance improvement from using 50 to 100 epochs, so we will choose 50 for optimal performance with time ..., continue tuning DL Model for hidden layers..")

#Try various hidden layers to identify best parameter
dl_hidden_1 = H2ODeepLearningEstimator(epochs=50, hidden=[200,200])
dl_hidden_1.train(X, y, train, validation_frame=valid)

dl_hidden_2 = H2ODeepLearningEstimator(epochs=50, hidden=[200,200,200])
dl_hidden_2.train(X, y, train, validation_frame=valid)

dl_hidden_3 = H2ODeepLearningEstimator(epochs=50, hidden=[50,50,50])
dl_hidden_3.train(X, y, train, validation_frame=valid)

models_dl = [dl_hidden_1, dl_hidden_2, dl_hidden_3]
for model in models_dl :
    print("\nTrain set RMSE : ", model.model_performance()['RMSE'])
    print("Validation set RMSE : ", model.model_performance(valid=True)['RMSE'])
print("Hidden layer of [200,200,200] performed better on validation set without overfitting, But since there is not much difference we will use [200,200] default to save time .. continue tuning DL Model for activation function..")

#Identify best activation function
dl_1 = H2ODeepLearningEstimator(epochs=50, activation="tanh")
dl_1.train(X, y, train, validation_frame=valid)

dl_2 = H2ODeepLearningEstimator(epochs=50, activation="rectifier")
dl_2.train(X, y, train, validation_frame=valid)

dl_3 = H2ODeepLearningEstimator(epochs=50, activation="tanh_with_dropout")
dl_3.train(X, y, train, validation_frame=valid)

models_dl = [dl_1, dl_2, dl_3]
for model in models_dl :
    print("\nTrain set RMSE : ", model.model_performance()['RMSE'])
    print("Validation set RMSE : ", model.model_performance(valid=True)['RMSE'])
print("Default activation function, rectifier performs pretty much same as tanh, so we go with default ... continue tuning DL Model for dropout ...")

#Try building more models with dropout
dl_1 = H2ODeepLearningEstimator(epochs=50,activation='RectifierWithDropout',hidden_dropout_ratios = [0.1, 0.1])
dl_1.train(X, y, train, validation_frame=valid)

dl_2 = H2ODeepLearningEstimator(epochs=50,activation='RectifierWithDropout',hidden_dropout_ratios = [0.25, 0.25])
dl_2.train(X, y, train, validation_frame=valid)

dl_3 = H2ODeepLearningEstimator(epochs=50,activation='RectifierWithDropout',hidden_dropout_ratios = [0.35, 0.35])
dl_3.train(X, y, train, validation_frame=valid)

dl_4 = H2ODeepLearningEstimator(epochs=50,activation='RectifierWithDropout',hidden_dropout_ratios = [0.5, 0.5])
dl_4.train(X, y, train, validation_frame=valid)

#So the least value of dropout worked best
models_dl = [dl_1, dl_2,dl_3,dl_4]
for model in models_dl :
    print("\nTrain set RMSE : ", model.model_performance()['RMSE'])
    print("Validation set RMSE : ", model.model_performance(valid=True)['RMSE'])

#So our final DL model has RMSE on test set well below target of $123000
#Final value achieved on my machine is always around $114000
print("\n\n############## Final DL Model chosen with : epochs=50,activation='RectifierWithDropout',hidden_dropout_ratios = [0.1, 0.1] ###########################")
DL_final_model = H2ODeepLearningEstimator(epochs=50,activation='RectifierWithDropout',hidden_dropout_ratios = [0.1, 0.1])
DL_final_model.train(X, y, train)
DL_final_model.model_performance(test)

all_models = [glm,rFm,mGBM, xgb, ensemble,DL_final_model]
names = ["Generalized Linear Model","Random Forest","GBM","XGBoost","Ensemble","Deep Learning"]
for model, name in zip(all_models, names) :
	#save models
	model_path = h2o.save_model(model=model, path=name, force=True)
	print(model_path)
	# load the model
	saved_model = h2o.load_model(model_path)

# Generalized Linear Model    187209.680059
# Random Forest               136060.668020
# GBM                         107839.096881
# XGBoost                      70550.287586
# Ensemble                    363638.715151
# Deep Learning                99626.672844
print("\n ############### Compare models for RMSE, XGBoost and Deep learning models are best #############")
print(pd.Series(map(lambda x:x.rmse(),all_models),names))

# Generalized Linear Model    0.257416
# Random Forest               0.181417
# GBM                         0.176280
# XGBoost                     0.136861
# Ensemble                    0.541232
# Deep Learning               0.167163
print("\n ############### Compare models for RMSLE, GBM, XGBoost and Deep learning models are best #############")
print(pd.Series(map(lambda x:x.rmsle(),all_models),names))

print("\n\n ############# Compare models for RMSE on test set, again XGBoost and Deep learning gives desired result ################### ")
all_models = [glm,rFm,mGBM, xgb, ensemble,DL_final_model]
names = ["Generalized Linear Model","Random Forest","GBM","XGBoost","Ensemble","Deep Learning"]
print(pd.Series(map(lambda x:x.model_performance(test)['RMSE'],all_models),names))

print("\n\n #################### Final selected Model is Deep Learning, printing Model Summary ######################### ", DL_final_model)
