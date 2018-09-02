library(h2o)
h2o.init(min_mem_size='100G', max_mem_size='200G')

########
# STEP 1
########
# load the data
df <- h2o.importFile("http://coursera.h2o.ai/house_data.3487.csv")

# create the year
df[,c("Year")]<-h2o.substring(df$date, start=1, stop=4)

# create the Month
df[,c("Month")]<-as.factor(h2o.substring(df$date, start=5, stop=6))


xall<-c("Year","Month", "bedrooms", "bathrooms","sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade",  "sqft_above", "sqft_basement", "yr_built", "yr_renovated",  "lat",  "long", "sqft_living15", "sqft_lot15")



y="price"

# split into train and test since we will use cross validation 80-20
parts<-h2o.splitFrame(df, c(0.9), seed=123)
sapply(parts,nrow) #[1] 19462  2151


train<-parts[[1]]
test<-parts[[2]]

########
# STEP 2
########


# GLM model
model0<-h2o.glm(x=xall, y=y, training_frame=train)
h2o.performance(model0, test)
# RMSE:  360914.7

# random forest
model1<-h2o.randomForest(x=xall, y=y, training_frame=train,  max_depth = 30, ntrees =150)
h2o.performance(model2, test)
# RMSE:  12950

# random forest
model2<-h2o.randomForest(x=xall, y=y, training_frame=train)
h2o.performance(model2, test)
# RMSE:  130840.5


# deep learning
model3<-h2o.deeplearning(x=xall, y=y, training_frame=train)
h2o.performance(model3, test)
# RMSE:  126341


# GBM
model4<-h2o.gbm(x=xall, y=y, training_frame=train)
h2o.performance(model4, test)
# RMSE:  131603.7


########
# STEP 3
########

nfolds<-5


m1<-h2o.randomForest(xall,y, train, nfolds = nfolds, max_depth = 30, ntrees =150, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
m2<-h2o.randomForest(xall,y, train, nfolds = nfolds, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
m3<-h2o.deeplearning(xall,y, train, nfolds = nfolds, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
m4<-h2o.gbm(xall,y, train, nfolds = nfolds, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE)
model_ids<-list(m1@model_id, m2@model_id,m3@model_id,m4@model_id)

m_SE<-h2o.stackedEnsemble(x=xall,y=y, training_frame=train, base_models = model_ids)

models<-c(m1,m2,m3,m4,m_SE)

sapply(models, h2o.rmse)
########
# STEP 4
########
h2o.performance(m_SE, test)



########
# STEP 5
########



model2.1<-h2o.randomForest(x=xall, y=y, max_depth = 30, ntrees =150,  training_frame=train)
h2o.performance(model2.1, test)


model_pathSE<-h2o.saveModel(object=m_SE, path=getwd(), force=TRUE)