import h2o
h2o.init()
# H2O cluster free memory: 	1.717 Gb
# H2O cluster total cores: 	4

url = "http://coursera.h2o.ai/cacao.882.csv"
cacao = h2o.import_file(url, destination_frame = "cacao")

train, valid, test = cacao.split_frame([0.8,0.1],seed=69)

cacao.columns
X = ['Maker',
 'Origin',
 'REF',
 'Review Date',
 'Cocoa Percent',
 'Rating',
 'Bean Type',
 'Bean Origin']
y = 'Maker Location'

from h2o.estimators.deeplearning import H2ODeepLearningEstimator

m1 = H2ODeepLearningEstimator()
%time m1.train(X, y, train, validation_frame=valid)
# CPU times: user 380 ms, sys: 44 ms, total: 424 ms
# Wall time: 16.4 s

m1.model_performance(test)
# MSE: 0.24169889403419217
# RMSE: 0.49162881733498104
# LogLoss: 0.9608104310489498
# Mean Per-Class Error: 0.3031816246285424

m2 = H2ODeepLearningEstimator(
    epochs=200, 
    stopping_metric = 'logloss', # logloss optimized
    activation = 'RectifierWithDropout',
    stopping_tolerance = 0.01,
    stopping_rounds = 7,
    input_dropout_ratio=0.3
)
%time m2.train(X, y, train, validation_frame=valid)
# CPU times: user 1.3 s, sys: 64 ms, total: 1.37 s
# Wall time: 2min 4s

m2.model_performance(test)
# MSE: 0.09892396523954171
# RMSE: 0.31452180407650865
# LogLoss: 0.5825130798587204

# not overfitted because of early stooping (error rate is not increasing for validation set)
m2.plot()

# saving models
fname1 = h2o.save_model(m1, "/home/dpejcoch/")
fname2 = h2o.save_model(m2, "/home/dpejcoch/")

