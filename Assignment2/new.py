import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# machine learning
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# from sklearn import linear_model
# from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# from sklearn.naive_bayes import GaussianNB
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import model_selection, metrics   #Additional scklearn functions
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV   #Perforing grid search
# from sklearn.metrics import explained_variance_score
# from xgboost import XGBRegressor


pd.set_option('display.max_rows', 100)
train_data = './data/training_set_VU_DM.csv'
test_data = './data/test_set_VU_DM.csv'

test = pd.read_csv(test_data, usecols=["srch_id", "date_time", 'site_id', 'visitor_location_country_id', 'visitor_hist_adr_usd', 'visitor_hist_starrating','prop_country_id','prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
'prop_location_score1', 'prop_location_score2','price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
'srch_children_count', 'srch_room_count','srch_query_affinity_score' ,'orig_destination_distance','random_bool'])

train = pd.read_csv(train_data, usecols=['date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_adr_usd', 'visitor_hist_starrating','prop_country_id','prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
'prop_location_score1', 'prop_location_score2', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
'srch_children_count', 'srch_room_count', 'srch_query_affinity_score', 'orig_destination_distance','random_bool','click_bool', 'booking_bool'], nrows=1000000)

srch_id = test.srch_id.values
prop_id = test.prop_id.values

# date_time
train['year']  = train['date_time'].apply(lambda x: int(str(x)[:4]) if x == x else np.nan)
train['year'] = train['year'].map(lambda x: 0 if x == 2012 else 1)
train['month']  = train['date_time'].apply(lambda x: int(str(x)[5:7]) if x == x else np.nan)
# train['month'] = train['month'].map(lambda x: 0 if x <=3 else (1 if x>3 and x <=6 else (2 if x>6 and x<=9 else 3)))
train = train.drop('date_time', axis=1)

test['year']  = test['date_time'].apply(lambda x: int(str(x)[:4]) if x == x else np.nan)
test['year'] = test['year'].map(lambda x: 0 if x == 2012 else 1)
test['month']  = test['date_time'].apply(lambda x: int(str(x)[5:7]) if x == x else np.nan)
test = test.drop(['date_time', 'srch_id'], axis=1)

train['orig_destination_distance'].fillna(train.groupby(['prop_country_id','visitor_location_country_id'])['orig_destination_distance'].transform("median"), inplace=True)
train['orig_destination_distance'].fillna(train.groupby('site_id')['orig_destination_distance'].transform("median"), inplace=True)
train['orig_destination_distance'].fillna(train.groupby('prop_country_id')['orig_destination_distance'].transform("median"), inplace=True)
train = train.drop(['site_id'], axis=1)

test['orig_destination_distance'].fillna(test.groupby(['prop_country_id','visitor_location_country_id'])['orig_destination_distance'].transform("median"), inplace=True)
test['orig_destination_distance'].fillna(test.groupby('site_id')['orig_destination_distance'].transform("median"), inplace=True)
test = test.drop(['site_id',], axis=1)

# test['orig_destination_distance'].fillna(test.groupby('prop_country_id')['orig_destination_distance'].transform("median"), inplace=True)

#search query affinity
train['srch_query_affinity_score'] = train['srch_query_affinity_score'].map(lambda x: 6 if x >=-4 else (5 if x>-13 and x<-4 else (4 if x>-21 and x<=-13 else (3 if x>-31 and x<=-21 else (2 if x>-56 and x <=-31 else(1 if x>-150 and x <=-56 else 0) )))))
test['srch_query_affinity_score'] = test['srch_query_affinity_score'].map(lambda x: 6 if x >=-4 else (5 if x>-13 and x<-4 else (4 if x>-21 and x<=-13 else (3 if x>-31 and x<=-21 else (2 if x>-56 and x <=-31 else(1 if x>-150 and x <=-56 else 0) )))))

# based on the correlation data mean fill review score
train['prop_review_score'] = train.groupby('prop_starrating').prop_review_score.apply(lambda x: x.fillna(round( x.median() ) /2 ) )
test['prop_review_score'] = test.groupby('prop_starrating').prop_review_score.apply(lambda x: x.fillna(round( x.median() ) /2 ) )

# #price_usd
train['price_usd'] = train.price_usd.apply(lambda x: round(x) )
# q = train.price_usd.quantile(0.999)
train['price_usd'] = train['price_usd'].apply(lambda x: 5000 if x > 5000 else x) 
# train['price_usd'] = train.groupby('prop_starrating').price_usd.apply(lambda x: x.fillna(x.median() ) )

test['price_usd'] = test.price_usd.apply(lambda x: round(x) )
# r = test.price_usd.quantile(0.999)
test['price_usd'] = test['price_usd'].apply(lambda x: 5000 if x > 5000 else x) 
# test['price_usd'] = test.groupby('prop_starrating').price_usd.apply(lambda x: x.fillna(x.median() ) )
# test['price_usd'] = train['price_usd'].map(lambda x: 0 if x < 50 else (1 if x>=50 and x<100 else (2 if x >=100 and x <150 else (3 if x >=150 and x <200 else(4 if x >=200 and x <300 else (5 if x >=300 and x <400 else 6) ) ))))

#prop_location score
train['prop_location_score2'] = train.groupby('prop_location_score1').prop_location_score2.apply(lambda x: x.fillna(x.median() ) )
train['prop_location_score2'] = train.groupby('prop_starrating').prop_location_score2.apply(lambda x: x.fillna(x.median() ) )

test['prop_location_score2'] = test.groupby('prop_location_score1').prop_location_score2.apply(lambda x: x.fillna(x.median() ) )
test['prop_location_score2'] = test.groupby('prop_starrating').prop_location_score2.apply(lambda x: x.fillna(x.median() ) )

# people
train['people_count'] = train.srch_adults_count + train.srch_children_count
train = train.drop(['srch_adults_count', 'srch_children_count'], axis=1)

test['people_count'] = test.srch_adults_count + test.srch_children_count
test = test.drop(['srch_adults_count', 'srch_children_count'], axis=1)

#history data
train['has_history'] = train.apply(lambda x: 0 if np.isnan(x['visitor_hist_starrating']) and np.isnan(x['visitor_hist_adr_usd']) else 1, axis=1) 
train = train.drop(['visitor_hist_starrating', 'visitor_hist_adr_usd'], axis=1)

test['has_history'] = test.apply(lambda x: 0 if np.isnan(x['visitor_hist_starrating']) or np.isnan(x['visitor_hist_adr_usd']) else 1, axis=1) 
test = test.drop(['visitor_hist_starrating', 'visitor_hist_adr_usd'], axis=1)

# #Hotelrecommendation
# train['recommendation'] = train.click_bool + train.booking_bool
# train = train.drop(['click_bool', 'booking_bool'], axis=1)

test['promotion_flag'] = test['promotion_flag'].fillna(test['promotion_flag'].mode()[0])
test['srch_length_of_stay'].fillna(test['srch_length_of_stay'].mode()[0], inplace=True)
test['srch_booking_window'].fillna(test['srch_booking_window'].mode()[0], inplace=True)
test['srch_room_count'].fillna(test['srch_room_count'].mode()[0], inplace=True)
test['orig_destination_distance'].fillna(test['orig_destination_distance'].mean(), inplace=True)
test['srch_destination_id'].fillna(test['srch_destination_id'].mode()[0], inplace=True)
test['random_bool'].fillna(test['random_bool'].mode()[0], inplace=True)
test['people_count'].fillna(test['people_count'].mode()[0], inplace=True)


# test = test.dropna()
test['random_bool'] = test.random_bool.astype(int)
test['people_count'] = test.people_count.astype(int)
test['srch_room_count'] = test.srch_room_count.astype(int)
test['srch_booking_window'] = test.srch_booking_window.astype(int)
test['srch_length_of_stay'] = test.srch_length_of_stay.astype(int)
test['promotion_flag'] = test.promotion_flag.astype(int)
test['srch_destination_id'] = test.srch_destination_id.astype(int)

#convert ints and floats
gl_int = train.select_dtypes(include=['int'])
converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
compare_ints.apply(pd.Series.value_counts)

gl_float = train.select_dtypes(include=['float'])
converted_float = gl_float.apply(pd.to_numeric,downcast='float')
compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['before','after']
compare_floats.apply(pd.Series.value_counts)

train[converted_int.columns] = converted_int
train[converted_float.columns] = converted_float
# #-----------------------------------------------
gl_int = test.select_dtypes(include=['int'])
converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
compare_ints.apply(pd.Series.value_counts)

gl_float = test.select_dtypes(include=['float'])
converted_float = gl_float.apply(pd.to_numeric,downcast='float')
compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['before','after']
compare_floats.apply(pd.Series.value_counts)

test[converted_int.columns] = converted_int
test[converted_float.columns] = converted_float

print(train.info())
print(test.info(verbose=True, null_counts=True))

y = train['booking_bool'].values
X = train.drop(['booking_bool','click_bool'], axis=1).values 
X_test = test.values

model_xgb = RandomForestRegressor()
parameters = {'n_estimators': [50, 140, 1200], 'max_depth':[3,5,7,9], 'min_samples_split':[2, 5, 10],'min_samples_leaf':[1, 2, 4], 'max_features': ['auto', 'sqrt'] }
grid_search = GridSearchCV(estimator=model_xgb, param_grid=parameters, cv=10, n_jobs=-1)
print("parameters:")
# pprint.pprint(parameters)
grid_search.fit(X, y)
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters=grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
# end_time = datetime.datetime.now()
# print 'Select Done..., Time Cost: %d' % ((end_time - start_time).seconds) 


# model_xgb.fit(X, y)

# # # clf = SVR(gamma='scale', C=1.0, epsilon=0.2) 
# # # clf.fit(X, y) 
# # # SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
# # # kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

# y_pred = model_xgb.predict_proba(X_test)

# print(y_pred)

# pred = y_pred[0:,[1]]
# pred = np.hstack(pred)

# kaggle = pd.DataFrame({'srch_id': srch_id, 'prop_id':prop_id, 'prediction':pred})
# df = kaggle.set_index(['prop_id']).groupby('srch_id')['prediction'].nlargest(50).reset_index()
# df = df.drop('prediction', axis=1)

# df.to_csv('./expedia2.csv', index=False) 

# pred = 5*y_pred[0:,[2]] + y_pred[0:,[1]]
# pred = np.hstack(pred)
# # destination = X_test['srch_destination_id'].values
# # prop = X_test['prop_id'].values

# kaggle = pd.DataFrame({'srch_id': srch_id, 'prop_id':prop_id, 'prediction':pred})
# df = kaggle.set_index(['prop_id']).groupby('srch_id')['prediction'].nlargest(40).reset_index()
# df = df.drop('prediction', axis=1)

# df.to_csv('./expedia.csv', index=False) 

