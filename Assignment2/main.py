import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import model_selection, metrics   #Additional scklearn functions
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import explained_variance_score

class Assignment:
	# pd.set_option('display.max_columns', 54)
	train_data = './data/training_set_VU_DM.csv'
	test_data = './data/test_set_VU_DM.csv'
	train = pd.read_csv(train_data,usecols=["date_time", 'visitor_location_country_id', 'visitor_hist_adr_usd', 'visitor_hist_starrating','prop_country_id','prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
	'prop_location_score1', 'prop_location_score2', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
	'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool', 'click_bool', 'booking_bool'] , nrows=200000)
	test = pd.read_csv(test_data, usecols=["srch_id", "date_time", 'visitor_location_country_id', 'visitor_hist_adr_usd', 'visitor_hist_starrating','prop_country_id','prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
	'prop_location_score1', 'prop_location_score2', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
	'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool'])

	srch_id = test.srch_id.values
	prop_id = test.prop_id.values

	# #drop
	# train = train.drop(['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 
	# 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
	# 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
	# 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'position', 'site_id','srch_id', 'srch_query_affinity_score','prop_log_historical_price'], axis=1)

	# date_time
	train['year']  = train['date_time'].apply(lambda x: int(str(x)[:4]) if x == x else np.nan)
	train['year'] = train['year'].map(lambda x: 0 if x == 2012 else 1)
	train['month']  = train['date_time'].apply(lambda x: int(str(x)[5:7]) if x == x else np.nan)
	train = train.drop('date_time', axis=1)

	test['year']  = test['date_time'].apply(lambda x: int(str(x)[:4]) if x == x else np.nan)
	test['year'] = test['year'].map(lambda x: 0 if x == 2012 else 1)
	test['month']  = test['date_time'].apply(lambda x: int(str(x)[5:7]) if x == x else np.nan)
	test = test.drop('date_time', axis=1)

	# grouped_affinity = train.groupby('visitor_hist_starrating')
	# train['srch_query_affinity_score'].fillna(grouped_affinity['srch_query_affinity_score'].transform("median"), inplace=True)
	# grouped_affinity = train.groupby('prop_starrating')
	# train['srch_query_affinity_score'].fillna(grouped_affinity['srch_query_affinity_score'].transform("median"), inplace=True)

	# test_grouped_affinity = test.groupby('visitor_hist_starrating')
	# test['srch_query_affinity_score'].fillna(test_grouped_affinity['srch_query_affinity_score'].transform("median"), inplace=True)
	# grouped_affinity = test.groupby('prop_starrating')
	# test['srch_query_affinity_score'].fillna(grouped_affinity['srch_query_affinity_score'].transform("median"), inplace=True)

	# #prop log istorical price
	# train['prop_log_historical_price'] = train['prop_log_historical_price'].replace(0, np.nan)
	# train['prop_log_historical_price'].fillna(train.groupby(['prop_starrating'])['prop_log_historical_price'].transform("median"), inplace=True)
	# train['prop_log_historical_price'] = train['prop_log_historical_price'].round(1)
	# train['prop_log_historical_price'] = train['prop_log_historical_price'].map(lambda x:0 if x < 4 else (1 if x>=4 and x<5 else (2 if x >=5 and x <6 else 3)))

	# test['prop_log_historical_price'] = test['prop_log_historical_price'].replace(0, np.nan)
	# test['prop_log_historical_price'].fillna(test.groupby(['prop_starrating'])['prop_log_historical_price'].transform("median"), inplace=True)
	# test['prop_log_historical_price'] = test['prop_log_historical_price'].round(1)
	# test['prop_log_historical_price'] = test['prop_log_historical_price'].map(lambda x:0 if x < 4 else (1 if x>=4 and x<5 else (2 if x >=5 and x <6 else 3)))

	# based on the correlation data mean fill review score
	grouped_score = train.groupby('prop_starrating')
	train['prop_review_score'] = grouped_score.prop_review_score.apply(lambda x: x.fillna(round( x.median() ) /2 ) )

	test_grouped_score = test.groupby('prop_starrating')
	test['prop_review_score'] = test_grouped_score.prop_review_score.apply(lambda x: x.fillna(round( x.median() ) /2 ) )

	#price_usd
	train['price_usd'] = train['price_usd'].map(lambda x: 0 if x < 50 else (1 if x>=50 and x<100 else (2 if x >=100 and x <150 else (3 if x >=150 and x <200 else(4 if x >=200 and x <300 else (5 if x >=300 and x <400 else 6) ) ))))
		
	test['price_usd'] = train['price_usd'].map(lambda x: 0 if x < 50 else (1 if x>=50 and x<100 else (2 if x >=100 and x <150 else (3 if x >=150 and x <200 else(4 if x >=200 and x <300 else (5 if x >=300 and x <400 else 6) ) ))))

	#prop_location score
	grouped_loc_score = train.groupby('prop_location_score1')
	train['prop_location_score2'] = grouped_loc_score.prop_location_score2.apply(lambda x: x.fillna(x.median() ) )

	test_grouped_loc_score = test.groupby('prop_location_score1')
	test['prop_location_score2'] = test_grouped_loc_score.prop_location_score2.apply(lambda x: x.fillna(x.median() ) )

	# grouped_gross_price = train.groupby('srch_length_of_stay')
	# train['gross_bookings_usd'] = grouped_gross_price.gross_bookings_usd.apply(lambda x: x.fillna(round(x.median()) ) )
	# grouped_gross_price = train.groupby('prop_starrating')
	# train['gross_bookings_usd'] = grouped_gross_price.gross_bookings_usd.apply(lambda x: x.fillna(round(x.median()) ) )

	# test['gross_bookings_usd'] = test.groupby('srch_length_of_stay').gross_bookings_usd.apply(lambda x: x.fillna(round(x.median()) ) )
	# test['gross_bookings_usd'] = test.groupby('prop_starrating').gross_bookings_usd.apply(lambda x: x.fillna(round(x.median()) ) )

	#people
	train['people_count'] = train.srch_adults_count + train.srch_children_count
	train = train.drop(['srch_adults_count', 'srch_children_count'], axis=1)

	test['people_count'] = test.srch_adults_count + test.srch_children_count
	test = test.drop(['srch_adults_count', 'srch_children_count','srch_id'], axis=1)

	#history data
	train['has_history'] = train.apply(lambda x: 0 if np.isnan(x['visitor_hist_starrating']) or np.isnan(x['visitor_hist_adr_usd']) else 1, axis=1) 
	train = train.drop(['visitor_hist_starrating', 'visitor_hist_adr_usd'], axis=1)

	test['has_history'] = test.apply(lambda x: 0 if np.isnan(x['visitor_hist_starrating']) or np.isnan(x['visitor_hist_adr_usd']) else 1, axis=1) 
	test = test.drop(['visitor_hist_starrating', 'visitor_hist_adr_usd'], axis=1)

	#Hotelrecommendation
	train['recommendation'] = train.click_bool + train.booking_bool
	train = train.drop(['click_bool', 'booking_bool'], axis=1)

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
	#-----------------------------------------------
	#convert ints and floats
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

	print(test.info())
	print(train.info())

	y = train['recommendation'].values
	X = train.drop('recommendation', axis=1).values 

	X_test = test.values

	from xgboost import XGBRegressor
	model_xgb = XGBRegressor()
	model_xgb = XGBRegressor(objective = 'multi:softprob', colsample_bytree = 0.9, max_depth=3, alpha=10, n_estimators=100, learning_rate=.1, num_class=3)

	model_xgb.fit(X, y)
	# parameters = {'n_estimators': [50, 140, 120], 'max_depth':[3,5,7,9], 'colsample_bytree':[0.3, 0.6, 0.9] }
	# grid_search = GridSearchCV(estimator=model_xgb, param_grid=parameters, cv=10, n_jobs=-1)
	# print("parameters:")
	# # pprint.pprint(parameters)
	# grid_search.fit(X, y)
	# print("Best score: %0.3f" % grid_search.best_score_)
	# print("Best parameters set:")
	# best_parameters=grid_search.best_estimator_.get_params()
	# for param_name in sorted(parameters.keys()):
	# 	print("\t%s: %r" % (param_name, best_parameters[param_name]))
	# # end_time = datetime.datetime.now()
	# # print 'Select Done..., Time Cost: %d' % ((end_time - start_time).seconds) 

	y_pred = model_xgb.predict(X_test)

	pred = 4*y_pred[0:,[2]] + y_pred[0:,[1]]
	pred = np.hstack(pred)
	# destination = X_test['srch_destination_id'].values
	# prop = X_test['prop_id'].values

	kaggle = pd.DataFrame({'srch_id': srch_id, 'prop_id':prop_id, 'prediction':pred})
	df = kaggle.set_index(['prop_id']).groupby('srch_id')['prediction'].nlargest(40).reset_index()
	df = df.drop('prediction', axis=1)

	df.to_csv('./expedia.csv', index=False) 

if __name__ == '__main__':
	a = Assignment()