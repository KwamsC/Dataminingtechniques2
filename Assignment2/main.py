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
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import explained_variance_score

class Assignment:
	# pd.set_option('display.max_columns', 54)
	pd.set_option('display.max_rows', 160)

	train_data = './data/training_set_VU_DM.csv'
	test_data = './data/test_set_VU_DM.csv'
	train = pd.read_csv(train_data, nrows=100000)
	test = pd.read_csv(test_data, nrows=100000)

	srchid = test.srch_id

	# train = train.append(test, ignore_index=True)


	#drop
	train = train.drop(['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 
	'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
	'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
	'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff'], axis=1)

	#1 visualize
	## save a histogram of the prices
	# train['price_usd'].plot.hist(bins=100,range=(0,600))
	# plt.xlabel('Booking price (USD)', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../hist_price_usd.png', dpi=600)
	# plt.clf()

	# 1 visualize
	# save a histogram of the prices
	# train['visitor_hist_starrating'].plot.hist()
	# plt.xlabel('Booking price (USD)', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../visitor_hist_starrating.png', dpi=600)
	# plt.clf()

	# # 1 visualize
	# # save a histogram of the prices
	# train['visitor_hist_adr_usd'].plot.hist()
	# plt.grid(True)
	# plt.savefig('../visitor_hist_adr_usd.png', dpi=600)
	# plt.clf()

	# # 1 visualize
	# # save a histogram of the prices
	# train['orig_destination_distance'].plot.hist()
	# plt.grid(True)
	# plt.savefig('../orig_destination_distance.png', dpi=600)
	# plt.clf()

	## save a histogram of the length of stay
	# xmax = 15
	# df_concat['srch_length_of_stay'].plot.hist(bins=xmax,range=(1,xmax))
	# plt.xticks(range(xmax+1))
	# plt.xlabel('Length of stay (number of nights)', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../hist_srch_length_of_stay.png', dpi=600)
	# plt.clf()

	# based on the correlation data mean fill review score
	grouped_score = train.groupby('prop_starrating')
	# train["prop_review_score"].fillna(round(train.groupby('prop_starrating')["prop_review_score"].transform("median")/2), inplace=True)
	train['prop_review_score'] = grouped_score.prop_review_score.apply(lambda x: x.fillna(round( x.median() ) /2 ) )

	
	train['price_usd'] = train['price_usd'].map(lambda x: 0 if x < 50 else (1 if x>=50 and x<100 else (2 if x >=100 and x <150 else (3 if x >=150 and x <200 else(4 if x >=200 and x <300 else (5 if x >=300 and x <400 else 6) ) ))))

	# print(train.prop_review_score.value_counts())
	# grouped_fare = titanic.groupby(['Fare','Pclass'])  
	# train['prop_review_score'] = train['prop_review_score'].apply(lambda x: x.fillna(round(x.median())))
	# train.prop_review_score = train.prop_review_score.apply(lambda x: x.fillna(round(x.median())))

	# train['prop_review_score'].plot.hist(bins=10,range=(0,5.1))
	# # plt.xticks(range(xmax+1))
	# plt.xlabel('Review score', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../review_score.png', dpi=600)
	# plt.clf()

	# #barchart
	# booked = train[train['booking_bool']==1]['prop_review_score'].value_counts()
	# not_booked = train[train['booking_bool']==0]['prop_review_score'].value_counts()
	# df = pd.DataFrame([not_booked])
	# # df.index = 'booked'
	# df.plot(kind='bar',stacked=True, figsize=(10,5))
	# plt.show()
	# plt.clf()

	## save a histogram of days between search and trip
	# train['srch_booking_window'].plot.hist(bins=100,range=(1,250))
	# plt.xlabel('Number of days between search and trip', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../hist_srch_booking_window.png', dpi=600)
	# plt.clf()

	# # save a histogram of distance between search and trip
	# train['orig_destination_distance'].plot.hist(bins=100, range=(0,11000))
	# plt.xlabel('Distance between hotel and search location', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../distance_hotel_search_location.png', dpi=600)
	# plt.clf()


	# # save histogram log price
	# train['prop_location_score1'].plot.hist(bins=100)
	# plt.xlabel('prop_location_score1', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../prop_location_score1.png', dpi=600)
	# plt.clf()

	# train['prop_location_score2'].plot.hist(bins=100)
	# plt.xlabel('prop_location_score2', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../prop_location_score2.png', dpi=600)
	# plt.clf()
	
	grouped_loc_score = train.groupby('prop_location_score1')
	train['prop_location_score2'] = grouped_loc_score.prop_location_score2.apply(lambda x: x.fillna(x.median() ) )
	# train['prop_loc_score'] = train.prop_location_score1 + train.prop_location_score2
	# train = train.drop(['prop_location_score1'], axis=1)

	# train['gross_bookings_usd'].plot.hist()
	# plt.savefig('../gross_bookings_usd.png', dpi=600)
	# plt.clf()

	grouped_gross_price = train.groupby('srch_length_of_stay')
	train['gross_bookings_usd'] = grouped_gross_price.gross_bookings_usd.apply(lambda x: x.fillna(round(x.median()) ) )

	# drop more
	train = train.drop(['site_id', 'srch_query_affinity_score', 'orig_destination_distance', 'gross_bookings_usd','position'], axis=1)

	# fig, ax = plt.subplots()
	# fig.set_size_inches(13, 8)
	# sns.countplot('orig_destination_distance',data=train[train["booking_bool"] == 1],order=list(range(0,11000)),ax=ax)
	# plt.ylabel('Booking',fontsize=8)
	# plt.show()

	# ## save a histogram of bookings per month
	# fig, ax = plt.subplots()
	# sns.countplot('Month',data=train[train['booking_bool'] == 1],order=list(range(1,17)),ax=ax)
	# plt.savefig('../booking_per_month.png', dpi=600)
	# plt.clf()

	# f, ax = plt.subplots()
	# corr = train.corr()

	# mask = np.zeros_like(corr, dtype=np.bool)
	# mask[np.triu_indices_from(mask)] = True

	# # Set up the matplotlib figure
	# f, ax = plt.subplots(figsize=(11, 9))

	# # Generate a custom diverging colormap
	# cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# # Draw the heatmap with the mask and correct aspect ratio
	# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
	# plt.savefig('../feature_correlation_heatmap.png', dpi=300,bbox_inches='tight')
	# plt.clf()

	# #missing data
	# total = train.isnull().sum().sort_values(ascending=False)
	# percent = (train.isnull().sum()/train['prop_id'].count()).sort_values(ascending=False)
	# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
	# missing_data = missing_data.drop('Percent', axis = 1)
	# plt.subplots()
	# missing_data.plot(kind = 'Bar')
	# plt.grid(True)
	# plt.tight_layout()
	# plt.savefig('../missing_data.png', dpi=600)
	# plt.clf()

	# hist_group = train.groupby(['visitor_hist_starrating','visitor_hist_adr_usd']) 
	train['has_history'] = train.apply(lambda x: 0 if np.isnan(x['visitor_hist_starrating']) or np.isnan(x['visitor_hist_adr_usd']) else 1, axis=1) 
	# drop
	train = train.drop(['visitor_hist_starrating', 'visitor_hist_adr_usd'], axis=1)

	train['year']  = train['date_time'].apply(lambda x: int(str(x)[:4]) if x == x else np.nan)
	train['month']  = train['date_time'].apply(lambda x: int(str(x)[5:7]) if x == x else np.nan)
	# train['day']  = train['date_time'].apply(lambda x: int(str(x)[8:10]) if x == x else np.nan)
	train = train.drop('date_time', axis=1)

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

	# agg = train.groupby(['srch_destination_id','prop_id'])['booking_bool'].agg(['sum','count'])
	# agg.reset_index(inplace=True)
	# print(agg.head(150))
	# CLICK_WEIGHT = 0.05
	# agg = agg.groupby(['srch_destination_id','prop_id']).sum().reset_index()
	# agg['count'] -= agg['sum']
	# agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
	# agg['relevance'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']
	# print(agg.head(50))

	#Hotelrecommendation
	train['recommendation'] = train.click_bool + train.booking_bool
	recommendation = train['recommendation']
	train = train.drop(['click_bool', 'booking_bool','recommendation'], axis=1)	
	print(train.info())

	from sklearn.model_selection import train_test_split

	X_train, X_test, Y_train, Y_test = train_test_split(train, recommendation, train_size=.7)

	from xgboost import XGBRegressor
	model_xgb = XGBRegressor(objective = 'multi:softprob', colsample_bytree = 0.3, max_depth=6, alpha=10, n_estimators=10, learning_rate=.1, num_class=3)
	model_xgb.fit(X_train.drop(columns=['srch_id']), Y_train)

	y_pred_1 = model_xgb.predict(X_test.drop(columns=['srch_id']))

	pred = 10*y_pred_1[0:,[2]] + y_pred_1[0:,[1]]
	preds = np.hstack(pred)

	#cutoff value will be 2

	kaggle = pd.DataFrame({'pred': preds})

	# print(kaggle.info())

	# X_test_x= X_test[['srch_destination_id', 'prop_id']]

	# result = pd.concat([X_test_x, kaggle], axis=1, sort=False)

	print(kaggle.head())
	print(X_test.head())

	# X_test = X_test.drop(['visitor_location_country_id', 'prop_country_id', 'prop_starrating','random_bool','has_history', 'year', 'month','srch_children_count','price_usd', ''   ], axis=1)	
	# kaggle1 = pd.concat([X_test, kaggle], axis=1)

	# kaggle = pd.DataFrame(X_test, {'pred': preds})

	# print(kaggle.head(150))

	# recommendation = train['recommendation']
	# train = train.drop(['click_bool', 'recommendation', 'booking_bool'], axis=1)

	# # X_train = train.drop(columns=['click_bool','booking_bool']).values
	# # y_train = train['recommendation'].values

	# from sklearn.model_selection import train_test_split
	# X_train, X_test, y_train, y_test = train_test_split(train, recommendation_variables, test_size = 0.3)

	# rf = RandomForestRegressor(max_depth=20, random_state=123, n_estimators=100)
	# # y_pred = cross_val_predict(rf, train, y_train, cv=kf)
	# # y_pred[y_pred < 0 ] = 0
	# rf.fit(X_train, y_train) 
	# y_pred = rf.predict(X_test)

	# kaggle = pd.DataFrame({'prob0': y_pred[0], 'prob1': y_pred[1], 'prob2': y_pred[2],})
	# kaggle1 = pd.concat([X_test, kaggle])

	# print(kaggle1.head(50))
	
	# bookings_df = train[train["booking_bool"] == 1]
	# bookings_cf = bookings_df[bookings_df['prop_country_id'] == 219]

	# pieces = [train.groupby(['srch_destination_id','prop_country_id','prop_id'])['booking_bool'].agg(['sum','count'])]
	# agg = pd.concat(pieces).groupby(level=[0,1,2]).sum()
	# agg.dropna(inplace=True)
	# # print(agg.head(100))

	# agg['sum_and_cnt'] = 0.85*agg['sum'] + 0.15*agg['count']
	# agg = agg.groupby(level=[0,1]).apply(lambda x: x.astype(float)/x.sum())
	# agg.reset_index(inplace=True)

	# agg_pivot = agg.pivot_table(index=['srch_destination_id','prop_country_id'], columns='prop_id', values='sum_and_cnt').reset_index()

	# df = pd.merge(train, agg_pivot, how='left', on=['srch_destination_id','prop_country_id'])
	# df.fillna(0, inplace=True)
	# print(df.head())
	# print(agg_pivot.head())

	# print(train.corr()["prop_id"].sort_values())

	# bookings_df['prop_id'].plot.hist()
	# # plt.xlabel('Booking price (USD)', fontsize=8)
	# # plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# # plt.savefig('../results/hist_price_usd.png', dpi=600)
	# plt.show()
	# plt.clf()

	# fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,10))
	# fig.set_size_inches(13, 8)
	# # sns.countplot('day',data=train[train["booking_bool"] == 1],order=list(range(1,32)),ax=ax)
	# # plt.ylabel('Booking',fontsize=8)
	# sns.countplot('srch_destination_id',data=bookings_df.sort_values(by=['srch_destination_id']),ax=axis1,palette="Set3")
	# sns.countplot('prop_country_id',data=bookings_df.sort_values(by=['prop_country_id']),ax=axis2,palette="Set3")
	# plt.show()

	# country_id = 219
	# fig, (axis1) = plt.subplots(1,1,figsize=(15,10))
	# country_clusters = bookings_df[bookings_df["prop_country_id"] == country_id]["prop_id"]
	# country_clusters.value_counts().plot(kind='bar',colormap="Set3",figsize=(15,5))
	# plt.show()

	# # srch_room_count 
	# # booked = df_concat[df_concat['booking_bool']==1]['srch_room_count'].value_counts()
	# # not_booked = df_concat[df_concat['booking_bool']==0]['srch_room_count'].value_counts()
	# # df = pd.DataFrame([booked, not_booked])
	# # df.index = ['Booked', 'Not Booked']
	# # df.plot(kind='bar', stacked = True)

	# # f, ax = plt.subplots()
	# # corr = df_concat.corr()

	# # mask = np.zeros_like(corr, dtype=np.bool)
	# # mask[np.triu_indices_from(mask)] = True

	# # # Set up the matplotlib figure
	# # f, ax = plt.subplots(figsize=(11, 9))

	# # # Generate a custom diverging colormap
	# # cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# # # Draw the heatmap with the mask and correct aspect ratio
	# # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

	# # df_concat.hist()

	# df_concat.groupby('position').count()['click_bool'].plot.hist()
	# # df_concat.groupby(['prop_country_id','prop_id']).plot.hist(bins=100,range=(0,10000))
	# plt.xlabel('country', fontsize=8)
	# plt.ylabel('Frequency',fontsize=8)
	# plt.grid(True)
	# plt.savefig('../country_hotel.png', dpi=600)
	# plt.clf()

if __name__ == '__main__':
	a = Assignment()