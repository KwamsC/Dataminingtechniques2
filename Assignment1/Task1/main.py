import datetime
from utils import Utils
from visualize import Visualize
import pandas as pd

class Assignment:

	path = 'dataset.csv'
	data = pd.read_csv(path, sep=';', header=0, engine='python')

	# converted_csv_path = 'new.csv'
	# converted_data = pd.read_csv(path, sep=',', header=0, engine='python')

	def __init__(self):
		# self.explore()
		self.normalize_data(self.data)

	def normalize_data(self, data):
		u = Utils()

		# Drop timestamp as they are almost identical across the dataset
		data = data.drop('Timestamp', axis=1)
		print(data[data.columns[-1]])
		# data = u.programme_conversion(data, 0)
		# data = u.map_option_questions(data)
		# data = u.chocolate_map(data)
		# data = u.date_conversion(data)
		# data = u.neighbour_conversion(data)
		# data = u.competition_conversion(data, 10)
		# data = u.random_number_conversion(data, 11)
		# data = u.bed_time_conversion(data, 12)
		# data = u.good_day_conversion(data, 13)
		# data = u.good_day_conversion(data, 14)
		data = u.stress_conversion(data, 15)
		print(data[data.columns[-1]])

		# print(data[data.columns[13]])
		# data.to_csv('./new.csv')

	def explore(self):
		print(self.data.shape)
		print(self.data.count())

if __name__ == '__main__':
	pd.set_option('display.max_rows', None)
	# pd.set_option('display.max_columns', None)  
	# pd.set_option('display.expand_frame_repr', False)
	# pd.set_option('max_colwidth', -1)

	a = Assignment()	