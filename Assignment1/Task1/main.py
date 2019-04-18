import datetime
from utils import Utils
from visualize import Visualize
import pandas as pd

class Assignment:

	path = 'dataset.csv'
	data = pd.read_csv(path, sep=';', header=0, engine='python')

	def __init__(self):
		self.explore()
		self.normalize_data(self.data)

	def normalize_data(self, data):
		u = Utils()

		# Drop timestamp as they are almost identical across the dataset
		data = data.drop(['Timestamp', data.columns[-1]], axis=1)
		# data = u.map_option_questions(data)
		# data = u.chocolate_map(data)
		# data = u.date_conversion(data)
		# data = u.neighbour_conversion(data)
		# print(data[data.columns[10]])

	def explore(self):
		print(self.data.shape)
		print(self.data.count())

if __name__ == '__main__':
	pd.set_option('display.max_rows', None)
	# pd.set_option('display.max_columns', None)  
	# pd.set_option('display.expand_frame_repr', False)
	# pd.set_option('max_colwidth', -1)

	a = Assignment()	