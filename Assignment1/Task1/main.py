import datetime
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)  
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)

class Assignment:
	path = 'dataset.csv'
	data = pd.read_csv(path, sep=';', header=0, engine='python')

	ja_map = {'ja':1, 'nee':0, 'unknown':2}
	yes_map = {'yes':1, 'no':0, 'unknown':2}
	mu_map = {'mu':1, 'sigma':0, 'unknown':2}
	gender_mapping = {'male':1, 'female':0, 'unknown':2}


	data = data.drop(['Timestamp', data.columns[-1]], axis=1)
	# print(data.head())

	data[data.columns[4]] = data[data.columns[4]].map(ja_map)
	data[data.columns[3]] = data[data.columns[3]].map(mu_map)
	data[data.columns[1]] = data[data.columns[1]].map(yes_map)
	data[data.columns[5]] = data[data.columns[5]].map(gender_mapping)

	# print(data[data.columns[4]])

	print(data[data.columns[13]].str.lower().value_counts())

	# def __init__(self):
		# features = ['What programme are you in?', 'Have you taken a course on machine learning?']
		# print(self.data['What programme are you in?'].value_counts())

	def process_study_programme(self):
		studies = ['Computer Science', 'Artificial Intelligence', 'Business Analytics', 'Econometrics', 'Bioinformatics and Systems Biology', 'Computational Science', 'Finance', 'Other']
		for element in self.data['What programme are you in?']:
			print(element)

if __name__ == '__main__':
	a = Assignment()
	# a.process_study_programme()