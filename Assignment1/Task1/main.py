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

	# def __init__(self):
		# features = ['What programme are you in?', 'Have you taken a course on machine learning?']
		# print(self.data['What programme are you in?'].value_counts())

	def process_study_programme(self):
		studies = ['Computer Science', 'Artificial Intelligence', 'Business Analytics', 'Econometrics', 'Bioinformatics and Systems Biology', 'Computational Science', 'Finance', 'Other']
		for element in self.data['What programme are you in?']:
			print(element)

if __name__ == '__main__':
	a = Assignment()
	a.process_study_programme()