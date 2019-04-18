import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

class Utils:

	# Map - no, yes, unknown questions to uniform format 0, 1, 2
	def map_option_questions(self, data):
		yes_map = {'no': 0, 'yes': 1,  'unknown': 2}
		data[data.columns[1]] = data[data.columns[1]].map(yes_map)
		data[data.columns[9]] = data[data.columns[9]].map(yes_map)

		one_map = {'0': 0, '1': 1,  'unknown': 2}
		data[data.columns[2]] = data[data.columns[2]].map(one_map)

		mu_map = {'sigma': 0, 'mu': 1, 'unknown': 2}
		data[data.columns[3]] = data[data.columns[3]].map(mu_map)

		ja_map = {'nee': 0, 'ja': 1, 'unknown': 2}
		data[data.columns[4]] = data[data.columns[4]].map(ja_map)

		gender_mapping = {'female': 0, 'male': 1, 'unknown': 2}
		data[data.columns[5]] = data[data.columns[5]].map(gender_mapping)

		return data

	# Chocolate mapping 'slim': 0, 'fat': 1, 'neither': 2, 'I have no idea what you are talking about': 3, 'unknown': 4
	def chocolate_map(self, data):
		chocolate_map = {'slim': 0, 'fat': 1, 'neither': 2, 'I have no idea what you are talking about': 3, 'unknown': 4}
		data[data.columns[6]] = data[data.columns[6]].map(chocolate_map)

		return data

	# Date conversion, we want format: dd/MM/YYYY
	def date_conversion(self, data):

		data[data.columns[7]] = data[data.columns[7]].str.lower()
		data[data.columns[7]] = data[data.columns[7]].str.strip()
		data[data.columns[7]] = data[data.columns[7]].str.replace("-", "/")
		data[data.columns[7]] = data[data.columns[7]].str.replace(".", "/")
		data[data.columns[7]] = data[data.columns[7]].str.replace(",", "/")
		data[data.columns[7]] = data[data.columns[7]].str.replace(' ', '')

		self.compile_with_value(data, re.compile('.{2}/.{2}/.{2}'), 7, 8, '19', 6)
		self.replace_months(data, 7)
		self.replace_other(data, 7)

		return data

	# 8 How many neighbours are sitting around you -> Conversion > 276 = 276, all string values convert to mean, Also replace outlier 276 with mean
	# data.loc[data[data.columns[8]] > 275 , data.columns[8]] = 276
	def neighbour_conversion(self, data):
		data[data.columns[8]] = data[data.columns[8]].str.replace('Zeven\"\'\:\;', '7')
		data[data.columns[8]] = data[data.columns[8]].str.replace('four (4)', '4')
		data[data.columns[8]] = data[data.columns[8]].str.replace('Twenty four', '24')

		data.loc[data[data.columns[8]].str.contains('four'), data.columns[8]] = '4'

		for i in range(len(data)):
			element = data[data.columns[8]][i]
			e = re.sub("[^0-9]", "", element)
			data[data.columns[8]][i] = e

		data[data.columns[8]] = data[data.columns[8]].str.strip()

		for i in range(len(data)):
			element = data[data.columns[8]][i]
			if len(element) == 0:
				e = 0
				data[data.columns[8]][i] = e

		for i in range(len(data)):
			element = data[data.columns[8]][i]
			if int(element) > 275:
				data[data.columns[8]][i] = 276

		return data

	def compile_with_value(self, data, r, column, length_string, value, insert_at):
		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if len(element) == length_string:
				if r.match(element) is not None:
					e = element[:insert_at] + value + element[insert_at:]
					data[data.columns[column]][i] = e

	def replace_months(self, data, column):
		month_number = ['/01/', '/02/', '/03/', '/04/', '/05/', '/06/', '/07/', '/08/', '/09/', '/10/', '11/', '/12/']
		months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
		
		i = 0
		for month in months:
			data[data.columns[column]] = data[data.columns[column]].str.replace(month, month_number[i])
			i = i + 1

	def replace_other(self, data, column):
		data[data.columns[column]] = data[data.columns[column]].str.replace('negentienhonderdtweeennegentig', '1992')
		data[data.columns[column]] = data[data.columns[column]].str.replace('tien', '10')
		data[data.columns[column]] = data[data.columns[column]].str.replace('unknown', '00/00/0000')
		data[data.columns[column]] = data[data.columns[column]].str.replace('kingsday', '30/04/0000')
		data[data.columns[column]] = data[data.columns[column]].str.replace('#', '00/00/0000')
		data[data.columns[column]] = data[data.columns[column]].str.replace('kuhbj', '00/00/0000')
		data[data.columns[column]] = data[data.columns[column]].str.replace('tomorrow', '01/04/0000')
		data[data.columns[column]] = data[data.columns[column]].str.replace('oct', '/10/')
		data[data.columns[column]] = data[data.columns[column]].str.replace('juni', '/06/')
		data[data.columns[column]] = data[data.columns[column]].str.replace('feb', '/02/')
		data[data.columns[column]] = data[data.columns[column]].str.replace('aug', '/08/')
		data[data.columns[column]] = data[data.columns[column]].str.replace('dec', '/12/')
		data[data.columns[column]] = data[data.columns[column]].str.replace('septembre', '/09/')
		data[data.columns[column]] = data[data.columns[column]].str.replace('19--', '0000')
		data[data.columns[column]] = data[data.columns[column]].str.replace('\'92', '1992')

		data.loc[data[data.columns[column]].str.contains('x'), data.columns[column]] = '00/00/0000'
		data.loc[data[data.columns[column]].str.contains('classified'), data.columns[column]] = '00/00/0000'
		data.loc[data[data.columns[column]].str.contains('firstof'), data.columns[column]] = '01/10/0000'
		data.loc[data[data.columns[column]].str.contains("\+"), data.columns[column]] = '23/04/1992'

		r_months = re.compile('.{2}/.{1}/.{2}')
		for i in range(len(data)):
			element = data[data.columns[7]][i]
			if r_months.match(element):
				e = element[:3] + '0' + element[3:]
				data[data.columns[7]][i] = e

		r_months = re.compile('.{1}/.{2}/.{2}')
		for i in range(len(data)):
			element = data[data.columns[7]][i]
			if r_months.match(element):
				e = element[:0] + '0' + element[0:]
				data[data.columns[7]][i] = e

		r_months = re.compile('.{1}/.{1}/.{2}')
		for i in range(len(data)):
			element = data[data.columns[7]][i]
			if r_months.match(element):
				e = element[:0] + '0' + element[0:]
				e = e[:3] + '0' + e[3:]
				data[data.columns[7]][i] = e

		for i in range(len(data)):
			element = data[data.columns[7]][i]
			if len(element) == 4 and '/' not in element:
				if int(element) > 1900:
					e = "00/00/{}".format(element)
					data[data.columns[7]][i] = e

		for i in range(len(data)):
			element = data[data.columns[7]][i]
			if len(element) == 8 and '/' not in element:
				e = element[:2] + '/' + element[2:]
				e = e[:5] + '/' + e[5:]
				data[data.columns[7]][i] = e

		for i in range(len(data)):
			element = data[data.columns[7]][i]
			if len(element) != 10:
				e = "00/00/0000"
				data[data.columns[7]][i] = e

		data[data.columns[column]] = data[data.columns[column]].str.replace('28/11/19——', '28/11/0000')