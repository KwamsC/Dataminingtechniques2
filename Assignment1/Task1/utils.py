import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re, math, sys

class Utils:

	def programme_conversion(self, data, column):
		programmes = ['cs', 'ai', 'ba', 'bio', 'eco', 'computation', 'math', 'is', 'psy', 'hs', 'qrm']
		mapping = {'cs': 0, 'ai': 1, 'ba': 2, 'bio': 3, 'eco': 4, 'computation': 5, 'math': 6, 'is': 7, 'psy': 8, 'hs': 9, 'qrm': 10, 'other': 11}

		data[data.columns[column]] = data[data.columns[column]].str.lower()
		data[data.columns[column]] = data[data.columns[column]].str.strip()

		data.loc[data[data.columns[column]].str.contains('ai'), data.columns[column]] = 'ai' # Artificial Intelligence
		data.loc[data[data.columns[column]].str.contains('artificial'), data.columns[column]] = 'ai' # Artificial Intelligence
		data.loc[data[data.columns[column]].str.contains('intelligence'), data.columns[column]] = 'ai' # Artificial Intelligence
		data.loc[data[data.columns[column]].str.contains('computer'), data.columns[column]] = 'cs' # Computer Science
		data.loc[data[data.columns[column]].str.contains('cs'), data.columns[column]] = 'cs' # Computer Science
		data.loc[data[data.columns[column]].str.contains('business'), data.columns[column]] = 'ba' # Business administration
		data.loc[data[data.columns[column]].str.contains('administration'), data.columns[column]] = 'ba' # Business administration
		data.loc[data[data.columns[column]].str.contains('bio'), data.columns[column]] = 'bio' # Bioinformatics & System biology
		data.loc[data[data.columns[column]].str.contains('eco'), data.columns[column]] = 'eco' # Econometrics
		data.loc[data[data.columns[column]].str.contains('computational'), data.columns[column]] = 'computation' # Computational Science
		data.loc[data[data.columns[column]].str.contains('math'), data.columns[column]] = 'math' # Math
		data.loc[data[data.columns[column]].str.contains('info'), data.columns[column]] = 'is' # Information Science
		data.loc[data[data.columns[column]].str.contains('psy'), data.columns[column]] = 'psy' # Psychology
		data.loc[data[data.columns[column]].str.contains('health'), data.columns[column]] = 'hs' # Health Sciences
		data.loc[data[data.columns[column]].str.contains('q'), data.columns[column]] = 'qrm' # Finance

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if element not in programmes:
				data[data.columns[column]][i] = 'other'

		data[data.columns[column]] = data[data.columns[column]].map(mapping)
		return data

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

	# Convert competition numbers -> remove everything which is not digit, strings -> 0
	# Strip strings
	# There were 276 students in the room, so every value higher than > 276 becomes 276
	# Replace zero's with avg
	def competition_conversion(self, data, column):
		data[data.columns[column]] = data[data.columns[column]].str.strip()

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			e = re.sub("[^0-9]", "", element)
			e = e.strip()
			e = e.lstrip('0') 
			if e == '':
				e = 0
			data[data.columns[column]][i] = e

		data[data.columns[column]] = data[data.columns[column]].astype(float)
		data.loc[data[data.columns[column]] > 275, data.columns[column]] = 276

		x = 1
		v = 0
		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if element != 0:
				x = x + 1
				v = v + int(element)

		d = round(v/x)
		data.loc[data[data.columns[column]] == 0, data.columns[column]] = d
		data[data.columns[column]] = data[data.columns[column]].astype(int)
		return data

	# Strip -> Replace , with . -> replace empty spaces, all character based answers converted to 0
	def random_number_conversion(self, data, column):
		data[data.columns[column]] = data[data.columns[column]].str.strip()
		data[data.columns[column]] = data[data.columns[column]].str.replace(",", ".")
		data[data.columns[column]] = data[data.columns[column]].str.replace(" ", "")

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			e = re.sub("[^0-9]", "", element)
			e = e.strip()
			e = e.lstrip('0') 
			if e == '':
				e = 0
			data[data.columns[column]][i] = e

		return data

	# convert to xx:xx format
	def bed_time_conversion(self, data, column):
		data[data.columns[column]] = data[data.columns[column]].str.strip()

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			e = re.sub("[^0-9]", "", element)
			e = e.strip()
			if e == '':
				e = '00:00'
			data[data.columns[column]][i] = e

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if len(element) == 4:
				e = element[:2] + ':' + element[2:]
				data[data.columns[column]][i] = e

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if len(element) == 1:
				e = "0{}:00".format(element)
				data[data.columns[column]][i] = e

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if len(element) == 2:
				if int(element) > 23:
					e = '00:00'
				else:
					e = "{}:00".format(element)
				data[data.columns[column]][i] = e

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if len(element) == 3:
				x = "{}{}".format(element[0], element[1])
				x = int(x)
				if x > 24:
					e = "0{}".format(element)
				else:
					e = "{}0".format(element)

				e = e[:2] + ':' + e[2:]
				data[data.columns[column]][i] = e

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if len(element) > 5:
				e = "{}{}:{}{}".format(element[0], element[1], element[2], element[3])
				data[data.columns[column]][i] = e

		return data

	# categories: relax, weather, friends, food, travel, training, other
	def good_day_conversion(self, data, column):
		subjects = ['relax', 'weather', 'social', 'food', 'travel', 'training', 'work']
		mapping = {'relax': 0, 'weather': 1,  'social': 2, 'food': 3, 'travel': 4, 'training': 5, 'other': 6, 'work': 7}

		data[data.columns[column]] = data[data.columns[column]].str.lower()
		data[data.columns[column]] = data[data.columns[column]].str.strip()

		data.loc[data[data.columns[column]].str.contains('food'), data.columns[column]] = 'food'
		data.loc[data[data.columns[column]].str.contains('choco'), data.columns[column]] = 'food'
		data.loc[data[data.columns[column]].str.contains('meal'), data.columns[column]] = 'food'
		data.loc[data[data.columns[column]].str.contains('pizza'), data.columns[column]] = 'food'
		data.loc[data[data.columns[column]].str.contains('coffee'), data.columns[column]] = 'food'
		data.loc[data[data.columns[column]].str.contains('eat'), data.columns[column]] = 'food'

		data.loc[data[data.columns[column]].str.contains('weather'), data.columns[column]] = 'weather'
		data.loc[data[data.columns[column]].str.contains('sun'), data.columns[column]] = 'weather'
		data.loc[data[data.columns[column]].str.contains('zon'), data.columns[column]] = 'weather'
		data.loc[data[data.columns[column]].str.contains('warm'), data.columns[column]] = 'weather'

		data.loc[data[data.columns[column]].str.contains('friend'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('person'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('relationship'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('people'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('bitch'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('girl'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('gf'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('sex'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('love'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('company'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('conversation'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('laugh'), data.columns[column]] = 'social'
		data.loc[data[data.columns[column]].str.contains('social'), data.columns[column]] = 'social'

		data.loc[data[data.columns[column]].str.contains('travel'), data.columns[column]] = 'travel'
		data.loc[data[data.columns[column]].str.contains('training'), data.columns[column]] = 'training'
		data.loc[data[data.columns[column]].str.contains('sport'), data.columns[column]] = 'training'
		data.loc[data[data.columns[column]].str.contains('gym'), data.columns[column]] = 'training'
		data.loc[data[data.columns[column]].str.contains('run'), data.columns[column]] = 'training'
		data.loc[data[data.columns[column]].str.contains('football'), data.columns[column]] = 'training'
		data.loc[data[data.columns[column]].str.contains('tennis'), data.columns[column]] = 'training'
		data.loc[data[data.columns[column]].str.contains('hike'), data.columns[column]] = 'training'

		data.loc[data[data.columns[column]].str.contains('work'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('money'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('goal'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('productive'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('task'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('grade'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('thesis'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('useful'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('learn'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('vu'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('success'), data.columns[column]] = 'work'
		data.loc[data[data.columns[column]].str.contains('succeed'), data.columns[column]] = 'work'

		data.loc[data[data.columns[column]].str.contains('relax'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('beer'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('beach'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('no'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('weekend'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('sleep'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('weed'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('drug'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('music'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('chill'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('slep'), data.columns[column]] = 'relax'
		data.loc[data[data.columns[column]].str.contains('swim'), data.columns[column]] = 'relax'

		for i in range(len(data)):
			element = data[data.columns[column]][i]
			if element not in subjects:
				data[data.columns[column]][i] = 'other'

		data[data.columns[column]] = data[data.columns[column]].map(mapping)
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