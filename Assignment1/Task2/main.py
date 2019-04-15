import pandas as pd
from sklearn import linear_model, preprocessing, tree, model_selection
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import datetime

class Task_Two:

	training_path = './train.csv'
	test_path = './test.csv'

	training_data = pd.read_csv(training_path)
	test_data = pd.read_csv(test_path)

	titanic = training_data.append(test_data, ignore_index=True)

	train_idx = len(training_data)
	test_idx = len(titanic) - len(test_data)

	embarked_mapping = {'S':0, 'C':1, 'Q':2}

	#Title
	title_mapping = {"Mr":0, "Miss": 1, "Mrs": 2, "Master":4, "Dr":4, "Rev":4, "Col":4, "Mlle":1, 
	"Major":4, "Don":3, "Mme":2, "Ms":2, "Jonkheer":3, "Sir":3, "Lady":3, "the Countess":3, "Capt":4, "Dona":3}

	titanic['Title'] = titanic.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
	titanic.Title = titanic.Title.map(title_mapping)
	title_variables = pd.get_dummies(titanic.Title, prefix="Title")

	#Age
	# group by Sex, Pclass, and Title 
	grouped = titanic.groupby(['Sex','Pclass', 'Title'])  
	# view the median Age by the grouped features 
	titanic.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
	titanic['AgeClass'] = titanic['Age'].map(lambda x: 0 if x <= 14 else (1 if x>14 and x<25 else (2 if x >=25 and x <45 else (3 if x >=40 and x <65 else 4))))
	age_variables = pd.get_dummies(titanic.AgeClass, prefix="AgeClass")

	#Embarked
	embarked_mapping = {'S':0, 'C':1, 'Q':2}
	embarked_values = titanic.Embarked.value_counts()
	titanic.Embarked = titanic.Embarked.fillna("S")
	titanic.Embarked = titanic.Embarked.map(embarked_mapping)
	embarked_variables = pd.get_dummies(titanic.Embarked, prefix="Embarked")

	#Fare
	grouped_fare = titanic.groupby(['Fare','Pclass'])  
	titanic.Fare = grouped_fare.Fare.apply(lambda x: x.fillna(x.median()))

	titanic.Fare = titanic.Fare.map(lambda x: 0 if x <= 17 else (1 if x>17 and x<=30 else (2 if x >30 and x <=100 else 3)))
	fare_variables = pd.get_dummies(titanic.Fare, prefix="Fare")

	#Pclass
	pclass_variables = pd.get_dummies(titanic.Pclass, prefix="Pclass")

	#FamilySize
	titanic['FamilySize'] = titanic.Parch + titanic.SibSp + 1
	familysize_variables = pd.get_dummies(titanic.FamilySize, prefix="FamilySize")

	#Cabin
	cabin_values = titanic.Cabin.value_counts()
	titanic.Cabin = titanic.Cabin.fillna('U')
	titanic.Cabin = titanic.Cabin.map(lambda x: x[0])
	cabin_variables = pd.get_dummies(titanic.Cabin, prefix="Cabin")

	#Sex
	titanic.Sex = titanic.Sex.map({"male": 0, "female":1})

	titanic_d = pd.concat([titanic, cabin_variables, familysize_variables, fare_variables, pclass_variables, embarked_variables, title_variables], axis=1)

	titanic_d.drop(['Name', 'Ticket', 'Parch', 'SibSp', 'Age', 'Cabin', 'FamilySize', 'AgeClass', 'Pclass', 'Embarked', 'Fare', 'Title' ], axis=1, inplace=True)

	train = titanic_d[ :train_idx]
	test =  titanic_d[test_idx: ]

	X = train.drop('Survived', axis=1)
	y = train.Survived

	X_test = test.drop('Survived', axis=1).values

	print(X.head())

	# def explore(self):
		# print(self.training_data.describe(include='all'))
		# print(self.training_data.head(100))
		# print(self.test_data.describe(include='all'))
		# print(self.test_data.head(100))



	# training_test_data = [training_data, test_data]


	# print(training_data.info())

	# def __init__(self):
	# 	df = self.normalize_data(self.training_data)

	# 	target = df["Survived"].values
	# 	feature_names = ["Pclass", "Age", "Sex", "Fare", "Embarked", "SibSp", "Parch"]
	# 	features = df[feature_names].values

	# 	generalized_tree = tree.DecisionTreeClassifier(
	# 		random_state=1,
	# 		max_depth = 7,
	# 		min_samples_split = 2
	# 	)
		# generalized_tree_ = generalized_tree.fit(features, target)
		# tree.export_graphviz(generalized_tree_, feature_names=feature_names, out_file="tree.dot")
		# print(decision_tree_.score(features, target))

		# scores = model_selection.cross_val_score(generalized_tree, features, target, scoring='accuracy', cv=50)
		# print(scores)
		# print(scores.mean())
	# 	print(self.training_data.shape) 
	# 	print(self.training_data.count())

	# def visualize(self):
	# 	fig = plt.figure(figsize=(18,6))
		
	# 	plt.subplot2grid((2,3), (0,0))
	# 	self.training_data.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Survived")

	# 	plt.subplot2grid((2,3), (0,1))
	# 	plt.scatter(self.training_data.Survived, self.training_data.Age, alpha=0.1)
	# 	plt.title("Age wrt Survived")

	# 	plt.subplot2grid((2,3), (0,2))
	# 	self.training_data.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Class")

	# 	plt.subplot2grid((2,3), (1,0), colspan=2)
	# 	for x in [1,2,3]:
	# 		self.training_data.Age[self.training_data.Pclass == x].plot(kind="kde")
	# 	plt.title("Class wrt Age")
	# 	plt.legend(("1st class", "2nd class", "3rd class"))

	# 	plt.subplot2grid((2,3), (1,2))
	# 	self.training_data.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Embarked")

	# 	plt.show()

	# def visualize_gender(self):
	# 	f_color = "#000000"
	# 	fig = plt.figure(figsize=(18,6))
		
	# 	plt.subplot2grid((3,4), (0,0))
	# 	self.training_data.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Survived")

	# 	plt.subplot2grid((3,4), (0,1))
	# 	self.training_data.Survived[self.training_data.Sex == "male"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Men survived")

	# 	plt.subplot2grid((3,4), (0,2))
	# 	self.training_data.Survived[self.training_data.Sex == "female"].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=f_color)
	# 	plt.title("Women Survived")

	# 	plt.subplot2grid((3,4), (0,3))
	# 	self.training_data.Sex[self.training_data.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=[f_color, 'b'])
	# 	plt.title("Sex of survived")

	# 	plt.subplot2grid((3,4), (1,0), colspan=4)
	# 	for x in [1, 2, 3]:
	# 		self.training_data.Survived[self.training_data.Pclass == x].plot(kind="kde")
	# 	plt.title("Class wrt Survived")
	# 	plt.legend(("1st", "2nd", "3rd"))

	# 	plt.subplot2grid((3,4), (2,0))
	# 	self.training_data.Survived[(self.training_data.Sex == "male") & (self.training_data.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Rich man Survived")

	# 	plt.subplot2grid((3,4), (2,1))
	# 	self.training_data.Survived[(self.training_data.Sex == "male") & (self.training_data.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Poor man survived")

	# 	plt.subplot2grid((3,4), (2,2))
	# 	self.training_data.Survived[(self.training_data.Sex == "female") & (self.training_data.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Rich women Survived")

	# 	plt.subplot2grid((3,4), (2,3))
	# 	self.training_data.Survived[(self.training_data.Sex == "female") & (self.training_data.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
	# 	plt.title("Poor women survived")

	# 	plt.show()

	# def hypothesis(self):
	# 	train = self.training_data.copy()
	# 	train["Hyp"] = 0
	# 	train.loc[train.Sex == "female", "Hyp"] = 1

	# 	train["Result"] = 0
	# 	train.loc[train.Survived == train["Hyp"], "Result"] = 1

	# 	print(train["Result"].value_counts(normalize=True))

	# def normalize_data(self, d):
	# 	data = d.copy()

	# 	data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
	# 	data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

	# 	data.loc[data["Sex"] == "male", "Sex"] = 0
	# 	data.loc[data["Sex"] == "female", "Sex"] = 1

	# 	data["Embarked"] = data["Embarked"].fillna("S")
	# 	data.loc[data["Embarked"] == "S", "Embarked"] = 0
	# 	data.loc[data["Embarked"] == "C", "Embarked"] = 1
	# 	data.loc[data["Embarked"] == "Q", "Embarked"] = 2

	# 	return data

	# def linear_process(self):
	# 	df = self.normalize_data(self.training_data)
	# 	target = df["Survived"].values
	# 	feature_names = ["Pclass", "Age", "Sex", "Fare", "Embarked", "SibSp", "Parch"]
	# 	features = df[feature_names].values

	# 	classifier = linear_model.LogisticRegression()
	# 	classifier_ = classifier.fit(features, target)

	# 	print(classifier_.score(features, target))

	# 	poly = preprocessing.PolynomialFeatures(degree=2)
	# 	poly_features = poly.fit_transform(features)

	# 	classifier_ = classifier.fit(poly_features, target)
	# 	print(classifier_.score(poly_features, target))

if __name__ == '__main__':
	a = Task_Two()
	# a.explore()
	# a.visualize()
	# a.visualize_gender()
	# a.hypothesis()