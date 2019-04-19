import pandas as pd
from sklearn import linear_model, preprocessing, tree, model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import seaborn as sns
sns.set()
import datetime

class Task_Two:

	training_path = './train.csv'
	test_path = './test.csv'

	training_data = pd.read_csv(training_path)
	test_data = pd.read_csv(test_path)

	titanic = training_data.append(test_data, ignore_index=True)

	# print(titanic.info())

	passengerId = test_data.PassengerId

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

	#Sex
	titanic.Sex = titanic.Sex.map({"male": 0, "female":1})

	titanic_d = pd.concat([titanic, familysize_variables, fare_variables, pclass_variables, embarked_variables, title_variables], axis=1)

	titanic_d.drop(['Name','PassengerId', 'Ticket', 'Parch', 'SibSp', 'Age', 'Cabin', 'FamilySize', 'AgeClass', 'Pclass', 'Embarked', 'Fare', 'Title' ], axis=1, inplace=True)

	train = titanic_d[ :train_idx]
	test =  titanic_d[test_idx: ]

	train.Survived = train.Survived.astype(int)

	X = train.drop('Survived', axis=1).values 
	y = train.Survived.values

	# print(train.head())

	X_test = test.drop('Survived', axis=1).values

	# create param grid object 
	forrest_params = dict(     
		max_depth = [n for n in range(5, 14)],    
		min_samples_split = [n for n in range(7, 13)],
		min_samples_leaf = [n for n in range(3, 5)],  
		criterion = ['entropy', 'gini'], 
		n_estimators = [n for n in range(10, 60, 5)],
		# max_depth = [9],     
		# min_samples_split = [5], 
		# min_samples_leaf = [4],     
		# n_estimators = [10],
	)
	#Modelaka
	model = tree.DecisionTreeClassifier(random_state=1)
	model = model.fit(X, y)
	
	decision_tree_pred = model.predict(X_test)

	dotfile = open("./img/dtree2.dot", 'w')
	tree.export_graphviz(model, out_file = dotfile, feature_names = train.drop('Survived', axis=1).columns)
	dotfile.close() 


		
	kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': decision_tree_pred})
	# kaggle = round(kaggle)
	kaggle.Survived = kaggle.Survived.astype(int)
	kaggle.to_csv('./pred_procedure_two.csv', index=False)

if __name__ == '__main__':
	a = Task_Two()
	# a.explore()
	# a.visualize()
	# a.visualize_gender()
	# a.hypothesis()