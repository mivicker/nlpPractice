from nltk.corpus import names
from sklearn.model_selection import train_test_split

def extract_features(name):
	"""
	Get the features used for name classification
	"""
	return {
		'last_letter': name[-1]
	}

#Get the names
boy_names = names.words('male.txt')
girl_names = names.words('female.txt')

#Build the dataset
boy_names_dataset = [(extract_features(name), 'boy') for name in boy_names]
girl_names_dataset = [(extract_features(name), 'girl') for name in girl_names]

data = boy_names_dataset + girl_names_dataset

X, y = list(zip(*data))

#split and randomize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, shuffle=True)
print(X_train[:10])
print(y_train[:10])

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

dict_vectorizer = DictVectorizer()
name_classifier = DecisionTreeClassifier()

dict_vectorizer.fit(X_train)

X_train_vectorized = dict_vectorizer.transform(X_train)

name_classifier.fit(X_train_vectorized, y_train)

X_test_vectorized = dict_vectorizer.transform(X_test)