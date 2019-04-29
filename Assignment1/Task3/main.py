import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
import warnings
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load("en")
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import fbeta_score

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
# nltk.download('stopwords')
string.punctuation
stopwords.words("english")[100:110]

messages = pd.read_csv('./spam.csv', encoding='latin-1')

messages["messageLength"] = messages["v2"].apply(len)
# print(messages["messageLength"].describe())

messages['spam'] = messages['v1'].map( {'spam': 1, 'ham': 0} ).astype(int)
# print(messages.head(15))

def print_validation_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : "+ str(acc_sc)) 
    return acc_sc

def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    #fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", cbar=False, ax=ax)
    #  square=True,
    plt.ylabel('true label')
    plt.xlabel('predicted label')

def remove_punctuation_and_stopwords(sms):
    sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split() 
    sms_no_punctuation_no_stopwords = \
        [word for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]    
    return sms_no_punctuation_no_stopwords

messages = messages.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages = messages.rename(columns={"v1":"label", "v2":"text"})

# print(messages.head())

#new
vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

def cleanText(message):
    
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]
    
    return " ".join(words)

messages["text"] = messages["text"].apply(cleanText)

features = vec.fit_transform(messages["text"])

def encodeCategory(cat):
    if cat == "spam":
        return 1
    else:
        return 0


        
messages["label"] = messages["label"].apply(encodeCategory)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, messages["label"], stratify = messages["label"], test_size = 0.3, random_state=21)

gaussianNb = MultinomialNB()
gaussianNb.fit(X_train, y_train)

y_pred = gaussianNb.predict(X_test)

# print(fbeta_score(y_test, y_pred, beta = 0.5))

print(classification_report(y_test, y_pred))

count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        count += 1
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)
print(X_train)
# print(messages.head())






# messages['text']= messages['text'].apply(remove_punctuation_and_stopwords)

# bow_transformer = CountVectorizer(analyzer = remove_punctuation_and_stopwords).fit(messages['text'])

# bow_data = bow_transformer.transform(messages['text'])

# tfidf_transformer = TfidfTransformer().fit(bow_data)

# data_tfidf = tfidf_transformer.transform(bow_data)

# spam_detect_model = MultinomialNB().fit(data_tfidf, messages["spam"])

# all_pred = spam_detect_model.predict(data_tfidf)


# sms_train, sms_test, label_train, label_test = train_test_split(messages["text"], messages["spam"], test_size=0.33, random_state=21)

# pipe_MNB = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
#                    ('tfidf'   , TfidfTransformer()),
#                    ('clf_MNB' , MultinomialNB()),
#                     ])

# pipe_MNB.fit(X=sms_train, y=label_train)
# pred_test_MNB = pipe_MNB.predict(sms_test)
# acc_MNB = accuracy_score(label_test, pred_test_MNB)

# count = 0
# for i in range(len(label_test)):
#     if label_test.iloc[i] != pred_test_MNB[i]:
#         count += 1
# print('Total number of test cases', len(label_test))
# print('Number of wrong of predictions', count)
# # print(acc_MNB)
# # print(pipe_MNB.score(sms_test, label_test))
# print(classification_report(label_test, pred_test_MNB))

# # print(sms_test[label_test != pred_test_MNB])
# # plot_confusion_matrix(label_test, pred_test_MNB)
