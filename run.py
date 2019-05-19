from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter
from stop_words import get_stop_words

stop = set(get_stop_words('russian'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
 
# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

print("There are 10 sentences of following three classes on which K-NN classification and K-means clustering"\
         " is performed : \n1. Математика \n2. Политика \n3. Реклама")
path = "Sentences.txt"
 
train_clean_sentences = []
fp = open(path,'r')
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)
 
vectorizer = TfidfVectorizer(stop_words=get_stop_words('russian'))
X = vectorizer.fit_transform(train_clean_sentences)
 
# Creating true labels for 30 training sentences
y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2

# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X,y_train)

test_sentences = ["изменение мест слагаемых суммы не меняет",\
"магазин одежды",\
"президент подал в отставку"]
 
test_clean_sentence = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+","",cleaned)
    test_clean_sentence.append(cleaned)
 
Test = vectorizer.transform(test_clean_sentence)
 
true_test_labels = ['Математика','Политика','Реклама']
predicted_labels_knn = modelknn.predict(Test)

 
print("\nBelow 3 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ",\
test_sentences[0],"\n2. ",test_sentences[1],"\n3. ",test_sentences[2])
print("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")
print("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_knn[0])],\
"\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_knn[1])],\
"\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_knn[2])])
 