'''
@Author Surajit Kundu
@Email surajit.113125@gmail.com
'''
## import required libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

def predict(x_train, y_train , x_test, k, metric='euclidean'):
    predictions = []
     
    #Loop through the Datapoints to be classified
    counter = 0
    for item in x_test:  
        #Array to store distances from the test_input
        point_dist = np.array(pairwise_distances(X=x_train, Y=item, metric=metric))
        #Keeping the first K datapoints
        dist = np.argsort(point_dist[np.where(point_dist)])[:k] 
        #Labels of the K datapoints from above
        labels = y_train[dist]
        #Majority classes
        lab = mode(labels) 
        lab = lab.mode[0]
        predictions.append(lab)
        counter = counter+1
        if counter%1000 == 0:
            print(counter, metric)
    return predictions


## Read the dataset of sentiment analysis on movie reviews
data = pd.read_csv("train.tsv", sep='\t')
print(data.head(5))
## Select the phrase and sentiment from the dataset
train_set = pd.DataFrame(data,columns = ['Phrase', 'Sentiment'])
print("train_set shape: ",train_set.shape)
## Conveting all to lower case
train_set['Phrase'] = train_set['Phrase'].str.lower()
## Downloading all stopwords
nltk.download('stopwords')
eng_stops =set(stopwords.words("english"))
## Remove Stop words
train_set["Phrase"] = train_set["Phrase"].apply(lambda func: ' '.join(sw 
                                        for sw in func.split() 
                                        if sw not in eng_stops))
# Removing numbers from all the tweets
train_set['Phrase'] = train_set['Phrase'].str.replace('\d+', '')
## Removing the rows where phrase is empty
train_set['Phrase'].replace('', np.nan, inplace=True)
train_set.dropna(subset = ["Phrase"], inplace=True)
print("train_set shape after preprocess: ", train_set.shape)
## Train test split 80/20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train_set, test_size = 0.2, random_state = 42)
#vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer()
train_tfidf_model = vectorizer.fit_transform(X_train["Phrase"])
test_tfidf_model = vectorizer.transform(X_test["Phrase"]) 

'''
A1 >>
'''
y_pred = predict(train_tfidf_model, np.array(X_train["Sentiment"]), test_tfidf_model, 5, metric='euclidean')
print(accuracy_score(np.array(X_test["Sentiment"]), y_pred))

'''
A2>> Vary the value of k (depending on the number of classes) with three different similarity/distance measures such as a) cosine similarity, b) Euclidean
distance, and c) Manhattan distance and evaluate the performance of your
classifier on each of them independently. Compare their performances and
analyse the results.
'''
all_accuracy_e, all_accuracy_c, all_accuracy_m = [],[],[]
max_k = 200
k_range = []
for i in range(2, max_k, 4):
    print(i)
    k_range.append(i)
    ## predictions using Euclidean distance
    y_pred_e = predict(train_tfidf_model, np.array(X_train["Sentiment"]), test_tfidf_model, i, metric='euclidean') 
    ## predictions using Cosine distance
    y_pred_c = predict(train_tfidf_model, np.array(X_train["Sentiment"]), test_tfidf_model, i, metric='cosine') 
    ## predictions using Manhattan distance
    y_pred_m = predict(train_tfidf_model, np.array(X_train["Sentiment"]), test_tfidf_model, i, metric='manhattan') 
    #Keeping the accuracy
    all_accuracy_e.append(accuracy_score(np.array(X_test["Sentiment"]), y_pred_e))
    all_accuracy_c.append(accuracy_score(np.array(X_test["Sentiment"]), y_pred_c))
    all_accuracy_m.append(accuracy_score(np.array(X_test["Sentiment"]), y_pred_m))
print("Complete!")
print("Euclidean Distance matrix", all_accuracy_e)
print("Cosine Distance matrix", all_accuracy_c)
print("Manhattan Distance matrix", all_accuracy_m)

plt.plot(k_range, all_accuracy_e, color = 'firebrick', label="Euclidean",drawstyle="steps-post", marker='o')
plt.plot(k_range, all_accuracy_c, color = 'darkorange', label="Cosine",drawstyle="steps-post", marker='o')
plt.plot(k_range, all_accuracy_m, color = 'darkgreen', label="Manhattan",drawstyle="steps-post", marker='o')
plt.title('K vs Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('K')
plt.legend(['Euclidean', 'Cosine', 'Manhattan'], loc='upper left')
plt.savefig('Accuracy_K'+str(max_k)+'.png')
plt.show()
