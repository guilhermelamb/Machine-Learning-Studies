import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

df = pd.read_csv(r'C:\Users\Guilherme\Machine-Learning-Studies\Databases\Tweets_Mg.csv', encoding='utf-8')
print(df.count())

#Count the classes
print(df.groupby('Classificacao')['Classificacao'].count())


#Splitting the dataframe into predictor variables and class
tweets = df['Text'].values
classes = df['Classificacao'].values


#Creating the model
vectorizer = CountVectorizer(analyzer='word')
tweet_freq = vectorizer.fit_transform(tweets)

model_bayes = MultinomialNB()
model_bayes.fit(tweet_freq, classes)

#validating the model
results = cross_val_predict(model_bayes, tweet_freq, classes, cv=10)

accuracy = metrics.accuracy_score(classes, results)
matrix = metrics.confusion_matrix(classes, results)
sentiment = ['Negativo', 'Neutro', 'Positivo']

report = metrics.classification_report(classes, results, sentiment)

#improving the model
#Now we are going to use the “bigrams” model. This modeling consists of passing
#two words as features to the classifier instead of just one
bigrams_vectorizer = CountVectorizer(ngram_range=(1,2))
bigrams_freq_tweet = bigrams_vectorizer.fit_transform(tweets)
bigrams_model = MultinomialNB()
bigram_result = cross_val_predict(bigrams_model, bigrams_freq_tweet, classes, cv=10)

accuracy_bigrams = metrics.accuracy_score(classes, bigram_result)
matrix_bigrams = metrics.confusion_matrix(classes, bigram_result)
sentiment_bigrams = ['Negativo', 'Neutro', 'Positivo']

report_bigrams = metrics.classification_report(classes, bigram_result, sentiment)
