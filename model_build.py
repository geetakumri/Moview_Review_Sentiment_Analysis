from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib


def model_building(data):
    def model_prediction(tfidf,name, model):
        
    # Training the classifier with Naive Bayes
        cassifier = Pipeline([tfidf,
               (name,model),
              ])

        cassifier.fit(X_train, Y_train)
        test_predict = cassifier.predict(X_test)
       #print("test_predict", set(test_predict))

        train_accuracy = round(cassifier.score(X_train, Y_train)*100)
        test_accuracy = round(accuracy_score(test_predict, Y_test)*100)

        print(f" {name} Train Accuracy Score : {train_accuracy}% ")
        print(f" {name} Test Accuracy Score  : {test_accuracy}% ")
        print()
        joblib.dump(cassifier,  open(name, "wb"))


    X_train, X_test, y_train, y_test = train_test_split(data.index.values, data.Sentiment.values, test_size=0.1, random_state=42, stratify=data.Sentiment)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=.15, random_state=42, stratify=y_train)
    
    data['data_type'] = ['not_set']*data.shape[0]
    data.loc[X_train, 'data_type'] = 'train'
    data.loc[X_val, 'data_type'] = 'val'
    data.loc[X_test,'data_type'] = 'test'

    data = data.dropna()
    train_set = data[data['data_type'] == 'train'].drop_duplicates(ignore_index=True)
    val_set = data[data['data_type'] == 'val'].drop_duplicates(ignore_index=True)
    test_set = data[data['data_type'] == 'test'].drop_duplicates(ignore_index=True)

    data = pd.concat([train_set, val_set, test_set], ignore_index=True)
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)

    X_train = train_set.Phrase.values
    Y_train = train_set.Sentiment.values
    X_test = test_set.Phrase.values
    Y_test = test_set.Sentiment.values

    #vect = CountVectorizer(stop_words='english', ngram_range=(1,1), )

    models = []
    models.append(('nb_clf', MultinomialNB()))
    models.append(('rf_clf',  DecisionTreeClassifier()))
    models.append(('sgd_clf', SGDClassifier()))

    for name, model in models:
        model_prediction(('tfidf', TfidfVectorizer()),name, model)
   




 


