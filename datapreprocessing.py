import re
import spacy
sp = spacy.load('en_core_web_sm')
stopwords = sp.Defaults.stop_words
insignificant_cols = ['PhraseId','SentenceId']
insignificant_cols_test = ['SentenceId']


def drop_insignificant_cols(df, insignificant_cols):
    for col in insignificant_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

def remove_stopwords(text):
    text_tokens = text.split(' ')
    text_tokens_filtered = [word for word in text_tokens if not word in stopwords]
    return (" ").join(text_tokens_filtered)

def clean(text):
    text = re.sub(r'@|#', r'', text.lower())
    text = re.sub(r'http.*', r'', text.lower())
    return ' '.join(re.findall(r'\w+', text.lower()))

def perform_data_preprocessing(df_train):
    drop_insignificant_cols(df_train, insignificant_cols)
    df_train['Phrase'] = df_train['Phrase'].apply(lambda x:remove_stopwords(x))
    df_train['Phrase'] = df_train['Phrase'].apply(lambda x: clean(x))
    #print(df_train.head())
    return df_train
    
def perform_data_preprocessing_on_test_data(df_test):
    drop_insignificant_cols(df_test, insignificant_cols_test)
    df_test['Phrase'] = df_test['Phrase'].apply(lambda x:remove_stopwords(x))
    df_test['Phrase'] = df_test['Phrase'].apply(lambda x:clean(x))
    return df_test







