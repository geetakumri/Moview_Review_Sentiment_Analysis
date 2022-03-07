from turtle import clear
from joblib import load
import pandas as pd

def perform_prediction(df_test):
    new_col = 'Sentiment'
    #print(df_test.head())
    df_test_without_phraseId = df_test.drop('PhraseId',axis=1)
    
    df_test_without_phraseId.reset_index(drop=True, inplace=True)
    print(df_test_without_phraseId.head())

    
    model = load('rf_clf',mmap_mode='r')
    
    cols = ['Phrase','sentiment']
    df_final = pd.DataFrame(columns = cols)
    for index, row in df_test_without_phraseId.iterrows():
        
        df = pd.DataFrame({'Phrase': row})
        sentiment = model.predict(df)
        df_final.loc[len(df_final)] = [row['Phrase'],sentiment]
      
    print('final df : ', df_final.head())
    df_test[new_col] = df_final['sentiment']

    prediction = df_test[['PhraseId', new_col]]
    return prediction




