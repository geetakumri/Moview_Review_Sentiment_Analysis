import pandas as pd
import numpy as np
import os

from datapreprocessing import perform_data_preprocessing,perform_data_preprocessing_on_test_data
from predict import perform_prediction
from model_build import model_building


def load_data(path1,path2):
    print('load data')
    df_train = pd.read_csv(path1, sep='\t')
    df_test = pd.read_csv(path2, sep='\t')

    return df_train, df_test

df_train, df_test = load_data(os.getcwd()+"\\Moview_Review_Sentiment_Analysis\\train.tsv", os.getcwd()+"\\Moview_Review_Sentiment_Analysis\\test.tsv")
df_train = perform_data_preprocessing(df_train)
#print(df_train.head())

model_building(df_train)

preprocessed_test_df = perform_data_preprocessing_on_test_data(df_test)
prediction = perform_prediction(preprocessed_test_df)
prediction.to_csv("prediction.csv", index = False)

