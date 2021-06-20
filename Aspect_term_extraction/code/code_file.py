# importing libraries

import pandas as pd
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from scipy.sparse import hstack
import warnings

from preprocessing.xml_2_dataframe import Xml2DataFrame
from preprocessing.pos_tagger import POSTagger


# path of training dataset

path_train = r'Laptops_Train_v2.xml'
path_test = r'C:Laptops_Test_Gold.xml'
new_test_path = r'test.xml'


# xml parser
def get_xml_data(path):
    xml2df = Xml2DataFrame()
    xml_dataframe = xml2df.process_data(path)
    return xml_dataframe


# Making list to train
train_dataframe = get_xml_data(path_train)
# print(train_dataframe.head())
train_text_list = train_dataframe['text']
train_aspects_list = list(train_dataframe['aspect_info'])
print(train_text_list.head())
