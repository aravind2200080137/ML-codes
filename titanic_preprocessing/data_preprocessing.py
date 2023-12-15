import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

PATH= 'C:\\Users\\91837\\PycharmProjects\\Ml_lab\\titanic_preprocessing\\DATA_SETS\\'


train_data = pd.read_csv(PATH+'train.csv')
test_data = pd.read_csv(PATH+'test.csv')
gender_submission_data = pd.read_csv(PATH+'gender_submission.csv')

#Train data analysis ==> finding Head,description,column names and data types
print("Train Data Head : ",train_data.head(),sep='\n')
print("Train data description : ",train_data.describe(),sep='\n')
print("Train data Columns : ",train_data.columns,sep='\n')
print("Train data Date types : ",train_data.dtypes,sep='\n')

#Test data analysis ==> finding Head,description,column names and data types
print("Test Data Head : ",test_data.head(),sep='\n')
print("Test data description : ",test_data.describe(),sep='\n')
print("Test Data Columns : ",test_data.columns,sep='\n')
print("Test Data Data types : ",test_data.dtypes,sep='\n')

#finding missing/Null values in Train data
print("Train data Null values Count(column wise) :")
train_column_names = train_data.columns
for t_column in train_column_names:
    print(t_column+' - '+str(train_data[t_column].isnull().sum()))

#finding missng/Null values in test data
print("Test data Null values Count(column wise) :")
test_column_names = test_data.columns
for column in test_column_names:
    print(column+' - '+str(test_data[column].isnull().sum()))

#




