import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

PATH = 'C:\\Users\\91837\\PycharmProjects\\Ml_lab\\titanic_preprocessing\\DATA_SETS\\'

train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'test.csv')
gender_submission_data = pd.read_csv(PATH + 'gender_submission.csv')

# Visualization of the 'Survived' column values in a bar graph
print('Count of "Survived" column values:\n0 - Not Survived\n1 - Survived\n', train_data['Survived'].value_counts(), sep='\n')
colors = ['skyblue', 'salmon']
bar_plot_survived = train_data['Survived'].value_counts().plot(kind='bar', color=colors)
bar_plot_survived.set_xlabel('Survived or not')
bar_plot_survived.set_ylabel('Passenger Count')
bar_plot_survived.set_title('Survival Distribution')
plt.show()  # Use plt.show() to display the plot

# Pclass
bar_plot_pclass = train_data['Pclass'].value_counts().sort_index().plot(kind='bar', title='')
bar_plot_pclass.set_xlabel('Pclass')
bar_plot_pclass.set_ylabel('Passenger Count')
plt.show()

print(train_data[['Pclass', 'Survived']].groupby('Pclass').count())
print(train_data[['Pclass', 'Survived']].groupby('Pclass').sum())

bar_plot_pclass_survived = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot(kind='bar')
bar_plot_pclass_survived.set_xlabel('Pclass')
bar_plot_pclass_survived.set_ylabel('Survival Probability')
plt.show()

# Sex
bar_plot_sex = train_data['Sex'].value_counts().sort_index().plot(kind='bar')
bar_plot_sex.set_xlabel('Sex')
bar_plot_sex.set_ylabel('Passenger Count')
plt.show()

bar_plot_sex_survived = train_data[['Sex', 'Survived']].groupby('Sex').mean().Survived.plot(kind='bar')
bar_plot_sex_survived.set_xlabel('Sex')
bar_plot_sex_survived.set_ylabel('Survival Probability')
plt.show()

# Embarked
bar_plot_embarked = train_data['Embarked'].value_counts().sort_index().plot(kind='bar')
bar_plot_embarked.set_xlabel('Embarked')
bar_plot_embarked.set_ylabel('Passenger Count')
plt.show()

plt = train_data[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot(kind='bar')
print(plt.set_xlabel('Embarked'))
print(plt.set_ylabel('Survival Probability'))

plt = train_data.SibSp.value_counts().sort_index().plot(kind='bar')
print(plt.set_xlabel('SibSp'))
print(plt.set_ylabel('Passenger count'))

plt = train_data[['SibSp', 'Survived']].groupby('SibSp').mean().Survived.plot(kind='bar')
print(plt.set_xlabel('SibSp'))
print(plt.set_ylabel('Survival Probability'))

plt = train_data.Parch.value_counts().sort_index().plot(kind='bar')
print(plt.set_xlabel('Parch'))
print(plt.set_ylabel('Passenger count'))

plt = train_data[['Parch', 'Survived']].groupby('Parch').mean().Survived.plot(kind='bar')
print(plt.set_xlabel('Parch'))
print(plt.set_ylabel('Survival Probability'))

print(sns.catplot(x='Pclass', col='Embarked', data=train_data, kind='count'))

print(sns.catplot(x='Sex', col = 'Pclass', data = train_data, kind = 'count'))

print(sns.catplot(x='Sex', col = 'Embarked', data = train_data, kind = 'count'))

print(train_data.head())

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
print(train_data.head())

train_data = train_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])
print(train_data.head())

train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'C':0, 'Q':1, 'S':2})

print(train_data.head())

train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data = train_data.drop(columns='Name')

print(train_data.Title.value_counts().plot(kind='bar'))

train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')

plt = train_data.Title.value_counts().sort_index().plot(kind='bar')
print(plt.set_xlabel('Title'))
print(plt.set_ylabel('Passenger count'))

plt = train_data[['Title', 'Survived']].groupby('Title').mean().Survived.plot(kind='bar')
print(plt.set_xlabel('Title'))
print(plt.set_ylabel('Survival Probability'))

train_data['Title'] = train_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
print(train_data.head())

corr_matrix = train_data.corr()

import matplotlib.pyplot as plt
print(plt.figure(figsize=(9, 8)))
print(sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2))

print(train_data.isnull().sum())
print(train_data['Embarked'].isnull().sum())

train_data['Embarked'] = train_data['Embarked'].fillna(2)
print(train_data.head())

corr_matrix = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].corr()

print(plt.figure(figsize=(7, 6)))
print(sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2))

NaN_indexes = train_data['Age'][train_data['Age'].isnull()].index

for i in NaN_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == train_data.iloc[i]["SibSp"]) & (train_data.Parch == train_data.iloc[i]["Parch"]) & (train_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        train_data['Age'].iloc[i] = pred_age
    else:
        train_data['Age'].iloc[i] = train_data['Age'].median()

print(train_data.isnull().sum())

print(train_data.head())

test_data = pd.read_csv(PATH + 'test.csv')
print(test_data.isnull().sum())

test_data = test_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])
print(test_data.head())

test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})
test_data['Embarked'] = test_data['Embarked'].map({'C':0, 'Q':1, 'S':2})

print(test_data.head())

test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data = test_data.drop(columns='Name')

test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

test_data['Title'] = test_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})

print(test_data.head())
print(test_data.isnull().sum())

NaN_indexes = test_data['Age'][test_data['Age'].isnull()].index

for i in NaN_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == test_data.iloc[i]["SibSp"]) & (train_data.Parch == test_data.iloc[i]["Parch"]) & (test_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        test_data['Age'].iloc[i] = pred_age
    else:
        test_data['Age'].iloc[i] = train_data['Age'].median()

title_mode = train_data.Title.mode()[0]
test_data.Title = test_data.Title.fillna(title_mode)
fare_mean = train_data.Fare.mean()
test_data.Fare = test_data.Fare.fillna(fare_mean)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
print(test_data.head())

#Split 'train data' into 'training data' and 'validation data'
print(train_data.head())

from sklearn.utils import shuffle
train_data = shuffle(train_data)

X_train = train_data.drop(columns='Survived')
y_train = train_data.Survived
y_train = pd.DataFrame({'Survived':y_train.values})

X_test = test_data
print(X_train.head())

print(y_train.head())

print(X_train.shape)
print(X_test.head())

X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)