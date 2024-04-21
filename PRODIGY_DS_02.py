import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv('C:\\Users\\kavit\\Downloads\\train.csv')
print(df.head())

print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print(df.describe())

sns.histplot(df['Age'])
plt.title('Age')
plt.show()

sns.countplot(x='Survived',data=df)
plt.title("Survival Count")
plt.show()

sns.barplot(x='Sex',y='Survived',data=df)
plt.title('Survival rate by Sex')
plt.show()

sns.barplot(x='Pclass',y='Survived', data=df)
plt.title('Survival rate by Class')
plt.show()

sns.scatterplot(x='Age',y='Fare',hue='Survived',data=df)
plt.title('Age vs Fare by Survival rate')
plt.show()


