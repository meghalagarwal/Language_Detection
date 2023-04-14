'''Import Libraries'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalysis():
    def understanding_data(self, dataframe: pd.DataFrame) -> None:
        print(f'\nNames of features are:\n{dataframe.columns}')
        print(f'\nTop 5 records of Data:\n{dataframe.head(10)}')
        print(f'\nNull values:\n{dataframe.isna().sum()}')
        print(f'\nType of data types of each features:\n{dataframe.dtypes}')
        print(f'\nMeasure of Central Tendency and Dispersion:\n{dataframe.describe()}')
        print(f'\nUnique value count of Languages:\n{len(dataframe.language.unique())}')
        print(f'\nValue count of Languages:\n{dataframe.language.value_counts()}')

    def language_distribution(self, dataframe: pd.DataFrame) -> None:
        sns.countplot(data=dataframe, x='language', order=dataframe['language'].value_counts().index)
        plt.show()