'''Import libraries'''
import pandas as pd
from sklearn.model_selection import train_test_split

class Models():
    def __init__(self) -> None:
        pass

    def training_testing_data(self, dataframe: pd.DataFrame):
        X_train, y_train, X_test, y_test = train_test_split(x=dataframe['Text'], y=dataframe['language'], test_size=0.25)

        print(y_train.language.value_counts())
        print(y_test.language.value_counts())