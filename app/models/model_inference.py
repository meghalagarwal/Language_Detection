'''Import libraries'''
import pandas as pd
import random
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
import pprint as pp
import pickle
from sklearn.preprocessing import LabelEncoder

class Models():
    def __init__(self) -> None:        
        self.label = LabelEncoder()

    def sampling_data(self, dataframe: pd.DataFrame, lang: str) -> pd.DataFrame:

        return dataframe[dataframe['language']==lang].sample(frac=0.75)
    
    def training_testing_data(self, dataframe: pd.DataFrame):
        dataframe['language'] = self.label.fit_transform(dataframe['language'])
        training_df = pd.DataFrame()
        testing_df = pd.DataFrame()
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        
        for lang in dataframe.language.unique():
            temp_df = self.sampling_data(dataframe=dataframe, lang=lang)
            training_df = pd.concat([training_df, temp_df], axis=0, ignore_index=True)
        
        training_df = training_df.sample(frac=1).reset_index(drop=True)
        testing_df = dataframe[~dataframe.Text.isin(training_df.Text)]
        
        return training_df, testing_df
    
    def training_model(self, training_df: pd.DataFrame, testig_df: pd.DataFrame):
        list_of_models = [
            (
            'LogisticRegression',
            LogisticRegression(),
            dict(
            penalty = ['l1', 'l2', 'elasticnet', None],
            dual = [True, False],
            C = [_ for _ in range(1, 101, 10)],
            random_state = [_ for _ in range(1, 55, 6)],
            solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            max_iter = [_ for _ in range(10, 101, 20)],
            multi_class = ['auto', 'ovr', 'multinomial'])),
            (
            'KNNClassifier',
            KNeighborsClassifier(),
            dict(
            n_neighbors = [2, 3, 5],
            weights = ['uniform', 'distance'],
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'],
            p = [1, 2])),
            (
            'SVM',
            SVC(),
            dict(
            C = [_ for _ in range(1, 101, 10)],
            kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])),
            (
            'RandomForest',
            RandomForestClassifier(),
            dict(
            n_estimators = [_ for _ in range(100, 500, 100)],
            max_depth = [3, None],
            max_features = ['sqrt', 'log2', None],
            min_samples_split = sp_randint(2, 11),
            min_samples_leaf = sp_randint(1, 11),
            bootstrap = [True, False],
            criterion = ["gini", "entropy", "log_loss"],
            class_weight = ['balanced', 'balanced_subsample']))
        ]

        final_result = {}
        for grp in list_of_models:
            print('\n\n', grp[0], '\n\n')
            grid_search = GridSearchCV(estimator=grp[1], param_grid=grp[2], n_jobs=20, cv=5, verbose=2)
            grid_search.fit(training_df['Text'], training_df['language'])
            final_result[grp[0]] = grid_search.score(testig_df['Text'], testig_df['language'])
            testig_df[f'{grp[0]}_pred'] = self.label.inverse_transform(grid_search.predict(testig_df['Text']))
            pp.pprint('\n\n********', grid_search.best_params_, '********\n\n')
            # save the model to disk
            filename = f'{grp[0]}_model.pkl'
            pickle.dump(grid_search, open(f'C:\\Dataset\\models\\{filename}', 'wb'))

        testig_df['language'] = self.label.inverse_transform(testig_df['language'])
        pp.pprint(final_result)