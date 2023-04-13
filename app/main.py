'''Import Libraries'''
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from preprocessing.preprocessing_inference import PreProcessing
from data_analysis.data_analysis import DataAnalysis
from models.model_inference import Models
from tqdm import tqdm
tqdm.pandas()
import json

class MainApplication():
    def __init__(self) -> None:
        self.preprocess = PreProcessing()
        self.data_analysis = DataAnalysis()
        self.model = Models()

    def loading_data(self, download_url: str, output_directory: str) -> None:
        '''This function downloads data from the URL and stores in local drive'''
        self.preprocess.download_data(
            url = download_url,
            out_dir = output_directory)
        
    def data_cleaning(self, data_location: str) -> None:
        '''This function cleans the downloaded data, deletes it and converts it into csv files'''
        Parallel(n_jobs=len(os.listdir(data_location)), backend='threading')(
            delayed(self.preprocess.dataframe_creation)(
            folders,
            files,
            'C:\Dataset\\')
            for folders in  tqdm(os.listdir(data_location), desc='Mutliple CSV Folder Loop', position=1)
            for files in tqdm(os.listdir(f'{data_location}{folders}\\'), desc='Mutliple CSV Files Loop', position=0))
        
    def data_merging(self, data_location: str) -> pd.DataFrame:
        '''This function merges all the csv files into single csv'''
        
        '''Since the size of the files are too large and due to hugh amount of time taking operation, we have reduced to 1000 files from each language folder'''
        lan_detect_df = pd.DataFrame(columns=['Text', 'language'])

        for folders in tqdm(os.listdir(data_location),
                            desc='Folder Loop for Merging',
                            position=1):
            for files in tqdm(os.listdir(f'{data_location}{folders}\\')[:1000],
                              desc='Files Loop for Merging',
                              position=0):
                try:
                    pd.read_csv(f'{data_location}{folders}\\{files}')

                    lan_detect_df = pd.concat([lan_detect_df, pd.read_csv(f'{data_location}{folders}\\{files}')],
                                              axis=0,
                                              ignore_index=True)
                except:
                    continue

        lan_detect_df['language'] = lan_detect_df.language.map(json.load(open('..\\data\\language_abbrivation.json')))
        lan_detect_df = pd.concat([lan_detect_df, pd.read_csv('..\\data\\other_language_dataset.csv')],
                                  axis=0,
                                  ignore_index=True)        
        lan_detect_df.dropna(inplace=True)
        
        lan_detect_df.to_csv('C:\Dataset\\europarl.csv', index=False)

        return lan_detect_df
    
    def ed_analysis(self, dataframe: pd.DataFrame) -> None:
        '''This function performs exploratory data analysis on the Dataframe'''
        self.data_analysis.understanding_data(df)
        self.data_analysis.language_distribution(df)

    def spliting_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.model.training_testing_data(dataframe=dataframe)

if __name__ == "__main__":
    start_app = MainApplication()

    if not os.path.exists('C:\\Dataset\\europarl.csv'):
        start_app.loading_data(download_url='http://www.statmt.org/europarl/v7/europarl.tgz', output_directory='C:\Dataset\\')
        start_app.data_cleaning(data_location='C:\\Dataset\\txt\\')
        df = start_app.data_merging(data_location='C:\\Dataset\\csv\\')
    else:
        df = pd.read_csv('C:\Dataset\\europarl.csv')

    # start_app.ed_analysis(df)

    start_app.spliting_data(df)
