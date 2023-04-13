'''IMport Libraries'''
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import wget
import tarfile
import os
import re
from csv import QUOTE_NONE

class PreProcessing():
    def download_data(self, url: str, out_dir: str) -> None:
        '''This function downloads the .gz file from the url and extracts it on the output drive'''
        if not os.path.exists(f'{out_dir}europarl.tgz'):
            wget.download(url, out=out_dir)
        
        tarfile.open(f'{out_dir}europarl.tgz').extractall(out_dir)
        
        os.remove(f'{out_dir}europarl.tgz')

    def text_to_dataframe(self, folders_name: str, file_name: str, final_file_path: str, language=None) -> pd.DataFrame:
        '''This function creates multiple csv files in the output drive'''
        new_df = pd.DataFrame(columns=['Text', 'language'])
        
        df = pd.read_table(final_file_path+f'cleaned\\{folders_name}\\{file_name}', quoting=QUOTE_NONE, header=None, names=['Text'])
        
        new_df['Text'] = df.iloc[:,0]
        new_df["language"] = language

        return new_df

    def text_cleaning(self, folders_name: str, file_name: str, final_file_path: str) -> str:
        '''This function performs multiple operations like cleaning text, removing html tags, and finally returns new file name'''
        in_text = open(final_file_path+f'txt\\{folders_name}\\{file_name}', encoding="utf-8", errors='ignore').read()

        cleaned = re.sub(r'<.*?>', '', in_text).lower().strip()

        if not os.path.exists(os.path.join(final_file_path, 'cleaned', folders_name)):
            os.makedirs(os.path.join(final_file_path, 'cleaned', folders_name), exist_ok=True)
        
        new_file_name = file_name.replace('.txt', '-cleaned.txt')
        open(final_file_path+f'cleaned\\{folders_name}\\{new_file_name}', 'w', encoding="utf-8", errors='ignore').write(cleaned)
        
        os.remove(final_file_path+f'txt\\{folders_name}\\{file_name}')

        return new_file_name

    def dataframe_creation(self, folders_name: str, files_name: str, final_text_file_path: str) -> None:
        '''This function reads all the text files in multiple directories parallely and converts into csv files'''
        language = folders_name
        
        new_file = self.text_cleaning(folders_name=folders_name, file_name=files_name, final_file_path=final_text_file_path)
        df2 = self.text_to_dataframe(folders_name=folders_name, file_name=new_file, language=language, final_file_path=final_text_file_path)

        if not os.path.exists(os.path.join(final_text_file_path, 'csv', folders_name)):
            os.makedirs(os.path.join(final_text_file_path, 'csv', folders_name), exist_ok=True)

        df2.to_csv(final_text_file_path+f'csv\\{folders_name}\\'+files_name.replace('.txt', '.csv'), index=False, header=True)

        os.remove(final_text_file_path+f'cleaned\\{folders_name}\\{new_file}')