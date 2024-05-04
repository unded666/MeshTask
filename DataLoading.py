import requests
import zipfile
import pandas as pd
import os
import io

DATASET_URL = 'https://www.kaggle.com/datasets/bhavikjikadara/student-study-performance/download?datasetVersionNumber=1'


response = requests.get(DATASET_URL)
zip_file_in_memory = io.BytesIO(response.content)
with zipfile.ZipFile(zip_file_in_memory, 'r') as zip_ref:
    for filename in zip_ref.namelist():
        if filename.endswith('.xlsx'):
            excel_file = zip_ref.open(filename)
            df = pd.read_excel(excel_file)
            break