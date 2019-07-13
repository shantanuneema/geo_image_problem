import io
import os
import pathlib
import requests
import zipfile

'''
Data Science Case Study
Codes to download and  extract the data for "geological_similarity" problem
link provided in file: 'Data Science Case Study Options 1.0.pdf' 
Option 2: Geological Image Similarity
'''

dir_path = pathlib.Path.cwd().parent

if os.path.isdir(str(dir_path)+ '\\geological_similarity'):
    print('Data is stored in', dir_path)
else:
    # unzipped the data from the handler
    print('Downloading and storing data in ', dir_path,'\geological_similarity...',sep='')
    res = requests.get('http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip')
    zipped_geo_data = zipfile.ZipFile(io.BytesIO(res.content))
    zipped_geo_data.extractall(dir_path)
    