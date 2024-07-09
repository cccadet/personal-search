"""
This script ingest data from PDF files and Youtube channels. 
"""

import os
from embedchain import App

# load chroma configuration from yaml file
app = App.from_config(config_path="ingestion/books.yaml")


paths = ['./ingestion/pdf/books']
for path in paths:
    files = os.listdir(path)
    pdfs = [f for f in files if f.endswith('.pdf')]

    for pdf in pdfs:
        app.add(os.path.join(path, pdf), data_type='pdf_file')
        

