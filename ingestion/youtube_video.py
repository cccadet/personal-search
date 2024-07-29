"""
This script ingest data from PDF files and Youtube channels. 
"""

import os
import ast
from embedchain import App
from dotenv import load_dotenv

load_dotenv()

app = App.from_config(config_path="ingestion/youtube_video.yaml")

urls = os.environ.get("YOUTUBE_VIDEOS")
urls = ast.literal_eval(urls)
for url in urls:
    app.add(url, data_type="youtube_video")
