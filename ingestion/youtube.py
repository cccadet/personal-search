"""
This script ingest data from PDF files and Youtube channels. 
"""

from embedchain import App

app = App.from_config(config_path="ingestion/youtube.yaml")

channels = ['@doisdedosdeteologia']
for channel in channels:
    app.add(channel, data_type="youtube_channel")
