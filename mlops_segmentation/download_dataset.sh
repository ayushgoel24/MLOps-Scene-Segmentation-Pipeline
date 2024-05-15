#!/bin/bash

# Google Drive file ID
FILE_ID="1mBAJTJ7fNmkcLRUvZa3yTr-hNx6fJVxS"
# The name you want to save your file as
FILE_NAME="cityscapes.zip"

# Create the data directory if it does not exist
mkdir -p data/cityscapes/

# Using curl to download the file
echo "Downloading the dataset from Google Drive..."
gdown --id $FILE_ID --output "data/$FILE_NAME"
echo "Download completed."

# Unzip the dataset
echo "Unzipping the dataset..."
unzip "data/$FILE_NAME" -d data
echo "Unzipping completed."

# Remove the zip file if no longer needed
rm "data/$FILE_NAME"

echo "Dataset is ready for use."
