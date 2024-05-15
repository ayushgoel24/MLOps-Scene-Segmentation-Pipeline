import requests
import zipfile
import os

class FileUtils:

    @staticmethod
    def download_file(url, destination):
        """
        Downloads a file from a given URL and saves it to a specified local destination.

        Args:
        - url (str): The URL of the file to download.
        - destination (str): The local path where the file should be saved.
        """
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()
        
        # Open a local file in binary write mode
        with open(destination, 'wb') as f:
            # Iterate over the response data in chunks
            for chunk in response.iter_content(chunk_size=8192):
                # Write the chunk to the local file
                f.write(chunk)

    @staticmethod
    def unzip_file(zip_path, extract_to):
        """
        Extracts a ZIP file to a specified directory.

        Args:
        - zip_path (str): The path to the ZIP file.
        - extract_to (str): The directory where the contents of the ZIP should be extracted.
        """
        # Ensure the extraction path exists
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)

        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all the contents into the directory
            zip_ref.extractall(extract_to)
            print(f"Files have been extracted to: {extract_to}")