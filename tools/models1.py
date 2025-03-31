import os
from glob import glob
import requests
from urllib.parse import urlparse


def local_models(local_directory=None):
    """
    List the models available in the local directory.
    :param local_directory: Local directory containing the models
    :return: List of models available in the local directory
    """

    if local_directory is None:
        local_directory = os.getcwd()

    if not os.path.exists(os.path.join(local_directory, "models")):
        raise FileNotFoundError(f"No models directory in {local_directory}.")

    models = os.listdir(os.path.join(local_directory, "models"))
    return models


def import_local(model_name, *args, **kwargs):
    """
    Import a model from the local directory.
    :param model_name: Name of the model to import
    :return: The model object
    """
    models = local_models(*args, **kwargs)
    if model_name not in models:
        raise ValueError(f"{model_name} not found in {os.getcwd()}/models. Available models: {models}")
    return glob(f"models/{model_name}")[0]


class ModelDownloader:
    """
    Tool to check if a URL points to a compressed model file (tarball or zip). This is used to check that
    remote models are compressed and not directories.
    """

    def __init__(self, url, local_directory):
        self.url = url
        self.local_directory = local_directory

    def check_compression(self):
        """
        Determines if the URL points to a tarball or zip.
        Raises an error if the URL points to a directory.

        :raises ValueError: If the URL points to a directory or is not a tarball/zip file
        :return: The type of compressed file ('tar', 'zip') or raises an error
        """
        response = requests.head(self.url, allow_redirects=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to access URL: {self.url}, Status Code: {response.status_code}")

        # Checking by MIME type
        content_type = response.headers.get('Content-Type', '').lower()
        if any(x in content_type for x in ['application/x-tar', 'application/gzip', 'application/zip']):
            return True

        # Checking by filename if Content-Disposition is provided
        parsed_url = urlparse(self.url)
        filename = parsed_url.path.split('/')[-1]
        if filename.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz', '.zip')):
            return True

        # Checking if URL points to a directory (usually ends with a slash)
        if filename == '' or filename.endswith('/'):
            raise ValueError("url points to a directory, not a file.")

        raise ValueError("The URL does not point to a recognized compressed file (tarball or zip).")


