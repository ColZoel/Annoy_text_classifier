import os
import requests
from urllib.parse import urlparse
from glob import glob

def check_compression(url):
    """
    Determines if the URL points to a tarball or zip.
    Raises an error if the URL points to a directory.

    :param url: The URL to check
    :raises ValueError: If the URL points to a directory or is not a tarball/zip file
    :return: The type of compressed file ('tar', 'zip') or raises an error
    """
    response = requests.head(url, allow_redirects=True)
    if response.status_code != 200:
        raise ValueError(f"Failed to access URL: {url}, Status Code: {response.status_code}")

    # Checking by MIME type
    content_type = response.headers.get('Content-Type', '').lower()
    if any(x in content_type for x in ['application/x-tar', 'application/gzip', 'application/zip']):
        return True

    # Checking by filename if Content-Disposition is provided
    parsed_url = urlparse(url)
    filename = parsed_url.path.split('/')[-1]
    if filename.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz', '.zip')):
        return True

    # Checking if URL points to a directory (usually ends with a slash)
    if filename == '' or filename.endswith('/'):
        raise ValueError("url points to a directory, not a file.")

    raise ValueError("The URL does not point to a recognized compressed file (tarball or zip).")


def import_from_server(url, local_directory):
    """
    Save the model from a remote server to the local directory.
    This is useful if HF models are saved remotely.
    :param url: URL of zip file
    :param local_directory: Local directory to save the model (adds a modules folder here)
    :return:
    """

    # check if the remote file is valid compression
    try:
        check_compression(url)
    except ValueError as e:
        raise ValueError(f"{e}")

    # Ensure the local directory exists
    if os.path.basename(local_directory) != "models":
        os.makedirs(os.path.join(local_directory, "models"), exist_ok=True)
        local_directory = os.path.join(local_directory, "models")

    filename = urlparse(url).path.split('/')[-1]

    # Extract the filename from the URL
    local_filename = os.path.join(local_directory, filename)

    # Send a GET request to the URL
    print(f"Pulling {filename}. . .")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Write the content to a local file
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)

    print(f"{filename} saved to {local_filename}")

    return local_filename


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


def import_local(model_name):
    """
    Import a model from the local directory.
    :param model_name: Name of the model to import
    :return: The model object
    """
    models = local_models()
    if model_name not in models:
        raise ValueError(f"{model_name} not found in {os.getcwd()}/models. Available models: {models}")
    return glob(f"models/{model_name}")[0]