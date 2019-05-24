import os
import requests
import urllib

from summarize.data.io import JsonlReader


# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_from_google_drive(file_id: str, expected_size: int, file_path: str) -> None:
    """
    Downloads a file from Google Drive if it does not already exist locally.

    Parameters
    ----------
    file_id: ``str``, required.
        The ID of the file to download. The ID can be found in the url of the
        file on Google Drive.
    expected_size: ``int``, required.
        The expected file size in bytes.
    file_path: ``str``, required.
        The path to where the file should be saved. The parent directories are
        created if they don't exist.
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination: str, expected_size: int):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            size = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
            if size != expected_size:
                raise Exception(f'{destination} is unexpected size. Found {size}, expected {expected_size}')

    if os.path.exists(file_path):
        print(f'Skipping downloading {file_path}')
        return

    print(f'Downloading {file_path}')
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, file_path, expected_size)


def download_url_to_file(url: str, file_path: str, force: bool = False) -> None:
    """
    Downloads a url to a local file if it does not already exist. The directory
    path to the download file will be created if it does not exist.

    Parameters
    ----------
    url: ``str``, required.
        The url of the file to download.
    file_path: ``str``, required.
        The path where the file should be saved.
    force: ``bool``, optional (default = ``False``)
        If false, the file is not downloaded if it exists. Otherwise, the
        file will be downloaded no matter what.
    """
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if os.path.exists(file_path) and not force:
        print(f'Skipping downloading {url}')
        return

    print(f'Downloading {url} to {file_path}')
    urllib.request.urlretrieve(url, file_path)


def assert_line_count(file_path: str, expected_count: int):
    """
    Checks to make sure the input file has an expected number of lines. The
    file should be a jsonl file.

    Parameters
    ----------
    file_path: ``str``
        The file to check.
    expected_count: ``int``
        The expected number of lines.

    Raises
    ------
    Exception: If the actual number of lines is not equal to the expected number.
    """
    count = 0
    with JsonlReader(file_path) as f:
        for _ in f:
            count += 1
    if count != expected_count:
        raise Exception(f'Unexpected number of lines in {file_path}. Found {count}, expected {expected_count}')
    print(f'{file_path} has expected number of lines')
