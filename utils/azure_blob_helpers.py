import os
#pip install azure-storage-blob
from azure.storage.blob import BlockBlobService


# Create the BlockBlobService that is used to call the Blob service for the storage account.
account_name = ''
token = ''
blob_name = ''

block_blob_service = BlockBlobService(account_name=account_name, account_key=None, sas_token=token)


def get_blobs_from_dir(path_prefix):
    """

    Args:
        path_prefix: directory path to a folder containing .wav files

    Returns: list of blobs that are .wav files

    """
    all_blobs = []
    generator =block_blob_service.list_blobs(blob_name, prefix=path_prefix)
    for blob in generator:
        if blob.name.lower().endswith('.wav'):
            all_blobs.append(blob)
    return all_blobs


def download_blob(blob, local_base_path='./'):
    """

    Args:
        blob: blob object from get_blobs_from_dir list
        local_base_path: local directory to save file.
                         Note: if blob contains subdirectories, those folders will be created from the local_base_path

    Returns: path to saved file

    """
    path, filename = os.path.split(blob.name)
    fullpath = os.path.join(local_base_path, path)
    fullpath = os.path.abspath(fullpath)
    os.makedirs(fullpath, exist_ok=True)
    save_path = os.path.join(fullpath, filename)
    block_blob_service.get_blob_to_path(blob_name, blob.name, save_path)
    return save_path
