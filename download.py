import argparse
import requests
import tarfile
import zipfile
import re
import os
import itertools
import shutil
from tqdm import tqdm

BLOCK_SIZE = 1024

def flatten(destination):
    all_files = []
    for root, _, files in itertools.islice(os.walk(destination), 1, None):
        for f in files:
            all_files.append(os.path.join(root, f))
    
    for f in all_files:
        name = f.split('\\')[-1]
        if re.search("\.mid$", name):
            if not os.path.isfile(os.path.join(destination, name)):
                shutil.move(f, destination)
        else:
            os.remove(f)

    # Cleanup empty directories
    for item in os.listdir(destination):
        path = os.path.join(destination, item)
        if os.path.isdir(path):
            shutil.rmtree(path)


def download(url, destination):
    print(f'Downloading {url}...')
    response = requests.get(url, stream=True)
    total_size = response.headers.get('content-length')
    try:
        total_size = int(total_size)
    except:
       pass

    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(destination, 'wb') as f:
        for data in response.iter_content(BLOCK_SIZE):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()


def extract(source, destination, type):
    print(f'Extracting {source}...')
    if type == 'tar':
        with tarfile.open(source) as extract:
            total = len(extract.getnames())
            for f in tqdm(extract, total=total):
                f.name = re.sub(r'[:?"* ]', '_', f.name)
                extract.extract(f, destination)
        os.remove(source)
    elif type == 'zip':
        with zipfile.ZipFile(source) as extract:
            total = len(extract.namelist())
            for f in tqdm(extract.infolist(), total=total):
                extract.extract(f, destination)

            extract.close()
        os.remove(source)
    else:
        print('Invalid file type.')


def download_lahk(output_dir):
    download_url = 'http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz'
    file_name = os.path.join(output_dir, 'downloaded.tar.gz')
    download(download_url, file_name)

    extract(file_name, output_dir, 'tar')

    print('Finishing up...')
    flatten(output_dir)

    all_files = os.listdir(output_dir)

    # Remove duplicate songs
    for f in all_files:
        if re.search("\.[0-9]\.mid$", f):
            os.remove(os.path.join(output_dir, f))


def download_runescape_ost(output_dir):
    download_url = 'https://drive.google.com/u/0/uc?id=0Bx3m4LvdozzGRE5hRlJjWFFGR0E&export=download'
    file_name = os.path.join(output_dir, 'downloaded.zip')
    download(download_url, file_name)

    extract(file_name, output_dir, 'zip')

    print('Finishing up...')
    # Remove unneeded folder
    shutil.rmtree(os.path.join(output_dir, '__MACOSX'))

    flatten(output_dir)
    
    all_files = os.listdir(output_dir)
    
    # Remove songs past 1000 as they require a special soundfont
    for f in filter(lambda x: re.search("[0-9]{4}\.mid$", x), all_files):
        os.remove(os.path.join(output_dir, f))


def download_dataset(dataset, output):
    if not os.path.exists(output):
        os.mkdir(output)

    if len(os.listdir(output)) > 0:
        print('Error: Output directory is not empty.')
        exit()

    if dataset == 'rs_ost':
        download_runescape_ost(output)
    elif dataset == 'lahk':
        download_lahk(output)
    else:
        print('Error: Invalid dataset. Choose between \'rs_ost\' and \'lahk\'.')
        exit()

    print(f'Dataset extracted to \'{output}\'.')


def main():
    parser = argparse.ArgumentParser(description='Train the model on a dataset.')
    parser.add_argument('-d', '--dataset', type=str, help='which dataset to download (rs_ost or lahk)', required=True)
    parser.add_argument('-o', '--output', type=str, help='output directory for the dataset', required=True)

    args = parser.parse_args()
    download_dataset(args.dataset, args.output)
    
if __name__ == "__main__":
    main()
