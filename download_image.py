from urllib.request import urlretrieve
from os import path
import csv

dataset = 'dataset/dataset.csv'

download_dir = '/home/aristo/Downloads/raw/'

def check_file_downloaded(name, dir):
    path_file = path.join(*[dir, name])
    return path.exists(path_file)



with open(dataset, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for idx, row in enumerate(reader):
        # if idx > 19:
        #     break
        filename = f"{row[0]}-{row[1]}-{row[2]}.jpg"
        
        if check_file_downloaded(filename, download_dir):
            continue

        image_url = row[3]
        fullpathname = path.join(*[download_dir, filename])
        try:
            urlretrieve(image_url, fullpathname)
        except: 
            print('Image Couldn\'t be retreived')

