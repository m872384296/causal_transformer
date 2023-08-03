import pandas as pd
import numpy as np
import shutil
import os, tarfile
from tqdm import tqdm
from zipfile import ZipFile

def create_testset(chestxray8_root, target_path, logger):
    chest_zip = os.path.join(chestxray8_root, 'CXR8.zip')
    logger.info('Unzipping testsets......')
    with ZipFile(chest_zip) as zfs:
        for zf in tqdm(zfs.infolist(), desc='Extracting'):
            zfs.extract(zf, chestxray8_root)
    for i in tqdm(range(12), desc='Copying'):
        t = tarfile.open(os.path.join(chestxray8_root, f'CXR8/images/images_{str(i + 1).zfill(3)}.tar.gz'))
        t.extractall(path = os.path.join(chestxray8_root, 'CXR8/images'))
    logger.info('Finish unzipping testsets !!!')    
    chest_path = os.path.join(chestxray8_root, 'CXR8')
    meta = pd.read_csv(f'{chest_path}/Data_Entry_2017_v2020.csv', index_col=0)
    ids = np.loadtxt(f'{chest_path}/train_val_list.txt', dtype=str)
    label = meta.loc[ids, 'Finding Labels']
    label = label[label.str.contains('No Finding') | label.str.contains('Pneumonia')]
    label[label.str.contains('No Finding')] = '0'
    label[label.str.contains('Pneumonia')] = '1'
    label_file = pd.DataFrame(columns=['file', 'label'])
    if os.path.exists(target_path) == False:
        os.makedirs(target_path)
    logger.info('Creating testsets......')
    for k in tqdm(range(label.shape[0]), desc='Creating'):
        path = os.path.join(chestxray8_root, f'CXR8/images/images/{label.index[k]}')
        save_path = os.path.join(target_path, f'{k}.jpg')
        shutil.copy(path, save_path)
        label_file.loc[k] = [f'{k}.jpg', label[k]]
    label_file.to_csv(os.path.join(target_path, 'label.csv'), index=False, header=False)
    logger.info('Finish creating testsets !!!')