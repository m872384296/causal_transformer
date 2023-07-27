import pandas as pd
import numpy as np
import shutil
import os
from tqdm import tqdm
from zipfile import ZipFile

def create_testset(chexpert_root, target_path, logger):
    chex_zip = os.path.join(chexpert_root, 'CheXpert-v1.0.zip')
    logger.info('Unzipping testsets......')
    with ZipFile(chex_zip) as zfs:
        for zf in tqdm(zfs.infolist(), desc='Extracting'):
            zfs.extract(zf, chexpert_root)
    logger.info('Finish unzipping testsets !!!')
    chexpert_path = os.path.join(chexpert_root, 'CheXpert-v1.0/train.csv')
    meta = pd.read_csv(chexpert_path)
    meta = meta[['Path', 'AP/PA', 'Pneumonia']].dropna()
    label = meta[['Path', 'Pneumonia']]
    label = label.drop(label[label['Pneumonia']==-1].index)
    label['Pneumonia'] = label['Pneumonia'].astype(int)
    label_file = pd.DataFrame(columns=['file', 'label'])
    if os.path.exists(target_path) == False:
        os.makedirs(target_path)
    logger.info('Creating testsets......')
    for i in tqdm(range(label.shape[0]), desc='Creating'):
        path = os.path.join(chexpert_root, label.iloc[i, 0])
        save_path = os.path.join(target_path, str(i) + '.jpg')
        shutil.copy(path, save_path)
        label_file.loc[i] = [str(i) +'.jpg', label.iloc[i, 1]]
    label_file.to_csv(os.path.join(target_path, 'label.csv'), index=False, header=False)
    logger.info('Finish creating testsets !!!')