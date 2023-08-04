import pandas as pd
import numpy as np
import shutil
import os, tarfile
from tqdm import tqdm
from zipfile import ZipFile

def create_testset(openi_root, target_path, logger):
    openi_zip = os.path.join(openi_root, 'archive.zip')
    logger.info('Unzipping testsets......')
    with ZipFile(openi_zip) as zfs:
        for zf in tqdm(zfs.infolist(), desc='Extracting', dynamic_ncols=True):
            zfs.extract(zf, openi_root)
    logger.info('Finish unzipping testsets !!!')    
    proj = pd.read_csv(os.path.join(openi_root, 'indiana_projections.csv'))
    rep = pd.read_csv(os.path.join(openi_root, 'indiana_reports.csv'), index_col=0)
    label = rep['MeSH']
    label = label[label.str.contains('normal') | label.str.contains('Pneumonia')]
    label[label.str.contains('normal')] = '0'
    label[label.str.contains('Pneumonia')] = '1'
    proj = proj[proj['uid'].isin(label.index)]
    proj = proj[proj['projection'] == 'Frontal']
    openi_path = os.path.join(openi_root, 'images/images_normalized/')
    label_file = pd.DataFrame(columns=['file', 'label'])
    if os.path.exists(target_path) == False:
        os.makedirs(target_path)
    logger.info('Creating testsets......')
    for i in tqdm(range(proj.shape[0]), desc='Creating', dynamic_ncols=True):
        filename = proj['filename'].iloc[i]
        path = f'{openi_path}{filename}'
        save_path = os.path.join(target_path, f'{i}.jpg')
        shutil.copy(path, save_path)
        label_i = label[label.index == proj['uid'].iloc[i]]
        label_file.loc[i] = [f'{i}.jpg', label_i.values[0]]
    label_file.to_csv(os.path.join(target_path, 'label.csv'), index=False, header=False)
    logger.info('Finish creating testsets !!!')