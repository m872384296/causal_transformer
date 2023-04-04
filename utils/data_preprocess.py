import pandas as pd
import numpy as np
import shutil
import os
from tqdm import tqdm

def create_dataset(cxr_root, mimic_root, target_path, logger):
    cxr_path = os.path.join(cxr_root, 'files/mimic-cxr-jpg/2.0.0/')
    physio_path = os.path.join(cxr_root, 'files/mimic-cxr-jpg/2.0.0/files/')
    logger.info('Loading data files......')
    label = pd.read_csv(cxr_path + 'mimic-cxr-2.0.0-chexpert.csv.gz', compression='gzip')
    meta = pd.read_csv(cxr_path + 'mimic-cxr-2.0.0-metadata.csv.gz', compression='gzip')
    spl = pd.read_csv(cxr_path + 'mimic-cxr-2.0.0-split.csv.gz', compression='gzip')
    adm = pd.read_csv(os.path.join(mimic_root, 'admissions.csv.gz'), compression='gzip')
    pat = pd.read_csv(os.path.join(mimic_root, 'patients.csv.gz'), compression='gzip')
    omr = pd.read_csv(os.path.join(mimic_root, 'omr.csv.gz'), compression='gzip')
    chunk = pd.read_csv(os.path.join(mimic_root, 'labevents.csv.gz'), chunksize=1000000, compression='gzip')
    lab = pd.concat(chunk)
    label = label[['subject_id', 'study_id', 'Pneumonia']].dropna()
    label = label.drop(label[label['Pneumonia']==-1].index)
    label['Pneumonia'] = label['Pneumonia'].astype(int)
    meta = meta[['dicom_id', 'subject_id', 'study_id', 'ViewPosition', 'StudyDate']]
    meta = meta[meta['study_id'].isin(label['study_id'])]
    meta = meta[meta['ViewPosition'].isin(['AP', 'PA'])]
    spl = spl[['dicom_id', 'subject_id', 'study_id', 'split']]
    spl = spl[spl["dicom_id"].isin(meta['dicom_id'])]
    adm_id = np.unique(adm['subject_id'], return_index=True)[1]
    adm = adm.iloc[adm_id, :]
    adm = adm[['subject_id', 'race']]
    adm['race'] = adm['race'].str.split(' - ', expand=True)[0]
    adm = adm.replace('HISPANIC OR LATINO', 'HISPANIC/LATINO')
    adm = adm.replace('UNKNOWN', 'OTHER')
    race_freq = (adm['race'].value_counts())/adm.shape[0]
    less_freq_races = race_freq[race_freq<=0.01]
    adm.loc[adm['race'].isin(less_freq_races.index.tolist())] = 'OTHER'
    pat = pat[['subject_id', 'gender', 'anchor_age', 'dod']]
    pat['dod'] = pat['dod'].notnull()
    bmi = omr[omr['result_name']=='BMI (kg/m2)'].copy()
    bmi['result_value'] = bmi['result_value'].astype('float')
    bmi['chartdate'] = bmi['chartdate'].str.split('-').str.join('').astype('int')
    blood = omr[omr['result_name']=='Blood Pressure'].copy()
    blood = pd.concat([blood, blood['result_value'].str.split('/', expand=True)], axis=1)
    blood = blood.drop('result_value', axis=1)
    blood[[0, 1]] = blood[[0, 1]].astype('float')
    blood['chartdate'] = blood['chartdate'].str.split('-').str.join('').astype('int')
    feature = []
    itemid = [50802, 50806, 50815, 50816, 50817, 50819, 50822, 50824, 50825, 50826, 52024]
    for i in range(len(itemid)):
        feature_i = lab[lab['itemid']==itemid[i]].copy()
        feature_i = feature_i[['subject_id', 'itemid', 'charttime', 'valuenum']].dropna()
        date = feature_i['charttime'].str.split(' ', expand=True)[0]
        feature_i['charttime'] = date.str.split('-').str.join('').astype('int')
        feature.append(feature_i)
    logger.info('Finish loading !!!')
    l = 0
    m = 0
    n = 0
    label_train = []
    label_val = []
    label_test = pd.DataFrame(columns=['file', 'label'])
    columns=['race', 'gender', 'age', 'death', 'BMI', 'SBP', 'DBP',
             'BE', 'chloride', 'o2flow', 'oxygen', 'sao2', 'peep',
             'potassium', 'sodium', 'temperature', 'TV', 'creatinine']
    confounder = pd.DataFrame(columns=columns)
    train_dir = os.path.join(target_path, 'train/')
    val_dir = os.path.join(target_path, 'val/')
    test_dir = os.path.join(target_path, 'test/')
    if os.path.exists(train_dir) == False:
        os.makedirs(train_dir)
    if os.path.exists(val_dir) == False:
        os.makedirs(val_dir)
    if os.path.exists(test_dir) == False:
        os.makedirs(test_dir)
    logger.info('Creating datasets......')
    for i in tqdm(spl.index, desc='Creating'):
        split = spl.loc[i, 'split']
        suid = spl.loc[i, 'subject_id']
        stid = spl.loc[i, 'study_id']
        dicid = spl.loc[i, 'dicom_id']
        date = meta.loc[i, 'StudyDate']
        if split == 'train':        
            if suid in range(10000000, 11000000):
                path = physio_path + 'p10/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(11000000, 12000000):
                path = physio_path + 'p11/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(12000000, 13000000):
                path = physio_path + 'p12/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(13000000, 14000000):
                path = physio_path + 'p13/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(14000000, 15000000):
                path = physio_path + 'p14/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(15000000, 16000000):
                path = physio_path + 'p15/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(16000000, 17000000):
                path = physio_path + 'p16/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(17000000, 18000000):
                path = physio_path + 'p17/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(18000000, 19000000):
                path = physio_path + 'p18/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(19000000, 20000000):
                path = physio_path + 'p19/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = train_dir + str(l) + '.jpg'
                shutil.copy(path, save_path)
            label_train.append(label.loc[label['study_id']==stid, 'Pneumonia'].values)
            if suid in list(adm['subject_id']):
                confounder.loc[l, ['race']] = adm.loc[adm['subject_id']==suid, 'race'].values
            else:
                confounder.loc[l, 'race'] = np.nan
            if suid in list(pat['subject_id']):
                confounder.loc[l, ['gender', 'age', 'death']] = pat.loc[pat['subject_id']==suid, ['gender', 'anchor_age', 'dod']].values
            else:
                confounder.loc[l, ['gender', 'age', 'death']] = np.nan
            if suid in list(bmi['subject_id']):
                dates = bmi.loc[bmi['subject_id']==suid, 'chartdate'] - date
                bmi_i = bmi.loc[dates[dates.abs().eq(dates.abs().min())].index]['result_value'].mean()
                confounder.loc[l, 'BMI'] = bmi_i
            else:
                confounder.loc[l, 'BMI'] = np.nan
            if suid in list(blood['subject_id']):
                dates = blood.loc[blood['subject_id']==suid, 'chartdate'] - date
                sbp = blood.loc[dates[dates.abs().eq(dates.abs().min())].index][0].mean()
                dbp = blood.loc[dates[dates.abs().eq(dates.abs().min())].index][1].mean()
                confounder.loc[l, ['SBP', 'DBP']] = [sbp, dbp]
            else:
                confounder.loc[l, ['SBP', 'DBP']] = np.nan
            for idx in range(len(itemid)):
                if suid in list(feature[idx]['subject_id']):
                    dates = feature[idx].loc[feature[idx]['subject_id']==suid, 'charttime'] - date
                    conf = feature[idx].loc[dates[dates.abs().eq(dates.abs().min())].index]['valuenum'].mean()
                    confounder.loc[l, columns[7+idx]] = conf
                else:
                    confounder.loc[l, columns[7+idx]] = np.nan 
            l = l + 1
        elif split == 'validate':
            if suid in range(10000000, 11000000):
                path = physio_path + 'p10/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(11000000, 12000000):
                path = physio_path + 'p11/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(12000000, 13000000):
                path = physio_path + 'p12/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(13000000, 14000000):
                path = physio_path + 'p13/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(14000000, 15000000):
                path = physio_path + 'p14/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(15000000, 16000000):
                path = physio_path + 'p15/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(16000000, 17000000):
                path = physio_path + 'p16/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(17000000, 18000000):
                path = physio_path + 'p17/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(18000000, 19000000):
                path = physio_path + 'p18/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(19000000, 20000000):
                path = physio_path + 'p19/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = val_dir + str(m) + '.jpg'
                shutil.copy(path, save_path)
            label_val.append(label.loc[label['study_id']==stid, 'Pneumonia'].values)
            m = m + 1
        else:
            if suid in range(10000000, 11000000):
                path = physio_path + 'p10/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(11000000, 12000000):
                path = physio_path + 'p11/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(12000000, 13000000):
                path = physio_path + 'p12/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(13000000, 14000000):
                path = physio_path + 'p13/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(14000000, 15000000):
                path = physio_path + 'p14/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(15000000, 16000000):
                path = physio_path + 'p15/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(16000000, 17000000):
                path = physio_path + 'p16/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(17000000, 18000000):
                path = physio_path + 'p17/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(18000000, 19000000):
                path = physio_path + 'p18/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            elif suid in range(19000000, 20000000):
                path = physio_path + 'p19/p' + str(suid) + '/s' + str(stid) + '/' + dicid + '.jpg'
                save_path = test_dir + dicid + '.jpg'
                shutil.copy(path, save_path)
            label_i = label.loc[label['study_id']==stid, 'Pneumonia'].values.tolist()
            label_test.loc[n] = [dicid+'.jpg', label_i[0]]
            n = n + 1
    confounder.to_csv(train_dir + 'confounder.csv', index=False)
    pd.DataFrame(label_train).to_csv(train_dir + 'label.txt', index=False, header=False, lineterminator="\r\n")
    pd.DataFrame(label_val).to_csv(val_dir + 'label.txt', index=False, header=False, lineterminator="\r\n")
    label_test.to_csv(test_dir + 'label.csv', index=False, header=False)
    logger.info('Finish creating datasets !!!')