#!/usr/bin/env python

import argparse
import yaml
from utils.data_preprocess import create_dataset
from utils.logger import create_logger

def main(args):
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    logger = create_logger(output_dir=config['log_path'], name='Generating datasets')
    create_dataset(args.cxr_root, args.mimic_root, config['data_path'], logger)
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/mimic.yaml', help='load the config file')
    parser.add_argument('--cxr_root', type=str, default='/data/physionet.org', help='the path where your mimic cxr files located')
    parser.add_argument('--mimic_root', type=str, default='/data/mimic_iv', help='the path where your mimic iv gzip files located')
    args = parser.parse_args()
    main(args)