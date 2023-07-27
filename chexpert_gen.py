#!/usr/bin/env python

import argparse
import yaml
from utils.chexpert_preprocess import create_testset
from utils.logger import create_logger

def main(args):
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    logger = create_logger(output_dir=config['log_path'], name='Generating datasets')
    create_testset(args.chexpert_root, args.test_path, logger)
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/swin384.yaml', help='load the config file')
    parser.add_argument('--chexpert_root', type=str, default='/data', help='the path where your chexpert cxr files located')
    parser.add_argument('--test_path', type=str, default='./datasets/chexpert/test', help='the path where the testset you want to put')
    args = parser.parse_args()
    main(args)