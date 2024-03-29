import cv2
import pandas as pd
import numpy as np
import os
import glob
from color_matcher.io_handler import load_img_file
from color_matcher.normalizer import Normalizer
from color_matcher import ColorMatcher
import cv2
import tqdm
import color_matcher
from multiprocessing import Process, Pool
import multiprocessing as mp
import sys
import time
from tqdm.contrib.concurrent import process_map
import istarmap
import gc

def process_one_pair(p_source, p_target, target_img_paths):
    CM = ColorMatcher()
    loc = os.path.join(p_source, 'images')
    for img_name in os.listdir(loc):
        target_img_path = np.random.choice(target_img_paths, 1)[0]
        img_ref = load_img_file(target_img_path)
        img_src = load_img_file(loc + '/' + img_name)
        img = CM.transfer(src=img_src, ref=img_ref, method='hm-mvgd-hm')
        img = Normalizer(img).uint8_norm()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(p_source + 'synth' + p_target + '/' + img_name, img)

def synth(img_src_path, img_ref_path, img_name):
    CM = ColorMatcher()
    img_src = load_img_file(img_src_path)
    img_ref = load_img_file(img_ref_path)
    img = CM.transfer(src=img_src, ref=img_ref, method='hm-mvgd-hm')
    img = Normalizer(img).uint8_norm()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_name, img)
    del CM
    del img_src
    del img_ref
    gc.collect()

if __name__ == '__main__':
    # source_ls = ['PAL_2021']
    # target_ls = ['LL', 'HUA', 'PAL', 'RAN', 'TAK']
    source_ls = ['TAK']
    target_ls = ['LL', 'HUA', 'PAL', 'RAN']
    for source in source_ls:
        for target in target_ls:
            if os.path.exists(source + 'synth' + target):
                os.system('rm -r ' + source + 'synth' + target)
            os.mkdir(source + 'synth' + target)
    target_img_paths = {}

    for target in target_ls:
        target_img_paths[target] = glob.glob(target + '/images/*.JPG', recursive=True)

    args = []
    print('preparing parse args')
    for target in target_ls:
        for source in source_ls:
            loc = os.path.join(source, 'images')
            for img_name in os.listdir(loc):
                if img_name.split('.')[1] == 'JPG':
                    target_img_path = np.random.choice(target_img_paths[target], 1)[0]
                    # img_ref = load_img_file(target_img_path)
                    # img_src = load_img_file(loc + '/' + img_name)
                    img_path = source + 'synth' + target + '/' + img_name
                    args.append((loc+'/'+img_name, target_img_path, img_path))
    print('done preparing parse args')

    with mp.Pool(mp.cpu_count()) as pool:
        print('#cpu available: %d'%mp.cpu_count())
        print('#args to run: %d'%len(args))
        for _ in tqdm.tqdm(pool.istarmap(synth, args),
                           total=len(args)):
            pass