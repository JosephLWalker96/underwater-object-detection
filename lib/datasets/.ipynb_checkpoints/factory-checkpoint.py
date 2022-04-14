# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.KITTI import KITTI
from datasets.cityscapes import cityscapes
from datasets.bdd100k import bdd100k
from datasets.HUA import HUA
from datasets.HUAsynthPAL import HUAsynthPAL
from datasets.PAL import PAL
from datasets.HUAsynthPAL2021 import HUAsynthPAL2021
from datasets.PAL2021 import PAL2021
from datasets.LL import LL
from datasets.HUAsynthLL import HUAsynthLL
from datasets.RAN import RAN
from datasets.HUAsynthRAN import HUAsynthRAN
from datasets.TAK import TAK
from datasets.HUAsynthTAK import HUAsynthTAK

from datasets.PALsynthTAK import PALsynthTAK
from datasets.PALsynthHUA import PALsynthHUA
from datasets.PALsynthPAL2021 import PALsynthPAL2021
from datasets.PALsynthRAN import PALsynthRAN
from datasets.PALsynthLL import PALsynthLL

import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

# Set up KITTI
for split in ['train', 'val', 'synthCity', 'trainval']:
  name = 'KITTI_{}'.format(split)
  __sets[name] = (lambda split=split, year=year: KITTI(split))

# Set up cityscapes
for split in ['train', 'val', 'foggytrain', 'foggyval', 'synthFoggytrain', 'synthBDDdaytrain', 'synthBDDdayval']:
  name = 'cityscapes_{}'.format(split)
  __sets[name] = (lambda split=split, year=year: cityscapes(split))
    
# Set up HUA
for split in ['train', 'val', 'synthPALtrain', 'synthPALval','synthPAL2021train', 'synthPAL2021val', 'synthLLtrain', 'synthLLval', 'synthRANtrain', 'synthRANval', 'synthTAKtrain', 'synthTAKval', 'trainval']:
  name = 'HUA_{}'.format(split)
  __sets[name] = (lambda split=split: HUA(split))
    
# Set up HUAsynthPAL
for split in ['train', 'val', 'trainval']:
  name = 'HUAsynthPAL_{}'.format(split)
  __sets[name] = (lambda split=split: HUAsynthPAL(split))   
    
# Set up HUAsynthLL
for split in ['train', 'val', 'trainval']:
  name = 'HUAsynthLL_{}'.format(split)
  __sets[name] = (lambda split=split: HUAsynthLL(split))   
    
# Set up HUAsynthPAL2021
for split in ['train', 'val', 'trainval']:
  name = 'HUAsynthPAL2021_{}'.format(split)
  __sets[name] = (lambda split=split: HUAsynthPAL2021(split))    
    
# Set up HUAsynthRAN
for split in ['train', 'val', 'trainval']:
  name = 'HUAsynthRAN_{}'.format(split)
  __sets[name] = (lambda split=split: HUAsynthRAN(split))   

# Set up HUAsynthTAK
for split in ['train', 'val', 'trainval']:
  name = 'HUAsynthTAK_{}'.format(split)
  __sets[name] = (lambda split=split: HUAsynthTAK(split))     
    
# Set up PAL
for split in ['train', 'val', 'trainval']:
  name = 'PAL_{}'.format(split)
  __sets[name] = (lambda split=split: PAL(split))
    
# Set up LL
for split in ['train', 'val', 'trainval']:
  name = 'LL_{}'.format(split)
  __sets[name] = (lambda split=split: LL(split))
    
# Set up PAL2021
for split in ['train', 'val', 'trainval']:
  name = 'PAL2021_{}'.format(split)
  __sets[name] = (lambda split=split: PAL2021(split))
    
# Set up RAN
for split in ['train', 'val', 'trainval']:
  name = 'RAN_{}'.format(split)
  __sets[name] = (lambda split=split: RAN(split))
    
# Set up TAK
for split in ['train', 'val', 'trainval']:
  name = 'TAK_{}'.format(split)
  __sets[name] = (lambda split=split: TAK(split))
    
# Set up bdd100k
for split in ['train', 'val', 'daytrain', 'dayval', 'nighttrain', 'nightval', 'citydaytrain', 'citydayval', 'cleardaytrain', 'cleardayval', 'rainydaytrain', 'rainydayval']:
  name = 'bdd100k_{}'.format(split)
  __sets[name] = (lambda split=split, year=year: bdd100k(split))

# Set up PALsynthHUA
for split in ['train', 'val', 'trainval']:
  name = 'PALsynthHUA_{}'.format(split)
  __sets[name] = (lambda split=split: PALsynthHUA(split))  
    
# Set up PALsynthPAL2021
for split in ['train', 'val', 'trainval']:
  name = 'PALsynthPAL2021_{}'.format(split)
  __sets[name] = (lambda split=split: PALsynthPAL2021(split))  

# Set up PALsynthRAN
for split in ['train', 'val', 'trainval']:
  name = 'PALsynthRAN_{}'.format(split)
  __sets[name] = (lambda split=split: PALsynthRAN(split))      

# Set up PALsynthTAK
for split in ['train', 'val', 'trainval']:
  name = 'PALsynthTAK_{}'.format(split)
  __sets[name] = (lambda split=split: PALsynthTAK(split))          

# Set up PALsynthLL
for split in ['train', 'val', 'trainval']:
  name = 'PALsynthLL_{}'.format(split)
  __sets[name] = (lambda split=split: PALsynthLL(split))        
    
def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
