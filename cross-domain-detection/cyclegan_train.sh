#!/bin/bash
# //Pure CycleGAN
python train_model.py --root datasets/TAKsynthHUA --subset train --result results/TAK_HUA_DT --eval_root datasets/TAKsynthHUA --max_iter 20000; 
python train_model.py --root datasets/TAKsynthLL --subset train --result results/TAK_LL_DT --eval_root datasets/TAKsynthLL --max_iter 20000; 
python train_model.py --root datasets/TAKsynthPAL --subset train --result results/TAK_PAL_DT --eval_root datasets/TAKsynthPAL --max_iter 20000; 
python train_model.py --root datasets/TAKsynthRAN --subset train --result results/TAK_RAN_DT --eval_root datasets/TAKsynthRAN --max_iter 20000; 
python train_model.py --root datasets/TAKsynthPAL2021 --subset train --result results/TAK_PAL2021_DT --eval_root datasets/TAKsynthPAL2021 --max_iter 20000; 

python train_model.py --root datasets/PAL2021synthHUA --subset train --result results/PAL2021_HUA_DT --eval_root datasets/PAL2021synthHUA --max_iter 20000; 
python train_model.py --root datasets/PAL2021synthLL --subset train --result results/PAL2021_LL_DT --eval_root datasets/PAL2021synthLL --max_iter 20000; 
python train_model.py --root datasets/PAL2021synthPAL --subset train --result results/PAL2021_PAL_DT --eval_root datasets/PAL2021synthPAL --max_iter 20000; 
python train_model.py --root datasets/PAL2021synthRAN --subset train --result results/PAL2021_RAN_DT --eval_root datasets/PAL2021synthRAN --max_iter 20000; 
python train_model.py --root datasets/PAL2021synthTAK --subset train --result results/PAL2021_TAK_DT --eval_root datasets/PAL2021synthTAK --max_iter 20000; 

# //Pure CycleGAN PL
python pseudo_label.py --root datasets/HUA --load results/TAK_HUA_DT/model_iter_20000 --result datasets/TAK_HUA_PL/; 
python pseudo_label.py --root datasets/LL --load results/TAK_LL_DT/model_iter_20000 --result datasets/TAK_LL_PL/; 
python pseudo_label.py --root datasets/PAL --load results/TAK_PAL_DT/model_iter_20000 --result datasets/TAK_PAL_PL/; 
python pseudo_label.py --root datasets/RAN --load results/TAK_RAN_DT/model_iter_20000 --result datasets/TAK_RAN_PL/; 
python pseudo_label.py --root datasets/PAL2021 --load results/TAK_PAL2021_DT/model_iter_20000 --result datasets/TAK_PAL2021_PL/

python pseudo_label.py --root datasets/HUA --load results/PAL2021_HUA_DT/model_iter_20000 --result datasets/PAL2021_HUA_PL/; 
python pseudo_label.py --root datasets/LL --load results/PAL2021_LL_DT/model_iter_20000 --result datasets/PAL2021_LL_PL/; 
python pseudo_label.py --root datasets/PAL --load results/PAL2021_PAL_DT/model_iter_20000 --result datasets/PAL2021_PAL_PL/; 
python pseudo_label.py --root datasets/RAN --load results/PAL2021_RAN_DT/model_iter_20000 --result datasets/PAL2021_RAN_PL/; 
python pseudo_label.py --root datasets/TAK --load results/PAL2021_TAK_DT/model_iter_20000 --result datasets/PAL2021_TAK_PL/;

# // Pure CycleGAN PL train
python train_model.py --root datasets/TAK_HUA_PL --subset train --result results/TAK_HUA_CG_PL --eval_root datasets/TAK_HUA_PL --load results/TAK_HUA_DT/model_iter_20000; 
python train_model.py --root datasets/TAK_LL_PL --subset train --result results/TAK_LL_CG_PL --eval_root datasets/TAK_LL_PL --load results/TAK_LL_DT/model_iter_20000;
python train_model.py --root datasets/TAK_PAL_PL --subset train --result results/TAK_PAL_CG_PL --eval_root datasets/TAK_PAL_PL --load results/TAK_PAL_DT/model_iter_20000; 
python train_model.py --root datasets/TAK_RAN_PL --subset train --result results/TAK_RAN_CG_PL --eval_root datasets/TAK_RAN_PL --load results/TAK_RAN_DT/model_iter_20000; 
python train_model.py --root datasets/TAK_PAL2021_PL --subset train --result results/TAK_PAL2021_CG_PL --eval_root datasets/TAK_PAL2021_PL --load results/TAK_PAL2021_DT/model_iter_20000; 

python train_model.py --root datasets/PAL2021_HUA_PL --subset train --result results/PAL2021_HUA_CG_PL --eval_root datasets/PAL2021_HUA_PL  --load results/PAL2021_HUA_DT/model_iter_20000; 
python train_model.py --root datasets/PAL2021_LL_PL --subset train --result results/PAL2021_LL_CG_PL --eval_root datasets/PAL2021_LL_PL  --load results/PAL2021_LL_DT/model_iter_20000;
python train_model.py --root datasets/PAL2021_PAL_PL --subset train --result results/PAL2021_PAL_CG_PL --eval_root datasets/PAL2021_PAL_PL  --load results/PAL2021_PAL_DT/model_iter_20000; 
python train_model.py --root datasets/PAL2021_RAN_PL --subset train --result results/PAL2021_RAN_CG_PL --eval_root datasets/PAL2021_RAN_PL   --load results/PAL2021_RAN_DT/model_iter_20000;
python train_model.py --root datasets/PAL2021_TAK_PL --subset train --result results/PAL2021_TAK_CG_PL --eval_root datasets/PAL2021_TAK_PL  --load results/PAL2021_TAK_DT/model_iter_20000; 