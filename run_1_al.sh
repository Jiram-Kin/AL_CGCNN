#!/bin/bash

python AL_dft/al_iteration_kcenter.py --explore 0 \
--exploit 10 \
--label_path AL_dft/datasets/final_mxene_os/iter_file/Iter7_list.csv \
--iteration 8 \
--model_path 'AL_dft/models/model_final/' \
--result_path 'AL_dft/result/mxene_final/' \
--suffix '' \
--seed 0 \
--transfer true