#! /bin/bash/
python3 ../main.py \
--device 0 \
--mode eval \
--rgb \
--flow \
--make_output \
--reduce 1 \
--load_model \
--model_path ../checkpoint/$1/sbj_$2/best_epoch.pth \
--code_testing \
--use_flag \
--use_focal \


"""
--model_path 'checkpoint/best_epoch.pth' \
"""