#!/bin/bash

# ==========hyper_ablation==========

#  task: square
#    dataset type: ph
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/low_dim/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/low_dim/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/low_dim/bc_rnn_change_mlp.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/low_dim/bc_rnn_change_rnnd_low_dim.json

#  task: square
#    dataset type: ph
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/image/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/image/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/image/bc_rnn_change_conv.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/ph/image/bc_rnn_change_rnnd_image.json

#  task: square
#    dataset type: mh
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/low_dim/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/low_dim/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/low_dim/bc_rnn_change_mlp.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/low_dim/bc_rnn_change_rnnd_low_dim.json

#  task: square
#    dataset type: mh
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/image/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/image/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/image/bc_rnn_change_conv.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/square/mh/image/bc_rnn_change_rnnd_image.json

#  task: transport
#    dataset type: ph
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/low_dim/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/low_dim/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/low_dim/bc_rnn_change_mlp.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/low_dim/bc_rnn_change_rnnd_low_dim.json

#  task: transport
#    dataset type: ph
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/image/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/image/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/image/bc_rnn_change_conv.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/ph/image/bc_rnn_change_rnnd_image.json

#  task: transport
#    dataset type: mh
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/low_dim/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/low_dim/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/low_dim/bc_rnn_change_mlp.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/low_dim/bc_rnn_change_rnnd_low_dim.json

#  task: transport
#    dataset type: mh
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/image/bc_rnn_change_lr.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/image/bc_rnn_change_gmm.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/image/bc_rnn_change_conv.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/hyper_ablation/transport/mh/image/bc_rnn_change_rnnd_image.json

