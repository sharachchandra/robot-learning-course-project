#!/bin/bash

# ==========obs_ablation==========

#  task: square
#    dataset type: ph
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_rnn_add_proprio.json

#  task: square
#    dataset type: ph
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_remove_rand.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_remove_rand.json

#  task: square
#    dataset type: mh
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_rnn_add_proprio.json

#  task: square
#    dataset type: mh
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_remove_rand.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_remove_rand.json

#  task: transport
#    dataset type: ph
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_rnn_add_proprio.json

#  task: transport
#    dataset type: ph
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_remove_rand.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_remove_rand.json

#  task: transport
#    dataset type: mh
#      hdf5 type: low_dim
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_rnn_add_proprio.json

#  task: transport
#    dataset type: mh
#      hdf5 type: image
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_remove_rand.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_add_eef_vel.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_add_proprio.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_remove_wrist.json
python /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/scripts/train.py --config /home/sb59938/Git/cs391r_course_project/robomimic/robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_remove_rand.json

