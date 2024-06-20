#!/bin/bash

visualize=true
if [ $visualize = "true" ]; then
    render_mode=human
else
    render_mode=dummy
fi

work_dir=$(dirname "$(readlink -f "$0")")

ln -s $work_dir/../../robot_sim/config/env $work_dir/config/task
export PYTHONPATH=$work_dir:$(realpath $work_dir/../ManiSkill):$(realpath $work_dir/../..):$PYTHONPATH
ckpt_dir=$work_dir/logs/table_arranged_grasp/2024.04.26-17.20.38
model_name=9-checkpoint.pth
python mt_evaluator.py \
    ckpt_dir=$ckpt_dir \
    model_name=$model_name \
    task.env.render_mode=$render_mode \
    task.env.obj_place_mode=tight \
    task.env.num_objects=5 \
    task.env.tight_factor=1.1 \
    test_num=100
