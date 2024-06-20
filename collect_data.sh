#!/bin/bash

task=$1
total_num_episodes=${2:-1000}

# threshold of maximum memory usage percentage
memory_threshold=80
export PYTHONPATH=$(pwd)/3rd_party/ManiSkill:$(pwd)

if [ $task = "pick" ]; then
    script_path=$(pwd)/robot_sim/tasks/pick_task_generation.py
    output_path=$(pwd)/outputs/scattered_pick
elif [ $task = "select" ]; then
    script_path=$(pwd)/robot_sim/tasks/select_task_generation.py
    output_path=$(pwd)/outputs/orderly_select
elif [ $task = "search" ]; then
    script_path=$(pwd)/robot_sim/tasks/search_task_generation.py
    output_path=$(pwd)/outputs/stacked_search
else
    echo "The task must be one of [pick, select, search]."
    exit 1
fi

episode_save_path=${output_path}/collected_data

function run_script() {
    if [ -d ${episode_save_path} ]; then
        num_episodes_remaining=$((total_num_episodes-$(ls ${episode_save_path} | wc -l)))
    else
        num_episodes_remaining=${total_num_episodes}
    fi
    python ${script_path} \
        hydra.run.dir=${output_path} \
        num_episodes=${num_episodes_remaining} \
        env.render_mode=dummy &
}

function check_memory_usage() {
    memory_usage=$(free | awk 'NR==2{printf "%.2f ", $3*100/$2 }')
    if (( $(echo "${memory_usage} > ${memory_threshold}" | bc -l) )); then
        echo "Memory usage is above the threshold. Restarting the program..."
        restart_program
    fi
}

function restart_program() {
    pkill -f ${script_path}
    sleep 5
    run_script
}

function kill_thread() {
    pkill -f ${script_path}
    exit 0
}

trap kill_thread INT

run_script

while true; do
    check_memory_usage
    sleep 5
    if ! pgrep -f ${script_path} > /dev/null; then
        if [ $(ls ${episode_save_path} | wc -l) -lt ${total_num_episodes} ]; then
            echo "Program has exited unexpectedly. Restarting the program..."
            run_script
        else
            echo "Program has exited. Exiting the loop."
            break
        fi
    fi
done
