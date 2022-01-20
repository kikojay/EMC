#!/bin/bash
   # Script to reproduce results

 Foldername="newtest001"
 mkdir out_logs/${Foldername} &> /dev/null
# mkdir /home/zll/projects/qplex/data/${Foldername} &> /dev/null
 #task= {"pred_prey_punish", "gridworld_reversed","corridor" "3s5z_vs_3s6z" "5s10z" "7s7z",...}
 # algos={"EMC_sc2","EMC_toygame","vdn",...}
 declare -a tasks=( "pred_prey_punish" )
 declare -a algos=( "EMC_toygame" )
 declare -a seeds=("1" "2" "3")

 n=4
 gpunum=8
 for task in "${tasks[@]}"
 do
 for algo in "${algos[@]}"
 do
 for seed in "${seeds[@]}"
 do

 if [ ${task} == 'gridworld_reversed' ]; then
 OMP_NUM_THREADS=16 KMP_AFFINITY="compact,granularity\=fine" CUDA_VISIBLE_DEVICES=${n} nohup python3 main.py \
 --config=${algo} --env-config=gridworld_reversed with env_args.map_name=reversed\
 >& out_logs/${Foldername}/${task}_${algo}_${seed}.txt &

 elif [ ${task} == 'pred_prey_punish' ]; then
  OMP_NUM_THREADS=16 KMP_AFFINITY="compact,granularity\=fine" CUDA_VISIBLE_DEVICES=${n} nohup python3 main.py \
 --config=${algo} --env-config=pred_prey_punish with env_args.map_name=origin\
 >& out_logs/${Foldername}/${task}_${algo}_${seed}.txt &
 else
 OMP_NUM_THREADS=16 KMP_AFFINITY="compact,granularity\=fine" CUDA_VISIBLE_DEVICES=${n} nohup python3 main.py \
 --config=${algo} --env-config=sc2 with env_args.map_name=${task} \
 env_args.seed=${seed} \
 >& out_logs/${Foldername}/'sc2_'${task}_${algo}_${seed}.txt &
 fi
 echo "task: ${task}, algo: ${algo}, seed: ${seed}, GPU: $n"
 n=$[($n+1) % ${gpunum}]
 sleep 20
 done
 done
 done
