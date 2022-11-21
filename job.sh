#!/bin/bash

#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=FAIL
#SBATCH -e /scratch/lu.xue/outputs/slurm-%j.err
#SBATCH -o /scratch/lu.xue/outputs/slurm-%j.out

tid=$SLURM_ARRAY_TASK_ID

# 有枣没枣打三杆
for ((i = 0; i < 2; i++)); do
	python /home/lu.xue/scratch/DTQN/run.py $@ &
	PIDS+=($!)
done

bs
# wait for all processes to finish, and store each process's exit code into array STATUS[].
for pid in ${PIDS[@]}; do
	echo "pid=${pid}"
	wait ${pid}
	STATUS+=($?)
done

# after all processed finish, check their exit codes in STATUS[].
i=0
for st in ${STATUS[@]}; do
	if [[ ${st} -ne 0 ]]; then
		echo "$i failed"
		exit 1
	else
		echo "$i finish"
	fi
	((i += 1))
done
