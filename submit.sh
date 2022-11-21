#!/bin/bash
# for epsilon in "0.1" "0.2" "0.05", "0.01"; do
# for i in {1..5}; do
	cd /home/lu.xue/scratch/DTQN
	# sbatch job.sh --epsilon $epsilon --learning_rate $learing_rate
	sbatch job.sh --env gv_memory.7x7.yaml --inembed 128 --device cpu
# done
# done;
#
