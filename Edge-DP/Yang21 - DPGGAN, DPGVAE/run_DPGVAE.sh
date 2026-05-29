#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 #$(nproc --all)
export MKL_NUM_THREADS=16 #$(nproc --all)
export OPENBLAS_NUM_THREADS=16 #$(nproc --all)
export NUMEXPR_NUM_THREADS=16 #$(nproc --all)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in 5; do
    for e in 16 20; do
        nice -n 10 python3 src/main.py --model_name DPGVAE --dataset ../../../Data/Brightkite/nx_adj.pkl --seed $i --epsilon $e
    done
done

for i in 6 7 8; do
    for e in 0.5 0.75 1 1.5 2 3 4.5 6.5 9 12 16 20; do
        nice -n 10 python3 src/main.py --model_name DPGVAE --dataset ../../../Data/Brightkite/nx_adj.pkl --seed $i --epsilon $e
    done
done

for i in 9; do
    for e in 0.5 0.75 1 1.5 2; do
        nice -n 10 python3 src/main.py --model_name DPGVAE --dataset ../../../Data/Brightkite/nx_adj.pkl --seed $i --epsilon $e
    done
done

~/scripts/notify.py DPGVAE "Completed."
