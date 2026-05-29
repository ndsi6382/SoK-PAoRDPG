#!/bin/bash

for i in 0; do # 1 2 3 4 5 6 7 8 9; do
    for e in 0.5; do # 0.75 1 1.5 2 3 4.5 6.5 9 12 16 20; do
        python3 PrivDPR.py --run_id "$i" --epsilon "$e" --dataset "Facebook" --data_path "../../Data/Facebook/edge_list.txt"
    done
done
