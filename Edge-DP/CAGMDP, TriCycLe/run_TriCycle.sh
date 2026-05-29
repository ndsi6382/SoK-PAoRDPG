#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9; do
    mkdir GitHub_TriCycle_combined_experiment
    (java -jar NewDPC.jar GitHub 1 2; ~/Workspace/Resources/Scripts/notify.py TriCycLe "GitHub trial $i complete.")
    mkdir TriCycLe_result/txts_GitHub_trial_$i
    mv GitHub_TriCycle_combined_experiment/* TriCycLe_result/txts_GitHub_trial_$i
    rm -rf GitHub_TriCycle_combined_experiment
done
