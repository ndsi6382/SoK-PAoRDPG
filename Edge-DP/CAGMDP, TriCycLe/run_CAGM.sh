#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9; do
    mkdir GitHub_combined_experiment
    #if [ "$i" -gt 0 ]; then
    #    cp CAGMDP_result/txts_GitHub_trial_0/*partition* GitHub_combined_experiment/
    #fi
    (java -jar NewDPC.jar GitHub 1 1; ~/Workspace/Resources/Scripts/notify.py CAGM "GitHub trial $i done.")
    mkdir CAGMDP_result/txts_GitHub_trial_$i
    mv GitHub_combined_experiment/* CAGMDP_result/txts_GitHub_trial_$i
    rm -rf GitHub_combined_experiment
done
