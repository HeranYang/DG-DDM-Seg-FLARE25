#!/bin/bash
#heran yang

for(( id=1; id<10; id++))

do

  code_path=/home/hryang/Project/DMSeg_Project/code/Ours_v3_gradientImageCorr_modLoss_test/our_test.py
  config_path=/home/hryang/Project/DMSeg_Project/code/Ours_v3_gradientImageCorr_modLoss_test/config/our_test.json
  
  python ${code_path} -p val -c ${config_path}

done

