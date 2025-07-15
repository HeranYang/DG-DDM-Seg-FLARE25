#!/bin/bash

set -e  # 遇到错误立即退出

echo "📂 Step 0: Running preprocess_test.py"
python preprocess_validation.py

echo "📂 Step 1: Entering SLAug and running test.py"
cd SLAug || { echo "SLAug directory not found!"; exit 1; }
python test.py -r logs/2025-06-10T15-50-21_seed23_2025-06-10T01-40-48-project -s ../dataset
cd ..

echo "📂 Step 2: Entering our_method and running our_test_mp.py"
cd our_method || { echo "our_method directory not found!"; exit 1; }
python our_test_mp.py -p val -c config/our_test_mp.json
cd ..

echo "📂 Step 3: Entering eval and running 3Ddice_comp.py"
cd eval || { echo "eval directory not found!"; exit 1; }
python 3Ddice_comp.py

echo "✅ All steps completed successfully."



