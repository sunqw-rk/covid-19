#!/bin/bash
cd /home/covid19/cron

source /home/covid19/.bashrc
conda deactivate
conda activate covid19

if [ -s lasttime_file1.txt ]; then
:
else
echo '2021-09-01 00:00:00' > lasttime_file1.txt 
fi

python download_csv.py
python mhlw_download_csv.py

# If observation is strange, then send alert email, then commit.
python check_obs.py && python analysis_and_copy.py

if [ -z "$(git status --porcelain)" ]; then
  # Working directory clean
  :
else
   # Uncommitted changes
   git add . 
   git commit -m "save $(date +"%Y/%m/%d %H:%M")"
   git tag -d $(date +"%Y%m%d_%H%M")
   git tag -a $(date +"%Y%m%d_%H%M") -m ""
fi

