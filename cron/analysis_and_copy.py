import os
import sys
import datetime
import subprocess


prefectures = ["Japan", "Tokyo"]
#prefectures = ["Japan"]
#prefectures = ["Tokyo"]

populations = { 
   "Japan":  125754000, 
   "Tokyo":  13951636
}

def get_last_day_obs_by_toyokeizai():
    csv_fname = "toyokeizai/prefectures.csv"
    date=""
    with open(csv_fname) as f:
         a  = f.readlines()[-1].split(',')
         date = datetime.datetime(int(a[0]),int(a[1]),int(a[2]))
         ##### date = "{:d}-{:02d}-{:02d}".format(int(a[0]),int(a[1]), int(a[2]))
         print(date, "(Last OBS)")
    return date 

def get_last_day_obs_by_mhlw():
    csv_fname = "mhlw/confirmed_cases_cumulative_daily.csv"
    date=""
    with open(csv_fname) as f:
         a  = f.readlines()[-1].split(',')[0].split('/')
         date = datetime.datetime(int(a[0]),int(a[1]),int(a[2]))
         ##### date = "{:d}-{:02d}-{:02d}".format(int(a[0]),int(a[1]), int(a[2]))
         print(date, "(Last OBS)")
    return date 

def get_last_day_data_full_all_history(prefecture):
    date=""
    with open(prefecture + "/data_full_all_history.csv") as f:
         a  = f.readlines()[-1].split(',')[1].split('-')
         date = datetime.datetime(int(a[0]),int(a[1]),int(a[2]))
         print(date, prefecture, "(last_day_data_full_all_history)")
    return date 

def analysis(prefecture, population):
    print("analysis...", prefecture, population)
    subprocess.call(["python3", "city_data_shaping.py", prefecture, str(population)])
    subprocess.call(["python3", "gen_natural_run.py"])
    subprocess.call(["python3", "generate_obs.py"])
    subprocess.call(["python3", "ana_ini.py"])
    subprocess.call(["python3", "gen_ana_En.py"])

def copy_files(prefecture, date):

#============
# source
#============
#Tokyo/data_full.csv
#Tokyo/d_simu_pred_20210914_reshape.csv
#Tokyo/h_simu_pred_20210914_reshape.csv
#Tokyo/rt_conf_20210914.csv
#Tokyo/rt_cont_conf_reshape_20210914.csv

#============
# destination
#============
# covid_19/tokyo/data_full.csv
# covid_19/tokyo/d_simu_pred_09_11_reshape.csv
# covid_19/tokyo/h_simu_pred_09_11_reshape.csv
# covid_19/tokyo/rt_conf_09_11.csv
# covid_19/tokyo/rt_cont_conf_reshape_09_11.csv

    print("copy_files...", prefecture, date)
    src="{}".format(prefecture)  # Tokyo
    dst="covid_19/{}/{}_{:02}_{:02}".format(prefecture.lower(), date.year,date.month,date.day) # tokyo
    os.makedirs(dst, exist_ok=True)

    # update the timestamp for directory.
    from pathlib import Path
    Path(dst).touch()
    
    #print("src {} dst {}".format(src,dst))

    subprocess.call(["cp", src + "/data_full_all_history.csv", dst + "/data_full_all_history.csv"])
    subprocess.call(["cp", src + "/d_simu_pred_{}{:02}{:02}_reshape.csv".format(date.year,date.month,date.day), dst + "/d_simu_pred_{:02}_{:02}_reshape.csv".format(date.month,date.day)])
    subprocess.call(["cp", src + "/h_simu_pred_{}{:02}{:02}_reshape.csv".format(date.year,date.month,date.day), dst + "/h_simu_pred_{:02}_{:02}_reshape.csv".format(date.month,date.day)])
    subprocess.call(["cp", src + "/rt_conf_{}{:02}{:02}.csv".format(date.year,date.month,date.day), dst + "/rt_conf_{:02}_{:02}.csv".format(date.month,date.day)])
    subprocess.call(["cp", src + "/rt_cont_conf_reshape_{}{:02}{:02}.csv".format(date.year,date.month,date.day), dst + "/rt_cont_conf_reshape_{:02}_{:02}.csv".format(date.month,date.day)])

def update_timestamp():
     subprocess.call(["touch", "/home/covid19/covid_19/release/lasttime_gendata.txt"])


done_analysis = False

for prefecture in prefectures:
    print("prefecture", prefecture)
    if prefecture == "Japan":
       last_day_obs = get_last_day_obs_by_mhlw()
    else:
       last_day_obs = get_last_day_obs_by_toyokeizai()

    last_day_ana = get_last_day_data_full_all_history(prefecture)
    #print("last_day_obs ", last_day_obs)
    #print("last_day_ana ", last_day_ana)
    #print("type(last_day_obs) ", type(last_day_obs))
    #print("type(last_day_ana) ", type(last_day_ana))
    #sys.exit(1)

    num_days_to_run = (last_day_obs - last_day_ana).days
    if num_days_to_run==0:
       print("{} is up to date".format(prefecture))
    else:
       print("{} is not up to date. run {} days".format(prefecture, num_days_to_run))

    date_list = [last_day_ana + datetime.timedelta(days=i+1) for i in range(num_days_to_run)]
    #print("date_list", date_list)

    if not done_analysis and len(date_list) > 0:
       done_analysis = True

    for date in date_list:
        print("{} run for {}".format(prefecture, date))
        #sys.exit(1)
        analysis(prefecture, populations[prefecture])
        copy_files(prefecture, date)

    last_day_ana = get_last_day_data_full_all_history(prefecture)
    num_days_to_run = (last_day_obs - last_day_ana).days
    if num_days_to_run==0:
       print("[Check] {} is up to date".format(prefecture))
    else:
       print("[Check] {} is not up to date. remaining{} days. check the program.".format(prefecture, num_days_to_run))


if done_analysis:
   update_timestamp()
