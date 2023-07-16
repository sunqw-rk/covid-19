import numpy as np
import pandas as pd
import sys

import subprocess

address = ["hideyuki.sakamoto@riken.jp", "qiwen.sun@riken.jp"]
message = ""
obs_is_bad = False

def send_email(msg):
    global address
    cmd=f"echo \"{msg}\" | mail -s \"Alert: covid-19 observation\" -r covid19@ardbeg.r-ccs27.riken.jp"
    cmd+=" "
    cmd+= " ".join([ addr for addr in address])
    print("cmd", cmd)
    subprocess.run(cmd, shell=True)

def check_obs_Japan(df):
    global message
    global obs_is_bad
    # date: YYYY/MM/DD
    # positive: number of positive people (not accumulated)
    
    prefecture="Japan"
    date1 = df["date"].values[-2]
    date2 = df["date"].values[-1]
    positives = df["positive"].values
    positive1 = positives[-2]
    positive2 = positives[-1]
    value = np.abs(positive2 - positive1)/positive1
    threshold = 5 # stop updating if value is greather than threshold
    message += f"{date1}: prefecture: {prefecture} positive {positive1}\n"
    message += f"{date2}: prefecture: {prefecture} positive {positive2}\n"
    
    if (value > threshold):
       message += f"value {value:.3f} < threshold {threshold} -> NG. analyis for {date2} stopped\n\n"
       obs_is_bad = True
    else:
       message += f"value {value:.3f} < threshold {threshold} -> OK\n\n"


def check_obs_prefecture(df, prefecture):
    global message
    global obs_is_bad

    #prefecture = "Tokyo"
    df = df[df["prefectureNameE"] == prefecture]
    
    #print(df.columns)
    years=df["year"].values
    months=df["month"].values
    dates=df["date"].values
    
    year1 = years[-2]
    month1 = months[-2]
    date1 = dates[-2]
    
    year2 = years[-1]
    month2 = months[-1]
    date2 = dates[-1]
    
    # testedPositive: accumulated number

    # suggests by Qiwen:
    # use "testedPositive - discharged - deaths" instead of "testedPositive"
    
    testedPositives = df["testedPositive"].values
    discharged= df["discharged"].values
    deaths = df["deaths"].values
    yesterday  = np.abs(testedPositives[-2] - discharged[-2] - deaths[-2])
    today      = np.abs(testedPositives[-1] - discharged[-1] - deaths[-1])

    value = np.abs(today-yesterday)/yesterday
    threshold = 5 # stop updating if value is greather than threshold
    message += f"{year1}/{month1}/{date1}: prefecture: {prefecture} testedPositive-discharged-deaths {yesterday}\n"
    message += f"{year2}/{month2}/{date2}: prefecture: {prefecture} testedPositive-discharged-deaths {today}\n"
    
    if (value > threshold):
       message += f"value {value:.3f} < threshold {threshold} -> NG. analyis for {year2}/{month2}/{date2} stopped\n\n"
       obs_is_bad = True
    else:
       message += f"value {value:.3f} < threshold {threshold} -> OK\n\n"


df1 = pd.read_csv("toyokeizai/prefectures.csv")
#prefectures = ["Tokyo", "Osaka", "Hyogo"]
prefectures = ["Tokyo"]
for prefecture in prefectures:
    check_obs_prefecture(df1, prefecture)

df2 = pd.read_csv("toyokeizai/pcr_positive_daily.csv", header=None, skiprows=1, names=["date", "positive"])
check_obs_Japan(df2)

if obs_is_bad:
   print(message)
   send_email(message)
   sys.exit(1)

print(message)
print("obs is good.")
sys.exit(0)
