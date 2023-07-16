import requests
import datetime
import sys

# 厚生労働省オープンデータを確認して、必要な3ファイルが更新されていれば、ダウンロードする。

to_be_downloaded1 = True
to_be_downloaded2 = True
to_be_downloaded3 = True
to_be_downloaded4 = True

#####################
#  1.
#####################
# サーバCSVファイルの更新時刻
url='https://covid19.mhlw.go.jp/public/opendata/requiring_inpatient_care_etc_daily.csv'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTL, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.38'}
response = requests.head(url)
last_modified1 = response.headers['last-Modified']
last_modified1 = datetime.datetime.strptime(last_modified1, "%a, %d %b %Y %H:%M:%S GMT")

# ローカルファイルの更新時刻
f = open('mhlw_lasttime_file1.txt', mode='r')
local_last_modified1 = f.readline().rstrip()
local_last_modified1 = datetime.datetime.strptime(local_last_modified1, "%Y-%m-%d %H:%M:%S")
f.close

# 更新されているか判定
if last_modified1 == local_last_modified1:
 print(last_modified1)
 print(local_last_modified1)
 print("requiring_inpatient_care_etc_daily.csv is not updated")
 to_be_downloaded1 = False

#####################
#  2.
#####################
# サーバCSVファイルの更新時刻
url='https://covid19.mhlw.go.jp/public/opendata/deaths_cumulative_daily.csv'
response = requests.head(url)
last_modified2 = response.headers['last-Modified']
last_modified2 = datetime.datetime.strptime(last_modified2, "%a, %d %b %Y %H:%M:%S GMT")

# ローカルCVSファイルの更新時刻
f = open('mhlw_lasttime_file2.txt', mode='r')
local_last_modified2 = f.readline().rstrip()
local_last_modified2 = datetime.datetime.strptime(local_last_modified2, "%Y-%m-%d %H:%M:%S")
f.close
 
# 更新されているか判定
if last_modified2 == local_last_modified2:
 print(last_modified2)
 print(local_last_modified2)
 print('deaths_cumulative_daily.csv is not updated')
 to_be_downloaded2 = False

#####################
#  3.
#####################
# サーバCSVファイルの更新時刻
url='https://covid19.mhlw.go.jp/public/opendata/confirmed_cases_cumulative_daily.csv'
response = requests.head(url)
last_modified3 = response.headers['last-Modified']
last_modified3 = datetime.datetime.strptime(last_modified3, "%a, %d %b %Y %H:%M:%S GMT")

# ローカルCVSファイルの更新時刻
f = open('mhlw_lasttime_file3.txt', mode='r')
local_last_modified3 = f.readline().rstrip()
local_last_modified3 = datetime.datetime.strptime(local_last_modified3, "%Y-%m-%d %H:%M:%S")
f.close
 
# 更新されているか判定
if last_modified3 == local_last_modified3:
 print(last_modified3)
 print(local_last_modified3)
 print('confirmed_cases_cumulative_daily.csv is not updated')
 to_be_downloaded3 = False

#####################
#  4.
#####################
# サーバCSVファイルの更新時刻
url='https://covid19.mhlw.go.jp/public/opendata/severe_cases_daily.csv'
response = requests.head(url)
last_modified4 = response.headers['last-Modified']
last_modified4 = datetime.datetime.strptime(last_modified4, "%a, %d %b %Y %H:%M:%S GMT")

# ローカルCVSファイルの更新時刻
f = open('mhlw_lasttime_file4.txt', mode='r')
local_last_modified4 = f.readline().rstrip()
local_last_modified4 = datetime.datetime.strptime(local_last_modified4, "%Y-%m-%d %H:%M:%S")
f.close
 
# 更新されているか判定
if last_modified4 == local_last_modified4:
 print(last_modified4)
 print(local_last_modified4)
 print('severe_cases_daily is not updated')
 to_be_downloaded4 = False



#####################
#  1.
#####################
if to_be_downloaded1:
    # ダウンロードを実行する
    url='https://covid19.mhlw.go.jp/public/opendata/requiring_inpatient_care_etc_daily.csv'
    response = requests.get(url,headers=headers)
    response.encoding="utf_8_sig"
    f = open('mhlw/requiring_inpatient_care_etc_daily.csv', mode="w")
    f.write(response.text.replace('\r',''))
    f.close
    
    # 前回時刻を更新する
    f = open('mhlw_lasttime_file1.txt', mode='w')
    f.write(last_modified1.strftime("%Y-%m-%d %H:%M:%S"))
    f.close

#####################
#  2.
#####################
if to_be_downloaded2:
    # ダウンロードを実行する
    url='https://covid19.mhlw.go.jp/public/opendata/deaths_cumulative_daily.csv'
    response = requests.get(url,headers=headers)
    response.encoding="utf_8_sig"
    f = open('mhlw/deaths_cumulative_daily.csv', mode="w")
    f.write(response.text.replace('\r',''))
    f.close
    
    # 前回時刻を更新する
    f = open('mhlw_lasttime_file2.txt', mode='w')
    f.write(last_modified2.strftime("%Y-%m-%d %H:%M:%S"))
    f.close

#####################
#  3.
#####################
if to_be_downloaded3:
   # ダウンロードを実行する
   url='https://covid19.mhlw.go.jp/public/opendata/confirmed_cases_cumulative_daily.csv'
   response = requests.get(url,headers=headers)
   response.encoding="utf_8_sig"
   f = open('mhlw/confirmed_cases_cumulative_daily.csv', mode="w")
   f.write(response.text.replace('\r',''))
   f.close
   
   # 前回時刻を更新する
   f = open('mhlw_lasttime_file3.txt', mode='w')
   f.write(last_modified3.strftime("%Y-%m-%d %H:%M:%S"))
   f.close

#####################
#  4.
#####################
if to_be_downloaded4:
   # ダウンロードを実行する
   url='https://covid19.mhlw.go.jp/public/opendata/severe_cases_daily.csv'
   response = requests.get(url,headers=headers)
   response.encoding="utf_8_sig"
   f = open('mhlw/severe_cases_daily.csv', mode="w")
   f.write(response.text.replace('\r',''))
   f.close
   
   # 前回時刻を更新する
   f = open('mhlw_lasttime_file4.txt', mode='w')
   f.write(last_modified4.strftime("%Y-%m-%d %H:%M:%S"))
   f.close
