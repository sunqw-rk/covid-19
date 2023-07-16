import requests
import datetime
import sys

def getContent(url, csv_name):
 response = requests.get(url,headers=headers)
 response.encoding="utf_8_sig"
 f = open("toyokeizai/" + csv_name + ".csv", mode="w")
 f.write(response.text.replace('\r',''))
 f.close

# データを確認して、ファイルが更新されていれば、ダウンロードする。

# サーバCSVファイルの更新時刻
url='https://toyokeizai.net/sp/visual/tko/covid19/csv/prefectures.csv'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTL, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.38'}
response = requests.head(url)
last_modified1 = response.headers['last-Modified']
last_modified1 = datetime.datetime.strptime(last_modified1, "%a, %d %b %Y %H:%M:%S GMT")


# 追加でダウンロードするcsvファイル
name_lst = ["pcr_positive_daily", "cases_total", "recovery_total", "severe_daily", "death_total", "effective_reproduction_number"]
url_lst = []
for name in name_lst:
 url_lst.append("https://toyokeizai.net/sp/visual/tko/covid19/csv/{}.csv".format(name))


# ローカルCSVファイルの更新時刻
f = open('lasttime_file1.txt', mode='r')
local_last_modified1 = f.readline().rstrip()
local_last_modified1 = datetime.datetime.strptime(local_last_modified1, "%Y-%m-%d %H:%M:%S")
f.close


# 更新されているか判定
if last_modified1 == local_last_modified1:
 print(last_modified1)
 print(local_last_modified1)
 sys.exit("prefectures.csv is not updated")

# ダウンロードを実行する
getContent(url, "prefectures")
for (url,csv_name) in zip(url_lst,name_lst):
 getContent(url, csv_name)

# 前回時刻を更新する
f = open('lasttime_file1.txt', mode='w')
f.write(last_modified1.strftime("%Y-%m-%d %H:%M:%S"))
f.close
