#!/usr/bin/env python
# coding: utf-8

# In[18]:


#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import requests
import io
import datetime as dt
import os
from class_state_vec import state_vector
import sys
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

country_name  =  "Tokyo"#sys.argv[1] #
sv = state_vector()
sv.setCountry(country_name)
sv.setPopulation(int(13951636))  #sys.argv[2]))#Kyo2563192)# #125360000)#)#J125360000)#9242724)#8815376)#5441276) #125754000)

try:
    os.mkdir(country_name)
    print("Directory " , country_name ,  " Created ") 
    first_time = True
    sv.setCondition(1) #full outbreak analysis
except FileExistsError:
    first_time = False
    sv.setCondition(2) #daily analysis
    print("Directory " , country_name ,  " already exists")


# In[19]:



#read csv
#Tsuzu-san made update on 17th Sep 2021
if country_name != 'Japan':
    #url = "https://toyokeizai.net/sp/visual/tko/covid19/csv/prefectures.csv"
    #csv_data = requests.get(url).content
    path = "toyokeizai/prefectures.csv"
    df = pd.read_csv(path, sep=",")
    df.head()

    #drop unnecessary columns and shape date 
    df_tokyo = df[df['prefectureNameE'] == country_name].reset_index() 
    df_tokyo['date'] = pd.to_datetime({'year': df_tokyo['year'], 'month': df_tokyo['month'], 'day': df_tokyo['date']})
    df_tokyo.drop(columns = ['index', 'year', 'month', 'prefectureNameJ', 'prefectureNameE'], inplace=True) 
    df_tokyo.head()

    #make our data
    df_data = pd.DataFrame({'date': df_tokyo['date'], 
                            'daily_confirmed': 0, 
                            'acc_confirmed': df_tokyo['testedPositive'], 
                            'daily_hos_add': 0, 
                            'acc_death': df_tokyo['deaths'], 
                            'acc_recovered': df_tokyo['discharged'], 
                            'serious': df_tokyo['serious'], 
                            'daily_death': 0, 
                            'effR': df_tokyo['effectiveReproductionNumber']})
    #Qiwen added the following two lines to replace the first several days NaN records by 0
    df_data['acc_death'].fillna(0, inplace = True)
    df_data['acc_recovered'].fillna(0, inplace = True)
    
    
    df_data['daily_confirmed'] = df_data['acc_confirmed'].diff()
    df_data['daily_hos_add'] = df_data['acc_confirmed'] - df_data['acc_death'] - df_data['acc_recovered']
    #Qiwen 2021-11-02 missing data
    #df_data['daily_hos_add'] = df_data['acc_confirmed'] - df_data['acc_death'] - (df_data['acc_recovered']+4512-447)
    
    df_data['daily_death'] = df_data['acc_death'].diff()
    
    #Qiwen modify 2021-11-05 Tokyo should always modifies the recovery data, no need to modify data for other cities
    if country_name == 'Tokyo':
        df_data['acc_recovered'] = df_data['acc_recovered'] -4512 + 447 #we keep the incorrect record to avoid the big jump, since Tokyo does not provide the exact release time.

        
    df_data.head()
    
    #fill nan
    df_data.fillna(0, inplace=True)
    df_data.head()
    
    #cut data from before 2020/3/6
    base_date = dt.datetime(2020, 3, 6)
    df_data = df_data[base_date <= df_data['date']].reset_index()
    df_data.drop(columns = 'index', inplace=True)

    #all value are int except date and effR
    for i in df_data.columns.values:
        if (i == 'date') or (i == 'effR') :
            pass
        else:
            df_data[i] = df_data[i].astype(int)
    df_data_with_makeup = df_data
else:
    name_lst = ["confirmed_cases_cumulative_daily", "requiring_inpatient_care_etc_daily", "deaths_cumulative_daily",
                "severe_cases_daily","effective_reproduction_number"]     
    url_lst = []
    for name in name_lst:
        if name != "effective_reproduction_number":
            #url = "https://toyokeizai.net/sp/visual/tko/covid19/csv/{}.csv".format(name)
            #url = "https://covid19.mhlw.go.jp/public/opendata/{}.csv".format(name)
            url = "mhlw/{}.csv".format(name)
        else:
            #url = "https://toyokeizai.net/sp/visual/tko/covid19/csv/effective_reproduction_number.csv"
            url = "toyokeizai/effective_reproduction_number.csv"
        url_lst.append(url)

    df_lst = []
    count =1
    base_date = dt.datetime(2020, 5, 9)
    for url in url_lst:
        #csv_data = requests.get(url).content
        #df = pd.read_csv(io.BytesIO(csv_data), sep=",")
        df = pd.read_csv(url, sep=",")
        '''
        if count<len(name_lst):
            for q in range(len(df)):
                df['Date'][q] = pd.to_datetime(df['Date'][q])
        df= df[base_date <= df['Date']].reset_index()
        df.drop(columns = 'index', inplace=True)
       '''
        if count == len(name_lst):
            df['日付'] = pd.to_datetime(df['日付'])
            df_ern = df[base_date<=df['日付']].reset_index()
            df_ern.drop(columns = 'index', inplace=True)
            break
        
        df_lst.append(df)
        count = count+1

    for i in range(len(df_lst)):
        #df_lst[i]['日付'] = pd.to_datetime(df_lst[i]['日付'])
        df_lst[i]['Date'] = pd.to_datetime(df_lst[i]['Date'])
        
    #2021-12-07 Qiwen   mhlw change the data format, qiwen modified the code correspondingly
    if country_name == 'Japan':
        df = pd.concat([df_lst[0]['Date'], df_lst[0]['ALL'], 
                        df_lst[1]['(ALL) Discharged from hospital or released from treatment'],
                        df_lst[2]['ALL'],
                        df_lst[3]['ALL']],axis = 1)
    else:
        df = pd.concat([df_lst[0]['Date'], df_lst[0][country_name], 
                        df_lst[1]['(%s) Discharged from hospital or released from treatment'%(country_name)],
                        df_lst[2][country_name],
                        df_lst[3][country_name]],axis = 1)
    df_data = pd.DataFrame({'date':np.zeros(len(df.iloc[:,[0]])), 
                            'daily_confirmed':np.zeros(len(df.iloc[:,[0]])),
                            'acc_confirmed':np.zeros(len(df.iloc[:,[0]])), 
                            'daily_hos_add':np.zeros(len(df.iloc[:,[0]])), 
                            'acc_death':np.zeros(len(df.iloc[:,[0]])), 
                            'acc_recovered':np.zeros(len(df.iloc[:,[0]])),
                            'serious':np.zeros(len(df.iloc[:,[0]])),
                            'daily_death':np.zeros(len(df.iloc[:,[0]])), 
                            'effR':np.zeros(len(df.iloc[:,[0]]))})
    df_data.iloc[:,[0,2,4,5,6]]=  df.iloc[:, [0,1,3,2,4]].copy()         
    #--#
    
    df_data.head()
    
    df_data['daily_confirmed'] = df_data['acc_confirmed'].diff()
    df_data['daily_hos_add'] = df_data['acc_confirmed'] - df_data['acc_death'] - df_data['acc_recovered']
    df_data['daily_death'] = df_data['acc_death'].diff()
    
    #20211108 Qiwen add for after 20211028 Japan
    df_data['acc_recovered'] = df_data['acc_recovered'] - 4512 + 447
    
    df_data.fillna(0, inplace=True)

    for i in df_data.columns.values:
        if (i == 'date') or (i == 'effR') :
            pass
        else:
            df_data[i] = df_data[i].astype(int)

    #df_data = df_data[50:].reset_index()
    #df_data.drop(columns = 'index', inplace=True)
    df_data.head()
    
    dd = df_data[0:64]
    fra = [df_data[0:64],df_data]
    results = pd.concat(fra,ignore_index = True, join='inner')
    df_data_with_makeup = results.reset_index()
    df_data_with_makeup.drop(columns = 'index', inplace=True)


# In[20]:


if first_time:
    df_data.to_csv('%s/data_full_all.csv'%(country_name))
    df_data.to_csv('%s/data_full_all_history.csv'%(country_name))
    #df_data[['date', 'daily_hos_add', 'acc_recovered','acc_death' ]].to_csv('%s/data_full.csv'%(country_name), header = None)
else:
    old_data = pd.read_csv ('%s/data_full_all_history.csv'%(country_name))
    #20220128
    #df_2days=df_data_with_makeup[len(old_data)-1:len(old_data)+1].reset_index()
    df_2days=df_data_with_makeup[len(old_data)-1:].reset_index()
    ##
    if len(df_2days)<2:
        print('no new data')
    else:
        df_2days.drop(columns = 'index', inplace=True)
        df_2days.to_csv('%s/data_full_all.csv'%(country_name), index=False)
        #Qiwen 2021-11-2 keep the old observations for the time being
        frames = [old_data, df_2days[-1:]]
        result = pd.concat(frames,ignore_index = True, join='inner')
        result['date']=pd.to_datetime(result.date)
        result.to_csv('%s/data_full_all_history.csv'%(country_name))
        #df_data[:len(old_data)+1].to_csv('%s/data_full_all_history.csv'%(country_name))
        #df_data[['date', 'daily_hos_add', 'acc_recovered','acc_death' ]][0:len(old_data)+1].to_csv('%s/data_full.csv'%(country_name), header = None)
    


# In[21]:


import numpy as np
df_data["date"] = pd.to_datetime(df_data["date"]).dt.strftime("%Y%m%d")
df_data_with_makeup["date"] = pd.to_datetime(df_data_with_makeup["date"]).dt.strftime("%Y%m%d")
if first_time:
    df_data_today = df_data_with_makeup
else:
    df_data_today = df_data_with_makeup[:len(old_data)+1]
bb = (df_data_today["date"][-2:])
bb = np.array(bb)
sv.setDate(bb)

sv.save('x_nature.pkl')


# In[ ]:




