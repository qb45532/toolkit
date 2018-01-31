# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import numba
import os
import datetime
from sklearn.cluster import KMeans

def get_stockdata(db,workdir,startdate = 20100101):
    '''
    提取wind股票日数据
    '''
    datafile = os.path.join(workdir,'stockdata.csv')
    stockdata = None
    if os.path.exists(datafile):
        stockdata = pd.read_csv(datafile)
        endday = pd.to_datetime(stockdata['DataDate'].max(),format = '%Y%m%d')
        startdate = endday + datetime.timedelta(days = 1)
        startdate = startdate.year*10000+startdate.month*100+startdate.day
    newdata = pd.read_sql('''SELECT A.`S_INFO_WINDCODE`, A.`TRADE_DT`, A.`S_DQ_ADJOPEN`, A.`S_DQ_ADJHIGH`,
                            A.`S_DQ_ADJLOW`, A.`S_DQ_ADJCLOSE`, A.`S_DQ_VOLUME`,A.`S_DQ_AMOUNT`,A.`S_DQ_ADJFACTOR` 
                            FROM wind.ASHAREEODPRICES A WHERE TRADE_DT >= %d;'''%startdate,con = db)
    newdata.columns = ['ukey','DataDate','open','high','low','close','volume','amount','adjfactor']
    newdata['DataDate'] = newdata['DataDate'].astype(np.int)
    newdata = newdata[~newdata['ukey'].str.startswith('T')]
    newdata['ukey'] = map(lambda x:int('10'+x.split('.')[0]) if x.startswith('6') \
              else int('11'+x.split('.')[0]),newdata['ukey'])
    if stockdata is not None:
        stockdata = pd.concat([stockdata,newdata])
    else:
        stockdata = newdata
    stockdata = stockdata.sort_values(['ukey','DataDate'])
    stockdata.to_csv(datafile,index = None)
    return stockdata

def get_index_data(db,workdir,startdate = 20100101):
    '''
    获取wind指数日数据
    '''
    datafile = os.path.join(workdir,'indexdata.csv')
    indexdata = None
    if os.path.exists(datafile):
        indexdata = pd.read_csv(datafile)
        enddate = pd.to_datetime(indexdata['DataDate'].max(),format='%Y%m%d')
        startdate = enddate + datetime.timedelta(days=1)
        startdate = startdate.year*10000+startdate.month*100+startdate.day
    newdata = pd.read_sql('''SELECT A.`S_INFO_WINDCODE`, A.`TRADE_DT`, A.`S_DQ_OPEN`,
A.`S_DQ_HIGH`, A.`S_DQ_LOW`, A.`S_DQ_CLOSE`, A.`S_DQ_PCTCHANGE`, A.`S_DQ_VOLUME`, A.`S_DQ_AMOUNT`
FROM wind.AINDEXEODPRICES A WHERE TRADE_DT>=%d;'''%startdate,con=db)
    newdata.columns = ['code','DataDate','open','high','low','close','pctchange','volume','amount']
    newdata['DataDate'] = newdata['DataDate'].astype(np.int)
    if indexdata is not None:
        indexdata = pd.concat([indexdata,newdata])
    else:
        indexdata = newdata
    indexdata = indexdata.sort_values(['code','DataDate'])
    indexdata.to_csv(datafile,index=None)
    return indexdata

def get_st_data(db):
    '''
    获取wind的st股
    '''
    today = datetime.datetime.now().date()
    today = today.year*10000 + today.month*100 + today.day
    
    st_data = pd.read_sql('''SELECT A.`S_INFO_WINDCODE`, A.`S_TYPE_ST`,A.`ENTRY_DT`,A.`REMOVE_DT`
                            FROM wind.ASHAREST A''',con = db)
    st_data.iloc[np.where(st_data['REMOVE_DT'].isnull())[0],3] = 0
    st_data['ENTRY_DT'] = st_data['ENTRY_DT'].astype(np.int)
    st_data['REMOVE_DT'] = st_data['REMOVE_DT'].astype(np.int)
    st_data.columns = ['ukey','stype','edate','rdate']
    st_data.loc[st_data['rdate']==0,'rdate'] = today
    st_data = st_data[~st_data['ukey'].str.startswith('T')]
    st_data['ukey'] = map(lambda x:int('10'+x.split('.')[0]) if x.startswith('6') \
              else int('11'+x.split('.')[0]),st_data['ukey'])
    return st_data


def weighted_corr(x,y,w):
    '''
    加权相关系数
    '''
    sumw = np.sum(w)
    wxm = np.sum(x*w)/sumw
    wym = np.sum(y*w)/sumw
    wx = x-wxm
    wy = y-wym
    covxyw = np.sum(w*wx*wy)/sumw
    covxxw = np.sum(w*wx*wx)/sumw
    covyyw = np.sum(w*wy*wy)/sumw
    return covxyw/np.sqrt(covxxw*covyyw)

def remove_st_stock(stockdata,st_data):
    '''
    剔除st股
    '''
    g = stockdata.groupby('ukey')
    n = len(st_data)
    drop_idx = []
    for i in xrange(n):
        ukey = st_data.iloc[i,0]
        if not g.groups.has_key(ukey):
            continue
        tmp = g.get_group(ukey)
        edate = st_data.iloc[i,2]
        rdate = st_data.iloc[i,3]
        idx = np.where((tmp['DataDate'] >= edate) & (tmp['DataDate'] <= rdate))[0]
        if len(idx) > 0:
            drop_idx += tmp.index[idx].tolist()
    #print drop_idx
    return stockdata.drop(drop_idx,axis = 0)

def remove_open_limit(stockdata):
    '''
    剔除开盘涨停卖不到的股票
    '''
    g = stockdata.groupby('ukey')
    stockdata['preclose'] = g['close'].shift(1)
    stockdata['is_open_uplimit'] = np.concatenate(g.apply(lambda df:is_open_uplimit(df['open'].values,\
             df['high'].values,df['close'].values,df['preclose'].values)).values)
    stockdata['is_open_uplimit'] = stockdata['is_open_uplimit'].astype(np.int)
    return stockdata[stockdata['is_open_uplimit'] == 0]

@numba.jit
def is_open_uplimit(open,high,close,preclose):
    '''
    非st股开盘是否涨停
    '''
    n = len(open)
    res = np.zeros(n)
    for i in xrange(n):
        if np.isnan(preclose[i]):
            continue
        if open[i] == high[i] and open[i]/preclose[i] >= 1.099 and close[i] == high[i]:
            res[i] = 1
    return res


def factor_cluster(factor,factor_name,target_name,n_clusters = 10,method = 'quantile'):
    '''
    :param factor: pd.DataFrame
    :param n_clusters:对因子分组数
    :param method: quantile 或者kmeans
    :return:
    '''
    tmp = factor.copy()
    cluster = None
    if method == 'kmeans':
        kmmodel = KMeans(n_clusters=n_clusters, init='k-means++', random_state=612)
        cluster = kmmodel.fit_predict(factor.reshape((-1, 1)))
        tmp['cluster'] = cluster
        target_mean = tmp.groupby('cluster')[target_name].mean().reset_index()
        target_count = tmp.groupby('cluster')[target_name].count().values
        if target_count.shape[1] > 1:
            target_mean['count'] = target_count[:,0]
        else:
            target_mean['count'] = target_count
        ncol = len(target_mean.columns)
        tmp_columns = target_mean.columns[1:ncol-1]
        target_mean.columns = [u'类别']+['%s_mean'%x for x in tmp_columns]+[u'次数']
        print(target_mean)
    if method == 'quantile':
        n = len(factor)
        q = [i*1.0/n_clusters for i in xrange(0,n_clusters+1)]
        fq = factor[factor_name].quantile(q).values
        cluster = np.zeros(n)*np.nan
        factor_values = factor[factor_name].values
        for i in xrange(1,len(q)):
            dl = fq[i-1]
            ul = fq[i]
            if i<n_clusters:
                idx = np.where((factor_values>=dl) & (factor_values<ul))[0]
            else:
                idx = np.where((factor_values>=dl) & (factor_values<=ul))[0]
            cluster[idx] = int(q[i]*n_clusters)
        tmp['cluster'] = cluster
        target_mean = tmp.groupby('cluster')[target_name].mean().reset_index()
        target_mean['cluster'] = target_mean['cluster'].astype(np.int)
        target_count = tmp.groupby('cluster')[target_name].count().values

        if target_count.shape[1] > 1:
            target_mean['count'] = target_count[:, 0]
        else:
            target_mean['count'] = target_count
        ncol = len(target_mean.columns)
        tmp_columns = target_mean.columns[1:ncol - 1]
        target_mean['thr_quantile'] = fq[target_mean['cluster'].values.astype(np.int)]
        target_mean['cluster'] = map(lambda x: 'Q%d/%d' % (x, n_clusters), \
                                     target_mean['cluster'].values)
        target_mean = target_mean[[target_mean.columns[0],'thr_quantile'] + tmp_columns.tolist() + ['count']]
        target_mean.columns = [u'分位数',u'%s_分位数数值'%factor_name]+['%s_mean'%x for x in tmp_columns]+[u'次数']
        print(target_mean)
    return {'target_mean':target_mean,'cluster':cluster}
