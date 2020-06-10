'''
Reads in articles from the New York Times API and saves
them to a cache.
'''

import time
import os
import datetime
from nytimesarticle import articleAPI
import pandas as pd
# News archive api
import requests

from dateutil.rrule import rrule, MONTHLY
NYT_KEY = open('nyt_key.txt').read().strip()
api = articleAPI(NYT_KEY)

def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    for i in articles['response']['docs']:
        if 'abstract' not in i.keys():
            continue
        if 'headline' not in i.keys():
            continue
        if 'news_desk' not in i.keys():
            continue
        if 'pub_date' not in i.keys():
            continue
        if 'snippet' not in i.keys():
            continue
        dic = {}
        dic['id'] = i['_id']
        if i.get('abstract', 'EMPTY') is not None:
            dic['abstract'] = i.get('abstract', 'EMPTY').encode("utf8")
        dic['headline'] = i['headline']['main'].encode("utf8")
        dic['desk'] = i.get('news_desk', 'EMPTY')
        dic['date'] = i['pub_date'][0:10] # cutting time of day.
        dic['time'] = i['pub_date'][11:19]
        dic['section'] = i.get('section_name', 'EMPTY')
        if i['snippet'] is not None:
            dic['snippet'] = i['snippet'].encode("utf8")
        dic['source'] = i.get('source', 'EMPTY')
        dic['type'] = i.get('type_of_material', 'EMPTY')
        dic['word_count'] = i.get('type_of_material', 0)
        news.append(dic)
    return pd.DataFrame(news)

def day_interval(days_back):
    today = datetime.datetime.today()
    that_day = today - datetime.timedelta(days=days_back)
    day_ago = that_day - datetime.timedelta(days=1)
    return (int(that_day.strftime('%Y%m%d')), int(day_ago.strftime('%Y%m%d')))

def bulk_look_up(start_year):
    # create a list of year, month, pairs for the data
    # from start dt to end date inclusive
    start_dt = datetime.date(start_year, 1, 1)
    end_dt = datetime.datetime.today()
    dates = [(dt.year, dt.month) for dt in rrule(MONTHLY, dtstart=start_dt, until=end_dt)]

    dfs = []
    for year, month in dates:
        url = (
            "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?&api-key={key}"
            .format(year=year, month=month, key=NYT_KEY)
        )
        r = requests.get(url)
        print(r.json())
        df = parse_articles(r.json())
        print('Got articles for {}/{}'.format(month, year))
        dfs.append(df)
        time.sleep(20)
    return pd.concat(dfs, ignore_index=True)

def download_or_cache(start_year):
    cache_path = 'nyt_data_from_{start_year}.pkl'.format(start_year=start_year)
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    df = bulk_look_up(start_year)
    df.to_pickle(cache_path)
    return df

if __name__ == '__main__':
    download_or_cache(2015)
