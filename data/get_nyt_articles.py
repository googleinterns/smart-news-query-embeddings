"""
Copyright 2020 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

'''
Reads in articles from the New York Times API and saves
them to a cache.
'''

import time
import os
import datetime
import argparse
# News archive api
from nytimesarticle import articleAPI
import pandas as pd
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
        if len(i['pub_date']) < 20:
            continue
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
    # Source of API data: https://developer.nytimes.com/docs/archive-product/1/overview
    start_dt = datetime.date(start_year, 1, 1)
    end_dt = datetime.datetime.today()
    dates = [(dt.year, dt.month) for dt in rrule(MONTHLY, dtstart=start_dt, until=end_dt)]
    wait = 20

    dfs = []
    for year, month in dates:
        found_df = False
        for i in range(20):
            try:
                url = (
                    "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?&api-key={key}"
                    .format(year=year, month=month, key=NYT_KEY)
                )
                r = requests.get(url)
                df = parse_articles(r.json())
                found_df = True
                break
            except:
                print(f'Error when getting articles, trying again in {wait} seconds...')
                continue
        if not found_df:
            continue
        print('Got {} articles for {}/{}'.format(df.shape[0], month, year))
        dfs.append(df)
        print(f'Waiting {wait} seconds for next request...')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', default=2015, type=int)

    args = parser.parse_args()
    download_or_cache(args.start_year)
