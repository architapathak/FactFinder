# -*- coding: utf-8 -*-
"""
"""

from newspaper import Article
import requests
from bs4 import BeautifulSoup as BS
from newspaper.article import ArticleException, ArticleDownloadState
from time import sleep
import pandas as pd
from dateutil import parser, relativedelta

def crawl_snopes(id_, url, article_date):
    try:
        page = requests.get(url)
        soup = BS(page.content, 'html.parser')
        res = soup.find('div', class_='content')
        if not res:
            res = soup.find('div', class_='single-body card card-body rich-text')

        #print(res)
        paras = res.find_all('p')
        ev = ''
        for para in paras:
            # Remove links
            ev = ev + para.text.strip() + ' '
        
        date_val = soup.find('li', {"class": "font-weight-bold text-muted"})
        ev_date = date_val.text.split('\t')[4]
        ev_date = parser.parse(ev_date).date()
        
        if ev_date:
          delta = relativedelta.relativedelta(ev_date, article_date)
          delta = delta.months + (12*delta.years)
          if delta in list(range(-3, 4)):
            return ev.strip()
            
        return 'date_issue'
    except Exception as e:
        print("exception caught...:", url)
        print(e)
        with open('missing.csv', 'a', encoding='utf-8') as f:
            f.write(str(id_) + ',' + url + '\n')
        return 'exp'

def crawl_news(id_, link, article_date):
    try:
        article = Article(link, keep_article_html = True)
        slept = 0
        article.download()
        while article.download_state == ArticleDownloadState.NOT_STARTED:
            if slept > 9:
                raise ArticleException('Download never started')
            sleep(1)
            slept += 1
        article.parse()
        article.nlp()
        
        ev_date = article.publish_date.date()
        if ev_date:
          delta = relativedelta.relativedelta(ev_date, article_date)
          delta = delta.months + (12*delta.years)
          
          if delta in list(range(-3, 4)):
            return article.text.strip()
            
        return 'date_issue'

    except Exception as e:
        print('Exception caught...', link)
        print(e)
        with open('missing.csv', 'a', encoding='utf-8') as f:
            f.write(str(id_) + ',' + link + '\n')
            
        return 'exp'

def crawl_truthoffiction(id_, link, article_date):

    try:
        article = Article(link, keep_article_html = True)
        slept = 0
        article.download()
        while article.download_state == ArticleDownloadState.NOT_STARTED:
            if slept > 9:
                raise ArticleException('Download never started')
            sleep(1)
            slept += 1
        article.parse()
        article.nlp()
        
        ev_date = article.publish_date.date()
        if ev_date:
          delta = relativedelta.relativedelta(ev_date, article_date)
          delta = delta.months + (12*delta.years)
          if delta in list(range(-3, 4)):
            return article.text.strip()
            
        return 'date_issue'

    except Exception as e:
        print("exception caught...:", link)
        print(e)
        with open('missing.csv', 'a', encoding='utf-8') as f:
            f.write(str(id_) + ',' + link + '\n')
            
        return 'exp'

def crawl_politifact(id_, link, article_date):

    try:
        article = Article(link, keep_article_html = True)
        slept = 0
        article.download()
        while article.download_state == ArticleDownloadState.NOT_STARTED:
            if slept > 9:
                raise ArticleException('Download never started')
            sleep(1)
            slept += 1
        article.parse()
        article.nlp()
        
        ev_date = article.publish_date.date()
        if ev_date:
          delta = relativedelta.relativedelta(ev_date, article_date)
          delta = delta.months + (12*delta.years)
          if delta in list(range(-3, 4)):
            return article.text.strip()
            
        return 'date_issue'

    except Exception as e:
        print("exception caught...:", link)
        print(e)
        with open('missing.csv', 'a', encoding='utf-8') as f:
            f.write(str(id_) + ',' + link + '\n')
        return 'exp'

def ev_gathering(id_, links, dop):
    article_date = parser.parse(dop).date()
    #links = string of urls separated by ','
    evidence = []
    count = 0
    print('Extracting id:', id_)
    if links == '':
      token = [id_, 'nan', '']
      evidence.append(token)
      df = pd.DataFrame(evidence, columns = ['id', 'evidence_text', 'evidence_url'])
      return df
      
    ev = links.split(',')
    if len(ev) == 0:
        token = [id_, '']
        evidence.append(token)

    else:
        for link in ev:
            if count == 3:
              break
            sleep(1)
            if 'snopes' in link:
                text = crawl_snopes(id_, link, article_date)
                if text not in ['nan', 'exp', 'date_issue']:
                  count+=1
                if text not in ['date_issue', 'exp']:
                  token = [id_, text, link]
                  evidence.append(token)

            elif 'truthorfiction' in link:
                text = crawl_truthoffiction(id_, link, article_date)
                if text not in ['nan', 'exp', 'date_issue']:
                  count+=1
                if text not in ['date_issue', 'exp']:
                  token = [id_, text, link]
                  evidence.append(token)

            elif 'politifact' in link:
                text = crawl_politifact(id_, link, article_date)
                if text not in ['nan', 'exp', 'date_issue']:
                  count+=1
                if text not in ['date_issue', 'exp']:
                  token = [id_, text, link]
                  evidence.append(token)

            else:
                text = crawl_news(id_, link, article_date)
                if text not in ['nan', 'exp', 'date_issue']:
                  count+=1
                if text not in ['date_issue', 'exp']:
                  token = [id_, text, link]
                  evidence.append(token)

        df = pd.DataFrame(evidence, columns = ['id', 'evidence_text', 'evidence_url'])
        return df