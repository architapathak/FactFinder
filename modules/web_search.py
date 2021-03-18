# -*- coding: utf-8 -*-
"""
"""

import requests
from time import sleep

def bing_search(query):
    query = query
    url = 'https://api.cognitive.microsoft.com/bing/v7.0/search'
    payload = {'q': query, 'count':30}
    headers = {'Ocp-Apim-Subscription-Key': 'bd6e44a8dc604d8c8894a22d71ec5ae1'}
    r = requests.get(url, params=payload, headers=headers)
    return r.json()

def get_evidence(text):
    with open('credible_sources.txt', 'r') as f:
        cred = [i.strip() for i in f.readlines()]

    j = bing_search(text)
    sleep(1)

    results = j.get('webPages', {}).get('value', {})
    print('Number of evidence URL retrieved: ', len(results))
    ev_urls = ''
    
    for res in results:
        if any(link in res['url'] for link in cred):
            try:
              ev_urls = ev_urls + res['url'] + ','
            except Exception as e:
              print(e)
              pass
    return ev_urls.strip(',')