from django.shortcuts import render
from django.shortcuts import redirect

import tensorflow as tf
tf.to_float = lambda x: tf.cast(x, tf.float32)
import numpy as np 
np.random.seed(42)

#For retrieving claim sentences
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
model_sent = SentenceTransformer('bert-base-nli-mean-tokens')
import scipy

#For cleaning
import re
from nltk.corpus import stopwords

#Importing modules for FactFinder pipeline
from modules.query_formulation import get_query
from modules.web_search import get_evidence
from modules.evidence_gathering import ev_gathering
from modules.nli_module import nli_prediction
from modules.clickbait import click_prediction

class Color:

    blue = '\033[94m'
    red =  '\033[93m'

def home(request):
    return render(request, 'home.html')
    
def basic_clean(text):
    lines = text.split('\n')
    new_text = ''
    for line in lines:
        if line:
            line = line.strip()
            line = re.sub(r" ?\([^)]+\)", "", line)
            new_text += line
            new_text += ' '
    
    return new_text.strip()

def clean_for_nli(text):
    text = str(text)
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub('<[^<]+?>', ' ', text)
    text = text.strip()
    return text.lower()
    
def clean_for_wordcloud(text):
    stop_words = list(set(stopwords.words('english')))
    
    def rm_stopwords(word):
        if (word in stop_words):
            return False
        else:
            return True
    unique_words = set(text.split(' '))

    filtered = list(filter(rm_stopwords, unique_words))
    
    return ' '.join(filtered)
    
def highlighting(final_query, text, claim):
    query_words = final_query.split(' ')
    blob = TextBlob(text) #text
    
    sentences = []
    
    #Highlight query words
    for sentence in blob.sentences:
      new_sent = ''  
      for word in sentence.split(' '):
        if word in query_words:
          word = '<b style="color:red;">' + word + '</b>'
          new_sent += ' ' + word
        else:
          new_sent += ' ' + word
  
      sentences.append(new_sent.strip())

    #Highlight top 3 claim sentences based
    sentence_embeddings = model_sent.encode(sentences)    
    query_embedding = model_sent.encode(claim)
    number_top_matches = 3
    
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    
    for idx, distance in results[0:number_top_matches]:
      sentences[idx] = '<mark>' + sentences[idx] + '</mark>'   
    
    return sentences
    
def result(request):
    #Loading NER BERT model
    lis = []
    lis.append(request.GET['headline'])
    lis.append(request.GET['article'])
    lis.append(request.GET['date'])

    #BUILDING QUERY FOR WEB SEARCH    
    final_query = get_query(lis[0], lis[1])
    
    #EXTRACTING EVIDENCE URLS
    urls = get_evidence(final_query)
    
    #EXTRACTING EVIDENCE TEXT
    print('********Extracting at most 3 evidence********')
    df = ev_gathering(0, urls, lis[2])

    #************PREDICTING VERACITY********************
    print('********Predicting degree of veracity********')
    evid_res = []
    urls_new = []
    evid_label = {0:'False', 1: 'Partial Truth', 2: 'Opinions Stated As Fact',
                    3: 'True', 4:'NEI'}
                    
    text = basic_clean(lis[1])
    text = clean_for_nli(text)

    for ev_article, url in zip(df['evidence_text'], df['evidence_url']):
        new_text = basic_clean(ev_article)
        initial_200 = ' '.join(text.split()[:200])
        
        if new_text == '':
          final_200 = 'nan'
        else:
          final_200 = ' '.join(new_text.split()[-200:])
        results, score, avg = nli_prediction(initial_200, final_200)
        if score >= avg:
          evid_res.append(evid_label.get(results))
          urls_new.append(url)

    
    #CLICKBAIT PREDICTION
    print('********Predicting clickbait probability********')
    cb1, ncb1 = click_prediction(lis[0])
    cb1 = int(cb1)
    cb1 = str(cb1) + '%'
    ncb1 = int(ncb1)
    ncb1 = str(ncb1) + '%'

    
    #HIGHLIGHT QUERY WORDS AND SIMILAR SENTENCES
    claim = lis[0] #Assume headline as the point of the article
    sentences = highlighting(final_query, lis[1], claim)

    #CLEAN STRING FOR WORD CLOUD
    cloud_string = clean_for_wordcloud(text)

    return render(request, 'result.html', {'ans' : urls_new, 'headline': lis[0], 'dt' : lis[2], 'evid' : evid_res, 'cb': cb1, 'ncb': ncb1, 'sent' : sentences, 'unique':cloud_string, 'range' : range(10)})

def url_redirect(request):
    test = request.GET['website_name']
    print("Hello")
    print(test)
    return redirect(test)
