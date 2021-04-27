# -*- coding: utf-8 -*-
"""
"""

import pickle
import re
from nltk import word_tokenize
import spacy
nlp = spacy.load("en_core_web_sm")
import os
path = os.path.join(os.path.dirname(__file__), 'models')

def load_model():
  return pickle.load(open(os.path.join(path, 'save.p'), "rb" ))

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
    
def get_query(hl, text):
    ner_model = load_model()
    headline_query = get_headline_query(ner_model, hl)
    
    text = basic_clean(text)
    text_query = get_text_query(ner_model, text)
    final_query = headline_query + ' ' + text_query
    
    return final_query
    
def get_headline_query(ner_model, hl):
    #Predicting NER tokens in HEADLINE
    text_tokens = word_tokenize(hl)
    y_preds = ner_model.predict([text_tokens])
    
    #Importance Sampling of NER tokens in HEADLINE
    entities = {}
    ner_query = ''
    query = ''
    avoid = ['O', None]
    token_index = []
    for l1 in y_preds:
        for i in range(len(l1)):
            if l1[i] not in avoid and text_tokens[i] not in entities:
                token_index.append(i)
                entities[text_tokens[i]] = ' '
                ner_query += text_tokens[i]
                ner_query += ' ' 
    if len(token_index) == 0:
        return hl
    query = ''
    tags = ['VBG','VBZ','VB', 'VBN', 'VBD']
    for i in range(len(token_index)):
        if i == len(token_index) - 1:
            query += text_tokens[token_index[i]]
            query += ' ' 
        else:
            if token_index[i+1] - token_index[i] == 1:
                query += text_tokens[token_index[i]]
                query += ' '
            else:
                query += text_tokens[token_index[i]]
                query += ' '
                for j in range(token_index[i] + 1, token_index[i+1]):
                    text = text_tokens[j]
                    doc = nlp(text)
                    for token in doc:
                        if token.tag_ in tags:
                            query += token.text
                            query += ' '
    last_index = token_index[len(token_index) - 1]
    if(last_index == len(text_tokens)):
        pass
    else:
        for k in range(last_index + 1, len(text_tokens)):
            text = text_tokens[k]
            doc = nlp(text)
            for token in doc:
                if token.tag_ in tags:
                    query += token.text
                    query += ' '
    return query.strip()
    
def get_text_query(ner_model, text):
    #Predicting NER tokens in ARTICLE TEXT
    # Cleaning....
    
    text_tokens = word_tokenize(text)
    y_preds = ner_model.predict([text_tokens])
 
    #Importance Sampling of NER tokens in ARTICLE TEXT
    tags = ['VBG','VBZ','VB', 'VBN', 'VBD']
    verb_tokens_text = []
    verb_tokens = []
    for i in range(len(text_tokens)):
        doc = nlp(text_tokens[i])
        for token in doc:
            if token.tag_ in tags:
                verb_tokens.append(i)
                verb_tokens_text.append(text_tokens[i])
    
    entities = {}
    query = ''
    avoid = ['O', 'I-MISC', 'B_MISC', None]
    verbs_used = []
    i = 0
    for l1 in y_preds:
        while i < len(l1):
            if l1[i] not in avoid:
                entity = ''
                for j in range(i, len(l1)):
                    if l1[j] not in avoid:
                        entity += text_tokens[j]
                        entity += ' '
                    else:
                        break
                if entity not in entities:
                    verbs = []
                    if i - 3 >= 0:
                        for k in reversed(range(4)):
                            if i - k in verb_tokens and i - k not in verbs_used:
                                verbs.append(i - k)
                                verbs_used.append(i - k)
                for k in range(j, j + 5):
                    if k in verb_tokens:
                        verbs.append(k)
                        verbs_used.append(k)
                if len(verbs) > 0:
                    entities[entity] = [[i, j - 1], verbs]
                i = j
            else:
                i += 1
    
    query = ''
    for entry in entities:
        query += entry.strip()
        query += ' '
        try:
            verb_list = entities[entry][1]
            for index in verb_list:
                query += text_tokens[index]
                query += ' '
        except:
            pass
    
    return query.strip()