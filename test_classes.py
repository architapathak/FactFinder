# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:25:21 2021

@author: archita
"""

from modules.query_formulation import get_query

headline = 'Physician Confirms Hillary Clinton Has Parkinsons Disease'
content = '''As Hillary Clinton makes her last few rounds of appearances before election day, the presidential hopeful will feel a little bit of the pressure that the presidency will demand. However, when she arrived at her rally in Lake Worth, Florida, she attempted to make one move that she hoped no one would notice. 

There have been many questions over the course of Hillary Clinton’s campaign regarding her health. As a result of her age and her lifestyle, there’s no doubt that her body is starting to feel the effects of aging – whether she likes to admit it or not. 

However, video footage of her recent rally in Florida shows how desperate she is to keep her health problems hidden, but her attempt was met with failure the moment she hit the stage. In fact, the Democratic presidential candidate struggled to climb one single step. The video captures one of her aides rushing towards her side as she attempted to climb up the riser. 

“She needed assistance to get onto it as she could be seen reaching her hand out for a boost or some added steadiness,” according to The American Mirror . “The aide extended his hand and Clinton held on tightly as she made her way up the 18 inches.” 

Although Hillary Clinton insists that her health isn’t an issue, the number of incidents she has had paints quite a different picture. If you take a close look at the video, the aide can even be seen standing behind her, reaching for her waist – probably preparing for the event of her falling. 

We have said it time and time again – she’s not fit to be president. Although her health is a major concern considering how demanding the presidency can be on a person, it’s her disastrous political history that raises, even more, concerns.'''

query = get_query(headline, content)
print(query)