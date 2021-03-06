from urllib.request import urlopen #Web browser tool
from bs4 import BeautifulSoup #html reader tool
import re #string parsing tool
import numpy as np
import pandas as pd

#creating databank based off first 100 words toddlers should learn, pronouns, and the 250 most common adverbs in English

#website is: http://teachmetotalk.com/2008/02/12/first-100-words-advancing-your-toddlers-vocabulary-with-words-and-signs/
#Conjugation website is: http://conjugator.reverso.net
#Pronouns from website: https://www.ef.edu/english-resources/english-grammar/pronouns/
#Adverbs from: http://www.talkenglish.com/vocabulary/top-250-adverbs.aspx

#Creating verbs databank
beginning_verbs = ['eat','be','drink', 'go', 'stop', 'run', 'jump', 'walk', 'sleep', 'wash', 'kiss', 'open', 'close', 
'push', 'pull', 'fix', 'broke', 'play','want', 'hug', 'love', 'hurt', 'tickle', 'give','dance', 
'help', 'fall', 'shake', 'see', 'watch', 'look', 'sit', 'stand','throw', 'catch', 'blow', 'cry', 'throw', 
'swing', 'slide', 'climb', 'ride', 'rock', 'come', 'color','draw']

verbs_learned = list()

#Add each conjugation of each verb in the list to the database, seperated by 1st person, 2nd person, and 3rd person
for element in beginning_verbs:
    verb_conjugations = urlopen('http://conjugator.reverso.net/conjugation-english-verb-'+element+'.html')
    soup = BeautifulSoup(verb_conjugations,"html5lib")
    Verbs = list()
    tenses = list()
    pov = list()
    i=0
    for perspectives in soup.find_all('div', attrs={'class': 'responsive-sub responsive-sub-50'}):
        if i < 2:
            Verbs.append(perspectives.get_text(" ",strip=True))
            tense_perspective = re.split('(?:^|(?<= ))(I|you|he/she/it|we|they)(?:(?= )|$)',Verbs[-1])
            tenses.append(tense_perspective[0])
            pov.append(tense_perspective[2::2])
        i=i+1
    for perspectives in soup.find_all('div', attrs={'class': 'responsive-sub responsive-sub-25'}):
        Verbs.append(perspectives.get_text(" ",strip=True))
        tense_perspective = re.split('(?:^|(?<= ))(I|you|he/she/it|we|they)(?:(?= )|$)',Verbs[-1])
        tenses.append(tense_perspective[0])
        pov.append(tense_perspective[2::2])

    pov = np.array(pov).transpose()
    pov = list(pov)
    verbs_learned.append(pov)
    
Verbs_1_S = list()
Verbs_2_S = list()
Verbs_3_S = list()
Verbs_1_P = list()
Verbs_2_P = list()
Verbs_3_P = list()
for verbs in verbs_learned:
    Verbs_1_S.append(verbs[0])
    Verbs_2_S.append(verbs[1])
    Verbs_3_S.append(verbs[2])
    Verbs_1_P.append(verbs[3])
    Verbs_2_P.append(verbs[4])
    Verbs_3_P.append(verbs[5])
    
#Converting Verb lists into dataframes indexed by present tense
Verbs_1_S = pd.DataFrame(Verbs_1_S, columns=tenses)
Verbs_1_S = Verbs_1_S.set_index(['Present '])

Verbs_2_S = pd.DataFrame(Verbs_2_S, columns=tenses)
Verbs_2_S = Verbs_2_S.set_index(['Present '])

Verbs_3_S = pd.DataFrame(Verbs_3_S, columns=tenses)
Verbs_3_S = Verbs_3_S.set_index(['Present '])

Verbs_1_P = pd.DataFrame(Verbs_1_P, columns=tenses)
Verbs_1_P = Verbs_1_P.set_index(['Present '])

Verbs_2_P = pd.DataFrame(Verbs_2_P, columns=tenses)
Verbs_2_P = Verbs_2_P.set_index(['Present '])

Verbs_3_P = pd.DataFrame(Verbs_3_P, columns=tenses)
Verbs_3_P = Verbs_3_P.set_index(['Present '])

#Prepositions list
Prepositions = [' up ', ' down ', ' in ', ' out ', ' off ', ' on ', ' here ', ' there ',' around ', ' under ', ' behind ', 
                ' over at ',' after ']
#Social Function list
Social_Function = [' more ', ' please ', ' thank you ', ' hi ',' hello ', ' bye-bye ', ' again ', ' sorry ',' uh-oh ', ' yes ',
                   ' uh-huh ',' okay ', ' no,uh-uh ']

#Pronouns list, separated by subjective, objective, possessive (n.), possessive(adj.), and reflexive tones
Pronouns_Subj_S = [' I ',' he ',' she ',' it ']
Pronouns_Subj_P = [' you ',' we ',' they ']
Pronouns_Obj_S = [' me ',' him ',' her ',' it ']
Pronouns_Obj_P = [' you ',' us ',' them ']
Pronouns_Poss_Adj_S = [' my ',' his ',' her ',' its ']
Pronouns_Poss_Adj_P = [' your ',' our ',' their ']
Pronouns_Poss_N_S = [' mine ',' his ',' hers ']
Pronouns_Poss_N_P = [' ours ',' yours ',' theirs ']
Pronouns_Refl_S = [' myself ',' yourself ',' himself ',' herself ']
Pronouns_Refl_P = [' ourselves ',' yourselves ',' themselves ']

#Adjectives list
Adjectives = ['big', 'little', 'hot', 'cold', 'loud', 'quiet', 'yucky', 'icky', 'scary', 'funny', 
              'silly', 'dirty', 'clean','gentle', 'wet', 'soft', 'fast', 'slow', 'red', 'blue', 'yellow', 
              'green', 'pink', 'orange', 'purple', 'black','white', 'brown', 'all', 'none', 'more', 
              'some', 'rest', '1', '2', '3','one','two','three']
#Nouns Singular list (plural will be created and updated as needed)
Nouns_S = ['ball', 'book', 'choo–choo', 'train', 'bike', 'rain', 'bubbles', 'car', 'truck', 'boat', 'plane',
                   'baby', 'bowl', 'spoon', 'diaper', 'sock', 'shoe', 'shirt', 'pants', 'hat', 'star', 'flower', 
                   'house', 'tree', 'brush', 'towel', 'bath', 'chair', 'table', 'bed', 'blanket', 'light', 'cookie', 
                   'cracker', 'chip', 'cheese', 'apple', 'banana', 'ice cream', 'cereal','candy', 'milk', 'juice', 
                   'water', 'dog', 'cat', 'fish', 'bird', 'duck', 'cow', 'horse', 'bunny', 'bear','pig', 'lion', 'elephant', 
                   'giraffe', 'zebra', 'monkey', 'chicken', 'butterfly', 'bee', 'frog','alligator, snake']

#Web Scraping 250 most common adverbs
adverbs_list = urlopen('http://www.talkenglish.com/vocabulary/top-250-adverbs.aspx')
soup = BeautifulSoup(adverbs_list,"html5lib")
Adverbs = list()
html_adverbs = soup.find_all('a')
for text in html_adverbs:
    if "/how-to-use/" in text['href']:
        if "sentences" not in text['href']:
            if "collocations" not in text['href']:
                Adverbs.append(text.get_text())

