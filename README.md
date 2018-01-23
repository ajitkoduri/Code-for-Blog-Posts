Code I used to work on my analytical overviews in my blog.

Release on: January 23, 2018

Machine Novelist Project: https://github.com/ajitkoduri/Code-for-Blog-Posts/tree/master/Machine%20Novelist%20Project
C++ Header files: 
https://github.com/ajitkoduri/Code-for-Blog-Posts/blob/master/Machine%20Novelist%20Project/CSVReader.h
https://github.com/ajitkoduri/Code-for-Blog-Posts/blob/master/Machine%20Novelist%20Project/TrieDS.h

C++ Part of Speech Labeller:
https://github.com/ajitkoduri/Code-for-Blog-Posts/blob/master/Machine%20Novelist%20Project/Part_of_Speech_Labeller.cpp

Folder of all the work I've done on the project till date. Updated VocabularyBuilder.py to write new data files for a part of speech labeller to operate in C++ environment. CSVReader.h is a header file for C++ that decodes .csv files. TrieDS.h is another header file for C++ that has the code for creation of a trie (prefix tree) data structure for C++. Part_of_Speech_Labeller.cpp is a C++ code that uses these two header files as well as the Standard Template Library for C++ to identify words based on their part of speech using the vocabulary allocated for this file.

Now that this is finished, future updates to the project will be on finding meaning in sentences and paragraphs, and using that to build meaning in sentences and paragraphs that the machine creates.


-------------

Release on: January 15, 2018

Machine Novelist Project: https://github.com/ajitkoduri/Code-for-Blog-Posts/blob/master/VocabularyBuilder.py

This is a python file for me as I am creating a basic vocabulary for the machine to be able to use. I use some webscraping to gather to data (toddler level vocabulary).

Python libraries required: numpy, pandas, re, bs4, urllib.request

-------------

Release on: Dec 25, 2017

Sex Crimes in Chicago: https://ajitdoes.com/2017/12/25/sex-crimes-in-chicago/

Chicago+Sex+Crimes+Analysis.py contains analysis of sex crimes (specifically prostitution, offenses against children, human trafficking, sexual assault, stalking, and sexual offenses), looking at a geographical and temporal spread. I use the population statistics of the zipcode that the crime is committed, and then I further break it down using Machine Learning to predict whether a criminal will be caught for the action or not.

MP4 files are videos of the change in crime location for each specified type of crime over the years.

Chicago_Zip_Data.csv is a file of population statistics of each zipcode

ChicagoSexCrimesZipCodeData.csv is a file that contains the zipcode of each location of the crime

Python libraries required: pandas, numpy, scikit-learn, itertools, matplotlib, IPython.display, seaborn

-------------

Release on: Dec 18, 2017

CDC 7 Word Ban: https://ajitdoes.com/2017/12/18/a-look-into-the-cdc-word-ban/

CDC 7 Word Ban for Reddit.py is a file that looks over the original Reddit post in R/politics about the CDC word ban. In there,
I look at the general sentiment of the posts versus the popularity of the opinion, explore the word usage in the comments with both
frequency distribution and sentiment analysis. I also look at the subreddits the users that posted on the thread frequent, and analyzed
which subreddits were the most popularly voted and which were the least. Following that, I make a word cloud from the cumulative comments
and a frequency distribution of words used.

Python libraries required: praw, pandas, numpy, nltk, textblob, matplotlib, time, and wordcloud
