Code I used to work on my analytical overviews in my blog.

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
