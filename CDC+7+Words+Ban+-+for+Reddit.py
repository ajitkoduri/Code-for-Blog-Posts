
# coding: utf-8

# In[1]:

import praw #library for Reddit APIs
import pandas as pd, numpy as np #Dataframes and easy number manipulations
import keys #personal library of API keys for Facebook, Google, Twitter, and Reddit
import nltk #Frequency distribution
from nltk.tokenize import RegexpTokenizer #tokenizer
from textblob import TextBlob #Sentiment Analysis
import matplotlib.pyplot as plt #Plot Functions
import time #Pause system if there are too many API requests
import matplotlib
from wordcloud import WordCloud #Makes a word cloud


# In[ ]:

Reddit_Data = []
reddit = praw.Reddit(user_agent = "CDC 7 Word Ban (by /u/MedianEnergy)",
                    client_id = keys.redkey, client_secret = keys.redsecret)
tokenizer = RegexpTokenizer(r'\w+') #Tokenizer that ignores punctuation
submission = reddit.submission(url="https://www.reddit.com/r/politics/comments/7k3sab/cdc_gets_list_of_forbidden_words_fetus/")
submission.comments.replace_more(limit=None)
with open("CDC7wordsbancomments.txt",'w+') as file:
    for comment in submission.comments.list():
        
        Comment_Tokens = tokenizer.tokenize(comment.body) #List of all words in comment
        
        if(len(list(set(Comment_Tokens))) >= 5): #Removing comments with less than 5 words in them
            #Process all comments individually and see what their sentiment of the new policy is
            if (comment.author != None):
                Reddit_Data.append(comment.body)
                Reddit_Data.append(comment.score)
                Reddit_Data.append(comment.author.name)
        #Get a file of all comments to later process cumulatively
        try:
            file.write(comment.body)
            file.write('\n')
        except:
            print("skipped comment")
file.close()
#Convert Reddit Data into a Dataframe with Columns for comment, score, and poster id for each comment
Reddit_Data = np.array(Reddit_Data)
Reddit_Data = np.reshape(Reddit_Data, (-1,3))
Reddit_Data = pd.DataFrame(Reddit_Data)

Reddit_Data.columns = ['Comment', 'Score', 'Poster']
Reddit_Data['Score'] = Reddit_Data['Score'].astype(str).astype(int)


# In[ ]:

#In Reddit, the vote feature is usually it's used to depict agree or disagreement of people
#coming across the post. Because of that, it is a really good measure of a popular sentiment
#in the community. For this upcoming part, I'll be trying to analyze which words, and how
#often their said in the comment relates to a higher popularity post. This may give us a picture
#of the sentiment of people, but more likely it will give us a picture of what subtopics of the 
#legislation are being talked about the most - what is the most controversial and which of these
#subtopics are most people interested in.

Most_Comm_Words = [] #list of top 5 most common words in comment

for comment in Reddit_Data['Comment']:
    Comment_Tokens = tokenizer.tokenize(comment)
    FDist = nltk.FreqDist(Comment_Tokens)
    for word in range(5): #we append all the words and frequencies to both lists
        Most_Comm_Words.append(FDist.most_common(5)[word][0])

#Resize the arrays to have 5 columns
Most_Comm_Words = np.reshape(np.array(Most_Comm_Words), (-1,5))

#Concatenating the new dataframes into the cumulative Reddit Comments Dataset
Most_Comm_Words = pd.DataFrame(Most_Comm_Words)
Most_Comm_Words.columns = ["1 Common", "2 Common", "3 Common", "4 Common", "5 Common"]
Reddit_Data = [Reddit_Data, Most_Comm_Words]
Reddit_Data = pd.concat(Reddit_Data, axis = 1)

#Add data on sentiment value of each comment
def sentiment(text):
    return TextBlob(text).sentiment.polarity #Positive polarity is a positive sentiment, and vice versa

Sentiment_list = []
for comment in Reddit_Data['Comment']:
    Sentiment_list.append(sentiment(comment))
Sentiment_list = pd.DataFrame(Sentiment_list)
Sentiment_list.columns = ['Sentiment']

#Completed dataset for each comment on Reddit Thread
Reddit_Data = pd.concat([Reddit_Data, Sentiment_list],axis=1)


# In[ ]:

#Explore Sentiment value with Score of post with score of over 1000 votes
f, ax = plt.subplots(4,3,figsize=(20,15))
f.subplots_adjust(hspace=1)
Sentiment_Data = Reddit_Data.loc[Reddit_Data['Score'] >= 1000]

ax[0][0].set_title("Histogram of Comments with Greater than 1000 Votes")
ax[0][0].hist(Sentiment_Data['Sentiment'], bins = 50,color='r')
ax[0][0].set_xlabel("Polarity")

ax[0][1].set_title("Histogram of Comments with Greater than 1000 Votes Relative to Score")
ax[0][1].hexbin(Sentiment_Data['Sentiment'],Sentiment_Data['Score'], gridsize=25, cmap = "hot")
ax[0][1].set_xlabel("Polarity")
ax[0][1].set_ylabel("Score")

ax[0][2].set_xlabel("Polarity")
ax[0][2].set_ylabel("Score")
ax[0][2].set_title("Score v. Polarity of Comment [>1000 votes]")
ax[0][2].scatter(Sentiment_Data['Sentiment'],Sentiment_Data['Score'],color='r')
z = np.polyfit(Sentiment_Data['Sentiment'], Sentiment_Data['Score'], 1)
p = np.poly1d(z)
ax[0][2].plot(Sentiment_Data['Sentiment'], p(Sentiment_Data['Sentiment']), "k--")
ax[0][2].text(0,10000,p)

#Explore Sentiment value with Score of post with score of over 250 votes
Sentiment_Data = Reddit_Data.loc[Reddit_Data['Score'] >= 250]

ax[1][0].set_title("Histogram of Comments with Greater than 250 Votes")
ax[1][0].hist(Sentiment_Data['Sentiment'], bins = 50,color='y')
ax[1][0].set_xlabel("Polarity")


ax[1][1].set_title("Histogram of Comments with Greater than 250 Votes Relative to Score")
ax[1][1].hexbin(Sentiment_Data['Sentiment'],Sentiment_Data['Score'], gridsize=25, cmap = "hot")
ax[1][1].set_xlabel("Polarity")
ax[1][1].set_ylabel("Score")


ax[1][2].set_xlabel("Polarity")
ax[1][2].set_ylabel("Score")
ax[1][2].set_title("Score v. Polarity of Comment [>250 votes]")
ax[1][2].scatter(Sentiment_Data['Sentiment'],Sentiment_Data['Score'],color='y')
z = np.polyfit(Sentiment_Data['Sentiment'], Sentiment_Data['Score'], 1)
p = np.poly1d(z)
ax[1][2].plot(Sentiment_Data['Sentiment'], p(Sentiment_Data['Sentiment']), "k--")
ax[1][2].text(0,10000,p)

#Explore Sentiment value with Score of post with score of over 100 votes
Sentiment_Data = Reddit_Data.loc[Reddit_Data['Score'] >= 100]

ax[2][0].set_title("Histogram of Comments with Greater than 100 Votes")
ax[2][0].hist(Sentiment_Data['Sentiment'], bins = 50,color='g')
ax[2][0].set_xlabel("Polarity")

ax[2][1].hexbin(Sentiment_Data['Sentiment'],Sentiment_Data['Score'], gridsize=25, cmap = "hot")
ax[2][1].set_title("Histogram of Comments with Greater than 100 Votes Relative to Score")
ax[2][1].set_xlabel("Polarity")
ax[2][1].set_ylabel("Score")

ax[2][2].scatter(Sentiment_Data['Sentiment'],Sentiment_Data['Score'],color='g')
ax[2][2].set_xlabel("Polarity")
ax[2][2].set_ylabel("Score")
ax[2][2].set_title("Score v. Polarity of Comment [>100 votes]")
z = np.polyfit(Sentiment_Data['Sentiment'], Sentiment_Data['Score'], 1)
p = np.poly1d(z)
ax[2][2].plot(Sentiment_Data['Sentiment'], p(Sentiment_Data['Sentiment']), "k--")
ax[2][2].text(0,10000,p)

#Explore Sentiment value with Score of post with score of less than 0 votes
Sentiment_Data = Reddit_Data.loc[Reddit_Data['Score'] < 0]

ax[3][0].set_title("Histogram of Comments with Lower than 0 Votes")
ax[3][0].hist(Sentiment_Data['Sentiment'], bins = 50,color='b')
ax[3][0].set_xlabel("Polarity")

ax[3][1].set_title("Histogram of Comments With Fewer than 0 Votes Relative to Score")
ax[3][1].hexbin(Sentiment_Data['Sentiment'],Sentiment_Data['Score'], gridsize=25, cmap = "hot")
ax[3][1].set_xlabel("Polarity")
ax[3][1].set_ylabel("Score")

ax[3][2].set_xlabel("Polarity")
ax[3][2].set_ylabel("Score")
ax[3][2].set_title("Score v. Polarity of Comment [< 0 votes]")
ax[3][2].scatter(Sentiment_Data['Sentiment'],Sentiment_Data['Score'],color='b')
z = np.polyfit(Sentiment_Data['Sentiment'], Sentiment_Data['Score'], 1)
p = np.poly1d(z)
ax[3][2].plot(Sentiment_Data['Sentiment'], p(Sentiment_Data['Sentiment']), "r--")
ax[3][2].text(0.25,-100, p)

plt.show()


# In[ ]:

#Look at most common words in the reddit comments
Word_Data_1 = Reddit_Data.groupby(["1 Common"]).sum()['Score'].sort_values(ascending=False)
Word_Data_2 = Reddit_Data.groupby(["2 Common"]).sum()['Score'].sort_values(ascending=False)
Word_Data_3 = Reddit_Data.groupby(["3 Common"]).sum()['Score'].sort_values(ascending=False)
Word_Data_4 = Reddit_Data.groupby(["4 Common"]).sum()['Score'].sort_values(ascending=False)
Word_Data_5 = Reddit_Data.groupby(["5 Common"]).sum()['Score'].sort_values(ascending=False)


# In[ ]:

#Bar graphs of how often a word was used with respect to the total vote the comment it was used in received
f = plt.figure(figsize=(20,30))
f.add_subplot(611)
Word_Data_1[10:60].plot(kind="bar")
plt.title("Most Highly Rated Most Common Word in Comment")
plt.ylabel("Total Score From Using Word")
plt.xlabel("")
plt.xticks(rotation=280)

f.add_subplot(612)
Word_Data_2[12:62].plot(kind="bar")
plt.title("Most Highly Rated 2nd Most Common Word in Comment")
plt.ylabel("Total Score From Using Word")
plt.xlabel("")
plt.xticks(rotation=280)

f.add_subplot(613)
Word_Data_3.head(50).plot(kind="bar")
plt.title("Most Highly Rated 3rd Most Common Word in Comment")
plt.ylabel("Total Score From Using Word")
plt.xlabel("")
plt.xticks(rotation=280)

f.add_subplot(614)
Word_Data_4[10:60].plot(kind="bar")
plt.title("Most Highly Rated 4th Most Common Word in Comment")
plt.ylabel("Total Score From Using Word")
plt.xlabel("")
plt.xticks(rotation=280)

f.add_subplot(615)
Word_Data_5[4:54].plot(kind="bar")
plt.title("Most Highly Rated 5th Most Common Word in Comment")
plt.ylabel("Total Score From Using Word")
plt.xlabel("")
plt.xticks(rotation=280)

f.subplots_adjust(hspace=0.5)
plt.show()


# In[ ]:

#In this part, I want to break the data down by poster, and look at which of them has the highest scores 
#and their general sentiment of the ban on words. I'm going to follow that up with a breakdown of
#what other subreddits they post on, and see if that information gives us more details about whether they
#were supportive or dismissive of this ban.

#Going to look at the top 3 subreddits that the poster most frequents to comment on
Poster_Data = Reddit_Data.groupby(['Poster']).sum()
Poster_Subreddits = []
ind = 0
for user in Poster_Data.index:
    subreddits_visited =  []
    
    try:
        for comment in reddit.redditor(user).comments.new(limit=100): #go through their post history and see their frequented
            subreddits_visited.append(str(comment.subreddit))         #subreddits, going only at most 100 posts back in their
                                                                      #history
        if len(list(set(subreddits_visited))) >= 3:                   #As long as they have visited more than 3 subreddits,
            FDist = nltk.FreqDist(subreddits_visited)                 #we consider them
            for it in range(3): 
                Poster_Subreddits.append(FDist.most_common(3)[it][0])
        else:                                                         #If they haven't visited at least 3 subreddits,
            Poster_Data = Poster_Data[Poster_Data.index != user]      #we throw them out
    except:
        Poster_Data = Poster_Data[Poster_Data.index != user]          #Remove users that were suspended 
        
#Adding Poster's top 3 most visited subreddits to Data of each poster's sentiment and total score in this reddit thread
Poster_Subreddits = np.reshape(np.array(Poster_Subreddits), (-1,3))
Poster_Subreddits = pd.DataFrame(Poster_Subreddits)
Poster_Subreddits.index = Poster_Data.index
Poster_Data = [Poster_Data, Poster_Subreddits]
Poster_Data = pd.concat(Poster_Data, axis = 1)
Poster_Data.columns = ['Score','Sentiment','1 Subreddit','2 Subreddit','3 Subreddit']
Poster_Data = Poster_Data.sort_values(['Score'],ascending=False)


# In[ ]:

#Creating Subreddit data for most commonly visited subreddit, 2nd most common, and 3rd most among commenters
#Cumulative data contains the combination of the data between them
#First, we look at it through Scores of posts coming from these subreddits
Sub_Data_1 = Poster_Data.groupby(['1 Subreddit']).sum().sort_values(['Score'],ascending=True)
Sub_Data_2 = Poster_Data.groupby(['2 Subreddit']).sum().sort_values(['Score'],ascending=True)
Sub_Data_3 = Poster_Data.groupby(['3 Subreddit']).sum().sort_values(['Score'],ascending=True)
Sub_Data = Sub_Data_1 + Sub_Data_2 + Sub_Data_3
Sub_Data = Sub_Data.sort_values(['Score'],ascending=True).dropna(axis=0,how='all')

f = plt.figure(figsize=(30,20))

matplotlib.rc('ytick', labelsize=12)
f.add_subplot(241)
Sub_Data_1['Score'].tail(25).plot(kind='barh', color='r')
plt.title("Most Popular Subreddits in Comments", fontsize=16)
plt.ylabel("")
f.add_subplot(242)
Sub_Data_2['Score'].tail(25).plot(kind='barh',color='y')
plt.title("2nd Most Popular Subreddits in Comments", fontsize=16)
plt.ylabel("")
f.add_subplot(243)
Sub_Data_3['Score'].tail(25).plot(kind='barh',color='g')
plt.title("3rd Most Popular Subreddits in Comments", fontsize=16)
plt.ylabel("")
f.add_subplot(244)
Sub_Data['Score'].tail(25).plot(kind='barh',color='b')
plt.title("Total Most Popular Subreddits in Comments", fontsize=16)
plt.ylabel("")

f.add_subplot(245)
Sub_Data_1['Score'].head(25).plot(kind='barh',color='r')
plt.xlabel("Score")
plt.title("Least Popular Subreddits in Comments", fontsize=16)
plt.ylabel("")
f.add_subplot(246)
Sub_Data_2['Score'].head(25).plot(kind='barh',color='y')
plt.xlabel("Score")
plt.title("2nd Least Popular Subreddits in Comments", fontsize=16)
plt.ylabel("")
f.add_subplot(247)
Sub_Data_3['Score'].head(25).plot(kind='barh',color='g')
plt.xlabel("Score")
plt.title("3rd Least Popular Subreddits in Comments", fontsize=16)
plt.ylabel("")
f.add_subplot(248)
Sub_Data['Score'].head(25).plot(kind='barh',color='b')
plt.title("Total Least Popular Subreddits in Comments", fontsize=16)
plt.xlabel("Score")
plt.ylabel("")

f.subplots_adjust(wspace=0.5)
plt.show()


# In[ ]:

#In the final part, I'm going to do a quick graph of the total frequency distribution of the words in every
#comment, which should give us a picture of how everyone talked without respect to popularity, a sentiment 
#analysis of the whole text, and a word cloud of the whole text.
with open("CDC7wordsbancomments.txt",'r') as file:
    All_Comments = file.read().replace('\n','')
#Sentiment of comments as a whole, no regard to popularity or person who wrote them
sentiment(All_Comments)

#Make a word cloud
wordcloud = WordCloud().generate(All_Comments)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

#Do a Frequency Distribution
Tokens = tokenizer.tokenize(All_Comments)
FDist = nltk.FreqDist(Tokens)
FDist_Data = pd.DataFrame(FDist.most_common(100))
FDist_Data.columns = ["Word","Frequency"]
FDist_Data = FDist_Data.set_index(['Word'])
FDist_Data.sort_values('Frequency',ascending=False).iloc[20:75].plot(kind="bar",figsize=(20,10))
plt.title("Cumulative Word Usage")
plt.xlabel("Word")
plt.xticks(rotation=280)
plt.show()

