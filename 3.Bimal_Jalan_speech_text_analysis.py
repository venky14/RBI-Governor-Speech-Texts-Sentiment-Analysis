#!/usr/bin/env python
# coding: utf-8

# ## Project: RBI Governor Speech Texts - Sentiment Analysis
# 
# ### Introduction
# RBI Governor Speech Texts Sentiment Analysis is the project on web scraping, text pre-processing and normalization, data visualization and sentiment analysis using data provided by [RBI](https://www.rbi.org.in/Scripts/BS_ViewSpeeches.aspx). Used various python tools and libraries to perform sentiment analysis over a speech texts by RBI governor.

# In[1]:


import nltk # for pre-processing text
from bs4 import BeautifulSoup # extracting speech text from HTML doc


# ## Extracting text from markup like HTML document formats for each speech

# In[2]:


htmlfile = open('RBI_governor_speech//Bimal_jalan//Reserve Bank of India - Speeches_1.htm', 'r').read()

soup = BeautifulSoup(htmlfile)

for speech_text_1 in soup.findAll(attrs={'class' : 'tablecontent2'}):
    speech_text_1 = speech_text_1.text.strip()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in speech_text_1.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    speech_1 = '\n'.join(chunk for chunk in chunks if chunk)
    
print(speech_1)


# In[3]:


htmlfile = open('RBI_governor_speech//Bimal_jalan//Reserve Bank of India - Speeches_2.htm', 'r').read()

soup = BeautifulSoup(htmlfile)

for speech_text_2 in soup.findAll(attrs={'class' : 'tablecontent2'}):
    speech_text_2 = speech_text_2.text.strip()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in speech_text_2.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    speech_2 = '\n'.join(chunk for chunk in chunks if chunk)
    
print(speech_2)


# In[5]:


htmlfile = open('RBI_governor_speech//Bimal_jalan//Reserve Bank of India - Speeches_3.htm', 'r').read()

soup = BeautifulSoup(htmlfile)

for speech_text_3 in soup.findAll(attrs={'class' : 'tablecontent2'}):
    speech_text_3 = speech_text_3.text.strip()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in speech_text_3.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    speech_3 = '\n'.join(chunk for chunk in chunks if chunk)
    
print(speech_3)


# In[7]:


# Combining all three speech texts into one 
bimal_jalan_speeches = (speech_1 +"\n"+ speech_2 +"\n"+ speech_3)
print(bimal_jalan_speeches)


# ## Text Analysis Operations using NLTK

# In[9]:


# Lets break text paragraphs into sentences
from nltk.tokenize import sent_tokenize
tokenized_text=sent_tokenize(bimal_jalan_speeches)
print(tokenized_text)


# In[10]:


# Word tokenizer breaks text paragraph into words.
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(bimal_jalan_speeches)
print(tokenized_word)


# In[11]:


# lets find Frequency Distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)


# In[12]:


fdist.most_common(5)


# In[14]:


# Frequency Distribution Plot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fdist.plot(30,cumulative=False)
plt.show()


# In[15]:


# Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[21]:


# Removing Stopwords
filtered_word=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_word.append(w)
print("Tokenized Words:",tokenized_word)
print("Filterd Words:",filtered_word)


# In[22]:


# lets find Frequency Distribution of filtered words
from nltk.probability import FreqDist
fdist = FreqDist(filtered_word)
print(fdist)


# In[23]:


fdist.plot(30,cumulative=False)
plt.show()


# #### Great!

# In[25]:


import string
print(string.punctuation)


# 
# 
# Python provides a constant called string.punctuation that provides a great list of punctuation characters.

# In[27]:


# Lets get rid of the punctuation
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in filtered_word]
print(stripped[:200])


# In[29]:


# remove all tokens that are not alphabetic
filtered_words = [word for word in filtered_word if word.isalpha()]
print(filtered_words[:100])


# In[30]:


# lets find Frequency Distribution of filtered words
from nltk.probability import FreqDist
fdist = FreqDist(filtered_words)
print(fdist)


# In[31]:


fdist.plot(30,cumulative=False)
plt.show()


# #### much better after removing punctuation!
# 

# #### Lexicon Normalization
# 
# 
# Lexicon normalization considers another type of noise in the text. For example, connection, connected, connecting word reduce to a common word "connect". It reduces derivationally related forms of a word to a common root word

# In[35]:


#Lexicon Normalization
#performing Stemming and Lemmatization
from nltk.stem import PorterStemmer
ps = PorterStemmer()

stemmed_words=[]
for w in filtered_words:
    stemmed_words.append(ps.stem(w))
    
print("Filtered Words:",filtered_words[:100])
print("Stemmed Words:",stemmed_words[:100])


# #### Stemming not helping much!

# In[36]:


# Lets try Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

lemma_words=[]
for w in filtered_words:
    lemma_words.append(lem.lemmatize(w))
    
print("Filtered Words:",filtered_words[:100])
print("Lemmatize Words:",lemma_words[:100])


# #### Much better than stemming we can say for now

# In[39]:


from wordcloud import WordCloud
import seaborn as sns
sns.set_context('notebook')


# #### Now Lets look at importance of each word frequency from Bimal Jalan's speech with WordCloud
# #### Which will help us to get insights about his Audience, for eg. Tenurity

# In[43]:


print('Total number of words after text pre-processing :', len(lemma_words))


# In[63]:


filtered_speech_words = str(filtered_words)


# ### Wordcloud for Bimal Jalan

# In[64]:


wordcloud = WordCloud(width=800, height=500,
                      random_state=21, max_font_size=110).generate(filtered_speech_words)
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# #### Great!

# #### Now lets plot most occurring words from Bimal Jalan's speech

# In[67]:


# Count all unique words
from collections import Counter

speech_word_counts = Counter(lemma_words)


# In[68]:


# Plot top 20 most frequently occuring words from Bimal Jalan
bm_common_words = [word[0] for word in speech_word_counts.most_common(20)]
bm_common_counts = [word[1] for word in speech_word_counts.most_common(20)]

# Using background style
plt.style.use('dark_background')
plt.figure(figsize=(15, 12))

sns.barplot(x=bm_common_words, y=bm_common_counts)
plt.title('Most Common Words used by Bimal Jalan')
plt.show()


# In[72]:


# See count list of most common words
print("25 most common words:\nWord\t\tCount")
for word, count in speech_word_counts.most_common(25):
    print("{}\t\t{}".format(word, count))


# ### Get Sentiment scores from Bimal Jalan's speech

# In[88]:


# Using TextBlob to get sentiment scores from text

from textblob import TextBlob
speech_text_object = TextBlob(bimal_jalan_speeches)

# textblob has a pre-trained sentiment analysis model that we can use
speech_text_object.sentiment


#     TextBlob.sentiment
# 
#     Return a tuple(value pair) of form (polarity, subjectivity ) where polarity is a float(number) within the range [-1.0, 1.0] and subjectivity is a float(number) within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.

# #### What these scores say is that Bimal Jalan's speech text is fairly subjective (opinionated) but very neutral in polarity (not phrased in a negative or positive way)

# In[98]:


# Lets plot the words by their sentiment

plt.figure(figsize=(18,15))

# for each word draw the text on the char using the sentiment score as the x and y coordinates
for word in lemma_words:
    word_sentiment = TextBlob(word).sentiment
    plt.text(word_sentiment.polarity, # x coordinate
             word_sentiment.subjectivity, # y coordinate
             word) # the text to draw

# set axis ranges 
plt.xlim(-1, 1)
plt.ylim(0, 1)

# draw line in middle
plt.axvline(0, color='red', linestyle='dashed')

# label axis
plt.title('Sentiment analysis of words from Bimal Jalan speech\n')
plt.xlabel('Polarity (negative OR positive)')
plt.ylabel('Subjectivity (0 - purly objective, 1 - purly subjective)')

# display
plt.show()


# #### Performing quality text cleaninng and pre-processing was a lengthy process for data from HTML doc.
# #### P.S.: I've tried to build a Word2Vec model from speech text of Bimal Jalan to create a custom word embeddings from that model with the help Genism_Word2Vec_model but performance wasn't that good.
# 
# ## Thank You.

# In[ ]:




