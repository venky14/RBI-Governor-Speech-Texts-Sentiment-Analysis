#!/usr/bin/env python
# coding: utf-8

# ## Project: RBI Governor Speech Texts - Sentiment Analysis
# 
# ### Introduction
# RBI Governor Speech Texts Sentiment Analysis is the project on web scraping, text pre-processing and normalization, data visualization and sentiment analysis using data provided by [RBI](https://www.rbi.org.in/Scripts/BS_ViewSpeeches.aspx). Used various python tools and libraries to perform sentiment analysis over a speech texts by RBI governor.

# In[1]:


from bs4 import BeautifulSoup # extracting speech text from HTML doc
import nltk # for pre-processing text
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob # for sentiment analysis

import string
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud
import seaborn as sns
sns.set_context('notebook')


# ### Extracting text from markup like HTML document formats for each speech

# In[14]:


htmlfile = open('RBI_governor_speech/Raghuram_rajan/Reserve Bank of India - Speeches_1.htm', encoding="utf8").read()

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


# In[15]:


htmlfile = open('RBI_governor_speech/Raghuram_rajan/Reserve Bank of India - Speeches_2.htm', encoding="utf8").read()

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


# In[16]:


htmlfile = open('RBI_governor_speech/Raghuram_rajan/Reserve Bank of India - Speeches_3.htm', encoding="utf8").read()

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


# In[17]:


# Combining all three speech texts into one 
raghuram_rajan_speeches = (speech_1 +"\n"+ speech_2 +"\n"+ speech_3)
print(raghuram_rajan_speeches)


# ## Text Analysis Operations using NLTK

# In[18]:


# Lets break text paragraphs into sentences

tokenized_text=sent_tokenize(raghuram_rajan_speeches)
print(tokenized_text)


# In[20]:


# Word tokenizer breaks text paragraph into words.

tokenized_word=word_tokenize(raghuram_rajan_speeches)
print(tokenized_word)


# In[21]:


# lets find Frequency Distribution of each words

fdist = FreqDist(tokenized_word)
print(fdist)


# In[22]:


fdist.most_common(5)


# In[23]:


# Frequency Distribution Plot

fdist.plot(30,cumulative=False)
plt.show()


# In[24]:


# Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.

stop_words=set(stopwords.words("english"))
print(stop_words)


# In[25]:


# Removing Stopwords
filtered_word=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_word.append(w)
print("Tokenized Words:",tokenized_word[:100])
print("Filterd Words:",filtered_word[:100])


# In[26]:


# lets find Frequency Distribution of filtered words

fdist = FreqDist(filtered_word)
print(fdist)


# In[27]:


fdist.plot(30,cumulative=False)
plt.show()


# In[28]:


# Lets get rid of the punctuation
# Python provides a constant called string.punctuation that provides a great list of punctuation characters.
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in filtered_word]
print(stripped[:200])


# In[29]:


# remove all tokens that are not alphabetic
filtered_words = [word for word in stripped if word.isalpha()]
print(filtered_words[:100])


# In[30]:


# lets find Frequency Distribution of filtered words

fdist = FreqDist(filtered_words)
print(fdist)


# In[31]:


fdist.plot(30,cumulative=False)
plt.show()


# #### Much better after removing stopwords and punctuation!

# #### Lexicon Normalization
# 
# 
# Lexicon normalization considers another type of noise in the text. For example, connection, connected, connecting word reduce to a common word "connect". It reduces derivationally related forms of a word to a common root word.

# In[33]:


#Lexicon Normalization
#performing Stemming and Lemmatization
ps = PorterStemmer()

stemmed_words=[]
for w in filtered_words:
    stemmed_words.append(ps.stem(w))
    
print("Filtered Words:",filtered_words[:100])
print("Stemmed Words:",stemmed_words[:100])


# In[34]:


# Lets try Lemmatization
lem = WordNetLemmatizer()

lemma_words=[]
for w in filtered_words:
    lemma_words.append(lem.lemmatize(w))
    
print("Filtered Words:",filtered_words[:100])
print("Lemmatize Words:",lemma_words[:100])


# #### Much better after performing Lemmatization

# In[37]:


print('Total number of words after text pre-processing :', len(lemma_words))


# ### WordCloud for Raghuram Rajan Speech

# #### Now Lets look at importance of each word frequency from Raghuram Rajan's speech using WordCloud
# #### Which will help us to get insights about his Audience, for eg. Tenurity

# In[43]:


filtered_speech_words = str(lemma_words)

wordcloud = WordCloud(width=1000, height=500,
                      random_state=21, max_font_size=110).generate(filtered_speech_words)
plt.figure(figsize=(18, 15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# #### Great! We can say his speech is mostly focused on topics - Economy growth, Market, Public sector bank, Government, Loan, Debt etc.

# #### Now lets plot most occurring words from Raghuram Rajan's speech

# In[44]:


# Count all unique words

speech_word_counts = Counter(lemma_words)


# In[49]:


# Plot top 20 most frequently occuring words from Raghuram Rajan
rr_common_words = [word[0] for word in speech_word_counts.most_common(20)]
rr_common_counts = [word[1] for word in speech_word_counts.most_common(20)]

# Using background style
plt.style.use('dark_background')
plt.figure(figsize=(18, 12))

sns.barplot(x=rr_common_words, y=rr_common_counts)
plt.title('Most Common Words used by Raghuram Rajan')
plt.show()


# In[50]:


# See count list of most common words
print("25 most common words:\nWord\t\tCount")
for word, count in speech_word_counts.most_common(25):
    print("{}\t\t{}".format(word, count))


# ### Get Sentiment scores from Raghuram Rajan's speech

# In[52]:


# Using TextBlob to get sentiment scores from text
speech_text_object = TextBlob(filtered_speech_words)

# textblob has a pre-trained sentiment analysis model that we can use
speech_text_object.sentiment


#     TextBlob.sentiment
# 
#     Return a tuple(value pair) of form (polarity, subjectivity ) where polarity is a float(number) within the range [-1.0, 1.0] and subjectivity is a float(number) within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.

# #### What these scores say is that Raghuram Rajan's speech text is fairly subjective (opinionated) but very neutral in polarity (not phrased in a negative or positive way)

# ### Plot the words by their sentiment from Raghuram Rajan's speech

# In[61]:


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
plt.title('Sentiment analysis of words from Raghuram Rajan speech\n')
plt.xlabel('Polarity (Negative or Positive)')
plt.ylabel('Subjectivity (0 - purly objective, 1 - purly subjective)')

# display
plt.show()


# ## Thank You.

# In[ ]:




