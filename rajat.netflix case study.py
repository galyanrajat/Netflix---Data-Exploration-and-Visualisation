#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


nf=pd.read_csv(r'C:\Users\galya\OneDrive\Desktop\netflix1.csv')


# In[3]:


nf.head(5)


# # 1. Defining Problem Statement and Analysing basic metrics

# In[4]:


nf.describe(include="all")


# 

# In[5]:


nf.describe(include=[np.number])


# In[6]:


nf.describe(include=[np.object]).T


# In[7]:


nf.info()


# In[8]:


# analyse null values


# In[9]:


nf.isnull().sum()/len(nf)*100


# In[10]:


# total null values


# In[11]:


nf.isnull().sum().sum()


# # 2. Observations on the shape of data, data types of all the attributes, conversion of categorical attributes to 'category' (If required), missing value detection, statistical summary# 

# In[12]:


# here is the data type of the data


# In[13]:


nf.dtypes


# In[14]:


nf.shape


# In[15]:


nf.shape[0]


# In[16]:


nf.shape[1]


# In[17]:


type(nf)


# In[18]:


# conversion of date_added(object) to datetime


# In[19]:


nf['date_added']=pd.to_datetime(nf['date_added'])


# In[20]:


# here is the information of the data


# In[21]:


nf.info()


# In[22]:


# missing value detection


# In[23]:


nf.isnull().sum()/len(nf)*100


# In[24]:


sns.heatmap(nf.isnull())


# In[25]:


# ststical summery


# In[26]:


nf.describe()


# In[27]:


nf.describe(include="all")


# # 3- Non-Graphical Analysis: Value counts and unique attributes

# In[28]:


# finding unique values


# In[29]:


nf.nunique()


# In[30]:


# here is the null value of director


# In[31]:


nf.isna()["director"].value_counts()


# In[32]:


# here is the unique value of the director


# In[33]:


nf["director"].nunique()


# In[34]:


# here is the count of values according to the release year 


# In[35]:


nf['release_year'].value_counts().head(15)


# In[36]:


# count of title


# In[37]:


nf['title'].value_counts().sum()


# In[38]:


# count of movies and shows where they listed in 


# In[39]:


nf.listed_in.value_counts().head(10)


# In[40]:


nf.groupby(['country'])['title'].count()


# In[ ]:





# In[ ]:





# # 4. Visual Analysis - Univariate, Bivariate after pre-processing of the data

# In[41]:


nf.head(2)


# In[42]:


nf_cast=nf['cast'].apply(lambda x:str(x).split(', ')).tolist()
nf_cast=pd.DataFrame(nf_cast,index=nf['title'])
nf_cast=nf_cast.stack()
nf_cast=pd.DataFrame(nf_cast)
nf_cast=nf_cast.reset_index()
nf_cast=nf_cast.drop('level_1',axis=1)
nf_cast.columns=['title','cast']
nf_cast


# In[43]:


nf_dir=nf['director'].apply(lambda x:str(x).split(', ')).tolist()
nf_dir=pd.DataFrame(nf_dir,index=nf['title'])
nf_dir=nf_dir.stack()
nf_dir=pd.DataFrame(nf_dir)
nf_dir=nf_dir.reset_index()
nf_dir=nf_dir.drop('level_1',axis=1)
nf_dir.columns=['title','director']
nf_dir


# In[44]:


nf_cou=nf['country'].apply(lambda x:str(x).split(', ')).tolist()
nf_cou=pd.DataFrame(nf_cou,index=nf['title'])
nf_cou=nf_cou.stack()
nf_cou=pd.DataFrame(nf_cou)
nf_cou=nf_cou.reset_index()
nf_cou=nf_cou.drop('level_1',axis=1)
nf_cou.columns=['title','country']
nf_cou


# In[45]:


nf_lis=nf['listed_in'].apply(lambda x:str(x).split(', ')).tolist()
nf_lis=pd.DataFrame(nf_lis,index=nf['title'])
nf_lis=nf_lis.stack()
nf_lis=pd.DataFrame(nf_lis)
nf_lis=nf_lis.reset_index()
nf_lis=nf_lis.drop('level_1',axis=1)
nf_lis.columns=['title','listed_in']
nf_lis


# In[46]:


nf_final=pd.merge(nf,nf_cast,on='title')
nf_final=pd.merge(nf_final,nf_dir,on='title')
nf_final=pd.merge(nf_final,nf_cou,on='title')
nf_final=pd.merge(nf_final,nf_lis,on='title')


# In[47]:


nf_final.head(2)


# In[48]:


nf_final.drop(columns=['director_x','cast_x','country_x','listed_in_x'],inplace=True)


# In[49]:


nf_final.rename({'cast_y':'cast',
                 'director_y':'director',
                 'listed_in_y':'listed_in',
                 'country_y':'country'},axis=1,inplace=True)


# In[50]:


nf_final.head(5)


# In[51]:


nf2=nf_final.copy()


# In[52]:


nf2.head(2)


# In[53]:


nf2.info()


# In[54]:


# replace nan values


# In[55]:


nf2['rating'].mode()


# In[56]:


nf2['rating'].fillna(value='TV-MA',inplace=True)


# In[57]:


nf2['duration'].fillna(value='not_know',inplace=True)


# In[58]:


nf2['director'].replace(['nan'],['unknown director'],inplace=True)


# In[59]:


nf2['country'].replace(['nan'],['unknown country'],inplace=True)


# In[60]:


nf2['cast'].replace(['nan'],['unknown cast'],inplace=True)


# In[61]:


#adding new columns of date added (month,month_name,year)

nf2['month_added']=nf2['date_added'].dt.month
nf2['month_name_added']=nf2['date_added'].dt.month_name()
nf2['year_added'] = nf2['date_added'].dt.year


# In[62]:


nf2.info()


# In[63]:


nf2.head(2)


# # we can fill the nan values of unnested data

# In[64]:


nf3=nf.copy()


# In[65]:


nf3['rating'].fillna(value='TV-MA',inplace=True)
nf3['duration'].fillna(value='not_know',inplace=True)
nf3['director'].fillna(value='unknown director',inplace=True)
nf3['country'].fillna(value='unknown country',inplace=True)
nf3['cast'].fillna(value='unknown cast',inplace=True)


# In[66]:


nf3[nf3.duplicated()]


# In[67]:


nf3.isnull().sum()


# In[68]:


nf3.info()


# In[69]:


nf3['date_n']=pd.to_datetime(nf3['release_year'])


# In[70]:


nf3=nf3.drop('date_n',axis=1)


# In[71]:


nf3.head(2)


# In[72]:


# in which year number of tv shows and movies were released ?


# In[73]:


nf_m=nf3['date_added'].dt.year.value_counts()
nf_m


# In[74]:


nf_m.plot(kind='bar')


# In[75]:


# no of movies and tvShow in the data set


# In[76]:


nf3.groupby('type').type.count()


# In[83]:


sns.countplot(nf.groupby('type').type.count())


# In[84]:


sns.countplot(nf['type'])


# In[85]:


#in which year highest number of tv show and movies released ?


# In[86]:


nf_m=nf3['date_added'].dt.year.value_counts()
nf_m


# In[87]:


nf_m.plot(kind='bar')


# In[88]:


plt.figure(figsize=(6,4))
sns.histplot(x='release_year',hue='type',data=nf3)


# In[89]:


# show all the movies that were release in year 2019?


# In[90]:


nf3[(nf3['type']=='Movie')& (nf3['release_year']==2019)]


# In[91]:


# show only the title of all tv shows that were released in india only


# In[92]:


nf3[(nf3['type']=='TV Show') & (nf3['country']=='India')]['title'].reset_index()


# In[93]:


# show top 10 director who gave highest no of tv shows ?


# In[94]:


nf_maxd=nf3['director'].value_counts().head(10)
nf_maxd


# In[95]:


nf_maxd.plot(kind='bar')


# In[96]:


# find all the recordes where (category is movie and type is comedies) or (country is india)? 


# In[97]:


nf_record=nf3[(nf3['type']=='Movie') & (nf3['listed_in']=='Comedies')|(nf3['country']=='India')]


# In[98]:


nf_record.head(2)


# In[99]:


# how many movies/show  only tom crues was cast?


# In[100]:


nf3[nf3['cast']=='Tom Cruise']


# In[101]:


nf3[nf3['cast'].str.contains('Tom Cruise')]


# In[102]:


# how many diff rating defined by netflix?


# In[103]:


nf3.rating.value_counts()


# In[104]:


nf3['rating'].nunique()


# In[105]:


plt.figure(figsize=(8,6))
sns.countplot(x=nf3["rating"])
plt.xticks(rotation=90)
plt.show()


# In[106]:


import seaborn as sns 
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
sns.countplot(x='rating',hue='type',data=nf3)


# insight- maximum no of rating is given to the TV-MA or least to UR

# In[107]:


# how many movies got the 'tv-ma' rating in india?


# In[108]:


nf3[(nf3['type']=='Movie')&(nf3['rating']=='TV-MA') & (nf3['country']=='India')].shape


# In[109]:


nf_rating=nf3[(nf3['type']=='Movie')&(nf3['rating']=='TV-MA') & (nf3['country']=='India')]


# In[110]:


nf_rating.head(5)


# In[111]:


# what is the maximum  of shows by country name ?


# In[112]:


nf4=nf3.copy()


# In[113]:


nf_tvshow=nf4[nf4['type']=='TV Show']


# In[114]:


nf_tvshow.head(2)


# In[115]:


plot_tv=nf_tvshow.country.value_counts()
plot_tv.head(10)


# In[116]:


plot_tv.head(10).plot(kind='bar')


# insight-united states has created maximum no. of tvshows

# In[117]:


# find the latest movie/show in the netflix when it is added ?


# In[118]:


nf4.head(2)


# In[119]:


nf4.sort_values(by='date_added',ascending=False).head(5)


# In[ ]:





# In[120]:


# find the instances where 
#type is 'movie' and type is 'drama'.
# or
# type is 'tv show & type is 'kids tv


# In[121]:


nf_k=nf4[(nf4['type']=='Movie') & (nf4['listed_in']=='Dramas')|(nf4['type']=='TV Show')&(nf4['listed_in']=='Kids TV')]


# In[122]:


nf_k.head(2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[123]:


b=(nf2.loc[nf2["type"]=="TV Show"])["listed_in"].value_counts()
x=b.index
y=b.values


# In[124]:


plt.figure(figsize=(8,6))
plt.bar(x,
y)
plt.xticks(rotation=90,fontsize=12)
plt.show


# insight-maximum listed in interbational tv shows

# In[125]:


nf2.head(2)


# In[126]:


import plotly.express as px


# In[132]:


# analyzing the content produce on netflix based on year


# In[133]:


nf6=nf3[['type','release_year']]


# In[135]:


nf7=nf6.groupby(['release_year','type']).size().reset_index(name='Total Count')


# In[138]:


nf7


# # For correlation: Pairplots

# In[213]:


nf_pairplot=sns.pairplot(nf)


# insight - maximum data id present between 2000 to 2021

# In[142]:


nf3.head(2)


# # For categorical variable(s): Boxplot

# In[155]:


sns.boxplot(data=nf, x="date_added", y="release_year", hue="type")


# In[79]:


sns.boxplot(x=nf3["release_year"])


# In[85]:


sns.kdeplot(data=nf3, x="release_year", bw_adjust=.2)


# insight - the maximum release year of movie and tv show in 2019

# In[77]:


nf3.info()


# # 6. Insights based on Non-Graphical and Visual Analysis

# # 6.1 Comments on the range of attributes

# insights     -- On the basic of the data there is 8807 show_id  in  this,there is 6173 number of directors                                       and out of  this 4528 are unique and 17 type of ratings and in  duration 220 is a unique value present in them .
# 
#           --frequency of director is 19 , show id freq=1 ,cast freq=19
#           --count according to the released year is 8807
#           --released year mean is 2014
#           --maximum data of the released year present in 2021
#           --top show_id =s1
#           --maximum no is movies in data set
#           --top director =rajiv chilaka
#            --maximum no of mvies is created by the united  states
#           --maximum rating is given to the TV-MA
#           --we have maximum no of null value is present in the   director column which is 2634
#           --also we have null value in cast,country date_added,rating,duration
#           --shape of a data is (8807,12) which means there is 8807 no. of rows and 12 no of columns
#           --type of a data is – panda.core.frame.dataframe
#            --there is 69.6% of movie present in the data and 30.3 % of Tv-Shows
# 
#                 

# # 6.2-Comments on the distribution of the variables and relationship between them

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# 6.2-Comments on the distribution of the variables and relationship between them
# insightes -- On the basis of heat map the maximum no of null values is present in the director column
# 
#        --there is a maximum no of -drama and international movies present in the data
#        --maximum no of movies released in the 2019 and no of movies released is 2016
#                          And minimum no of movies released in 2010 and no is 1
#        --the no. movies present in the data set is 6131 and tv_shows are 2676
#        --the no. of movies released in the year 2019 is 633 movies
#        --the director which direct maximum no of movies is (Rajiv Chilaka)
#        -- the only 2 movies created by tom cruise
#        --maximum ratting is given to the –3211
#        -- minimum ratting is given to the-  66min
#        --there is 215 no of movies in india which has maximum rating(TV-MA)
#        --maximum no of tvshow is by the United States
#        --the latest movie is present in the data -Title =Dick jonson is dead
#                                                  Country=united states
#                                                  Release date=2021-09-25
#                                                     Duration =90 min
#                                                   Listed_in =documentaries
