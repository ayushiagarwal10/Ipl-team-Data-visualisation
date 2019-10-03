
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


# In[2]:


data1 = pd.read_csv("deliveries.csv")
data2 = pd.read_csv("matches.csv")


# In[3]:


data1.head()
data1.columns


# In[4]:


data1.shape


# In[5]:


data2.head()


# In[6]:


data2.shape


# In[7]:


categorical_data1 = data1.dtypes[data1.dtypes == "object"].index
print(categorical_data1)


# In[8]:


data2.info()


# In[9]:


data1.apply(lambda x:sum(x.isnull()))


# In[10]:


data1 = data1.fillna('0')


# In[11]:


data1.apply(lambda x:sum(x.isnull()))


# In[12]:


data2.apply(lambda x:sum(x.isnull()))


# In[13]:


data2 =data2.drop('umpire3',axis =1)


# In[14]:


data2['umpire1'].value_counts()


# In[15]:


data2['umpire1'] =data2['umpire1'].fillna('HDPK Dharmasena ')


# In[16]:


data2['umpire2'].value_counts()


# In[17]:


data2['umpire2']=data2['umpire2'].fillna("SJA Taufel")


# In[18]:


data2['city'].value_counts()


# In[19]:


data2['city']=data2['city'].fillna("Mumbai")


# In[20]:


data1['match_id']


# In[21]:


data2['season'].value_counts()


# In[22]:


sns.countplot(x=data2['season'], data=data2)


# In[23]:


data2['city'].value_counts()


# In[24]:


plt.figure(figsize=(15,7))
sns.countplot(x=data2['city'], data=data2)
plt.xticks(rotation = 'vertical')


# In[25]:


data2['toss_winner'].value_counts()


# In[26]:



plt.figure(figsize=(15,7))
sns.countplot(x=data2['toss_winner'], data=data2)
plt.xticks(rotation = 'vertical')                  


# In[27]:


data2['result'].value_counts()


# In[28]:


data2['city'].value_counts().plot(kind='bar', color='blue')


# In[29]:


data2['toss_winner'].value_counts()


# In[30]:


data2['winner'].value_counts()


# In[31]:



plt.figure(figsize=(15,7))
sns.countplot(x=data2['toss_winner'],hue=data2['toss_decision'],data=data2)
plt.xticks(rotation='vertical')


# In[32]:


plt.figure(figsize=(15,7))
sns.countplot(x=data2['winner'],hue=data2['toss_decision'],data=data2)
plt.xticks(rotation='vertical')


# In[33]:



plt.figure(figsize=(12,7))
temp=data2['toss_decision'].value_counts()
sizes = (np.array((temp / temp.sum())*100))
plt.pie(sizes, labels=(np.array(temp.index)),colors=['lightgreen', 'lightblue'],
        autopct='%1.1f%%',shadow=True, startangle=90,explode=(0,0.03))
plt.title("Toss decision percentage")
plt.show()
          


# In[34]:


plt.figure(figsize=(12,7))
temp=data2[data2['toss_winner']==data2['winner']]
sizes = (len(temp),(data2.shape[0]-len(temp)))
labels = ['toss_winner wins match','toss_winner loses match']
plt.pie(sizes, labels=labels,colors=['yellow', 'pink'],
        autopct='%1.2f%%',shadow=True, startangle=90,explode=(0,0.03))
plt.title("toss wins vs toss loss")
plt.show()


# In[35]:


temp1 = data2
temp1['Toss_Winner_is_Match_Winner'] = 'no'
temp1['Toss_Winner_is_Match_Winner'].loc[data2['toss_winner']==data2['winner']] = 'yes'
plt.figure(figsize=(15,7))
sns.countplot(x='toss_winner', hue='Toss_Winner_is_Match_Winner', data=temp1)
plt.xticks(rotation='vertical')
plt.show()


# In[36]:


temp1['Toss_Winner_is_Match_Winner'].value_counts()


# In[37]:


bowlers = data2[['id','season']].merge(data1, right_on='match_id',left_on='id',how='left').drop('id',axis=1)
bowlers.head()


# In[38]:


bowlers.info()


# In[39]:


total_wickets = bowlers[bowlers.dismissal_kind !='0']
total_wickets['dismissal_kind'].value_counts()


# In[40]:



plt.figure(figsize=(12,7))
sns.countplot(x=total_wickets['dismissal_kind'],data=total_wickets)
plt.xticks(rotation='vertical')


# In[41]:


matches_played_byteams=pd.concat([data2['team1'],data2['team2']])


# In[42]:


matches_played_byteams.head()


# In[43]:


matches_played_byteams=matches_played_byteams.value_counts().reset_index()
matches_played_byteams.columns=['Team','Total Matches']
matches_played_byteams['wins']=data2['winner'].value_counts().reset_index()['winner']
matches_played_byteams.set_index('Team',inplace=True)


# In[44]:


trace1 = plt.Bar(x=matches_played_byteams.index,
                y=matches_played_byteams['Total Matches'],
                name='Total Matches')

trace2 = plt.Bar(x=matches_played_byteams.index,
                y=matches_played_byteams['wins'],
                name='Matches Won')

data = [trace1, trace2]
layout = plt.Layout(barmode='stack')


# In[45]:


plt.figure(figsize=(15,6))
temp = sns.countplot(x='season',data=total_wickets)
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+.20, i.get_height()),fontsize=15)


# In[46]:


total_wickets['bowler'].value_counts()


# In[47]:


plt.figure(figsize=(25,16))
temp = total_wickets['bowler'].value_counts()[:20].plot(kind='bar', color=sns.color_palette('autumn',10))
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+.20, i.get_height()),fontsize=15)


# In[48]:


batsmen = data2[['id','season']].merge(data1, right_on='match_id',left_on='id',how='left').drop('id',axis=1)
batsmen.head()


# In[65]:


temp = batsmen.groupby('batsman')['batsman_runs'].sum().reset_index()
temp = temp.sort_values('batsman_runs', ascending=False)[:10]
temp.reset_index(drop=True)


# In[66]:


temp = temp.plot(kind='bar', x='batsman', y='batsman_runs', width=0.8, color=sns.color_palette('summer',20))
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)

fig=plt.gcf()
fig.set_size_inches(14,6)
plt.show()


# In[67]:


total_wickets.columns


# In[78]:


temp = batsmen.groupby('season')['total_runs'].sum()
temp.head()


# In[79]:


temp = temp.plot(kind='bar', x='season', y='total_runs', width=0.8, color=sns.color_palette('summer',20))
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+0.20, i.get_height()),fontsize=15)

fig=plt.gcf()
fig.set_size_inches(14,6)
plt.show()


# In[83]:


boundary = ['4']
fours = batsmen[batsmen['batsman_runs'].isin(boundary)]
fours['batsman'].value_counts()[:10]


# In[91]:


plt.figure(figsize=(25,16))
temp = fours['batsman'].value_counts()[:10].plot(kind='bar', color=sns.color_palette('autumn',10))
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+.20, i.get_height()),fontsize=15)


# In[93]:


plt.figure(figsize=(25,16))
temp=sns.countplot(x=fours['season'],data=fours)
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+.20, i.get_height()),fontsize=15)
   


# In[95]:


boundary = ['6']
sixes = batsmen[batsmen['batsman_runs'].isin(boundary)]
sixes['batsman'].value_counts()[:10]


# In[96]:


plt.figure(figsize=(25,16))
temp = sixes['batsman'].value_counts()[:10].plot(kind='bar', color=sns.color_palette('autumn',10))
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+.20, i.get_height()),fontsize=15)


# In[97]:


plt.figure(figsize=(25,16))
temp=sns.countplot(x=sixes['season'],data=sixes)
for i in temp.patches:
    temp.annotate(format(i.get_height()),(i.get_x()+.20, i.get_height()),fontsize=15)
   


# In[101]:


a=sixes.groupby("season")["batsman_runs"].agg(lambda four : four.sum()).reset_index()

b=fours.groupby("season")["batsman_runs"].agg(lambda six: six.sum()).reset_index()

boundaries=a.merge(b,left_on='season',right_on='season',how='left')


# In[99]:


boundaries.head()


# In[102]:


boundaries.plot(x='batsman_runs_x', y='batsman_runs_y', marker='o')


# In[103]:


boundaries.set_index('season')[['batsman_runs_x','batsman_runs_y']].plot(marker='o',color=['red','green'])
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

