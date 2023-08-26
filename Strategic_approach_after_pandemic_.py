#!/usr/bin/env python
# coding: utf-8

# ## [1. Initial setup and modules/packages loading](#InitialSetup)
# * [**1.1 - Import libraries &  dataframe**](#import)
# * [**1.2 - Data correction**](#Datacorrection)
# * [**1.3 - Definition of cluster perspectives and selection of variables**](#Perspective)
# * [**1.4 - Standardize Vision Dataframes**](#Standard)
# 
# 
# ## [2. Data Understanding](#DataUnderstanding)
# 
# 
# ## [3. Data Preparation](#DataPreparation)
# * [**3.1. Checking duplicated lines**](#Duplicates)
#     * [3.1.1. Make Copies of the Original dataset](#Copies)
#     * [3.1.2. Replacing Values](#ReplacingValues)
#     * [3.1.3. Set an index](#SetIndex)
# * [**3.2 Data Cleaning**](#DataCleaning)
#     * [3.2.1. Checking Missing Values](#CheckingMissingValues)
#     * [3.2.2 Filter out missing values](#fmv)
#     * [3.2.3. Fill in missing values and drop remaining rows](#fmvd)
#     * [3.2.4. Detect and Treat the outliers](#dto)
#     * [3.2.5 Skewness And Kurtosis](#sk)
#     * [3.2.6. Check Correlations](#cc)
# 
# 
# ## [4. Feature Engineering](#fe)
# * [**4.1. Changing Variables**](#cv)
# 
# 
# ## [5. Modeling](#m)
# * [**5.1. RFM Model**](#rfm)
# 
# 
# ## [6. Data Visualization](#dv)
# * [**6.1. General analysis**](#ga)
# 
# 
# ## [7. Recommendation  systems](#rs)
# 

# <a class="anchor" id="InitialSetup">
#     
# # 1. Initial setup and modules/packages loading

# In[17]:


# Packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import ipympl
from mpl_toolkits.mplot3d import Axes3D
import squarify
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

# Force widgets to be shown on notebook (may need permissions from the user)
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


# Global definitions
baseFolder = os.getcwd()
exportsFolder = baseFolder + os.sep +'Exports' + os.sep


# In[19]:


subPlots_Title_fontSize = 12
subPlots_xAxis_fontSize = 10
subPlots_yAxis_fontSize = 10
subPlots_label_fontSize = 10
heatmaps_text_fontSize = 8

plots_Title_fontSize = 14
plots_Title_textColour = 'black'

plots_Legend_fontSize = 12
plots_Legend_textColour = 'black'


# <a class="anchor" id="DataUnderstanding">
#     
# # 2. Data Understanding

# In[20]:


# Add Attractions sheet to Reviews
dseurotop = pd.ExcelFile('EuropeTop100Attractions_ENG_20190101_20210821.xlsx')

# 'extrai' aba 1
dfReviews = pd.read_excel(dseurotop, sheet_name='Reviews')

# 'extrai' aba 2
dfAttractions = pd.read_excel(dseurotop, sheet_name='Attractions')


# In[22]:


# Changing first column name
dfAttractions.rename(columns = {'ID':'localID'},
                     inplace = True)

# merge tabs
dseurotopattractions = dfReviews.merge(dfAttractions, on='localID', how='left', validate='m:1')

# Rename the file to eurotop
dseurotop = dseurotopattractions


# In[23]:


# Display info
dseurotop.info()


# In[24]:


# Display top 10 rows
dseurotop.head(10)


# In[25]:


# Summary statistics for all variables
dseurotop.describe(include='all', datetime_is_numeric=True).T


# General observations:
# 
# - There are a total of 100 unique attractions with a total of 92120 reviews
# - The most evaluated attraction is MAG001 with 8309 reviews
# - There are 5 different types of trips and the most frequent is Couples with 31702 presences
# - The review varies between 1 and 5
# - All the reviews are in one Language - English
# 
# Considerations on data quality:
# - *extractionDate*, *reviewWritten* and *reviewVisited* are date/time variables and are recognize as such
# - *localID, userName, userLocation, tripType, reviewLanguage, reviewFullText, Name, Country, ISO* are a categorical variables
# - *userLocation, reviewVisited* have a high proportion of missing values
# - *positionOnRanking, sitesOnRanking, userContributions* seem to have outliers

# In[26]:


# Check the differences in the summary statistics for all variables
dseurotop.describe(include='all', datetime_is_numeric=True).T


# In[27]:


# Check localID
dseurotop['localID'].value_counts()


# In[28]:


#Check summary statistics for categorical variables

dseurotop.describe(include = ['O'])


# In[29]:


#Check number of values

dseurotop.count()


# In[30]:


#Check distribution of reviewRating 

dseurotop['reviewRating'].value_counts()


# In[31]:


#Check distribution of globalRating 

dseurotop['globalRating'].value_counts()


# In[32]:


#Check distribution of trip Type

dseurotop['tripType'].value_counts()


# In[33]:


#Check first Review date

dseurotop['reviewWritten'].min()


# In[34]:


#Check last Review date

dseurotop['reviewWritten'].max()


# In[35]:


#Check last Visit date

dseurotop['reviewVisited'].min()


# In[36]:


#Check last Visit date

dseurotop['reviewVisited'].max()


# <a class="anchor" id="DatasetPreparation">
# 
# # 3. Data Preparation

# </a>
# 
# After performing Data Understanding this Project will step up to the Data Preparation phase. 
# In this phase, we are going to perform some useful commands as preprocessing tasks to prepare and clean the dataset.
# After this data preparation we will be able to better approach the data.
# 
# Following steps:
# - Check Duplicated Data; <br>
# - Make copies of the original dataset; <br>
# - Replace values; <br>
# - Data Cleaning; <br>

# <a class="anchor" id="Duplicates">
# 
# ## 3.1. Checking duplicated lines

# In[37]:


# Check for duplicates
dseurotop.duplicated().sum()


# In[38]:


# Check duplicated lines
dseurotop[dseurotop.duplicated(keep=False)]


# <a class="anchor" id="Copies">
# 
# ## 3.1.1. Make Copies of the Original dataset

# In[39]:


dseurotop_prep = dseurotop.copy()
dseurotop_prep.head()


# <a class="anchor" id="ReplacingValues">
# 
# ## 3.1.2. Replacing Values

# Looking into the dataset information the group understood that there are values which will need to be replaced.
# 
# - LocalID Column
# 
# In the Understading phase, the group have grouped from the 'csv' dataset two sheets ('reviews' and 'attractions') and connected them within the LocalID. But, after some analysis, were identified that there are 2 different types of LocalID which couldnt be connected with any attraction in a first place, the 'genis' and 'u' localIDs. Because of this, we have done a deep analysis throughout other information available in the review sheet, such as comments on 'reviewFullText', 'sitesOnRanking' and 'positionOnRanking', those info pointed for the following conclusion:
# 
#     - genis = MAG005
#     - u = MAG006
#     
# For more information, check analysis below:

# In[40]:


cols = ['localID']
dseurotop_prep[cols].apply(pd.Series.value_counts)


# In[41]:


#Cheking the localID which does not contains "MAG"

dseurotop_notmag = dseurotop_prep[~dseurotop_prep['localID'].str.contains('MAG')]
cols_1 = ['localID']
dseurotop_notmag[cols_1].apply(pd.Series.value_counts)


# In[42]:


# Checking if the PositionOnRanking is the same for every line
dseurotop_notmag['positionOnRanking'].value_counts()


# In[43]:


# Checking if the sitesOnRanking is the same for every line
dseurotop_notmag['sitesOnRanking'].value_counts()


# In[44]:


dseurotop_prep[dseurotop_prep['localID'].str.contains('u')].head()


# In[45]:


#looking into a dataset with only LocalID genis, in order to check specific descripitons on 'reviewFullText'
dseurotop_genislocalID = dseurotop_notmag[dseurotop_notmag['localID'].str.contains('genis')]

#looking for specific words, regarding our check into the 'csv' dataset
dseurotop_genislocalID[dseurotop_genislocalID['reviewFullText'].str.contains('Staromestske')]


# In[46]:


#looking for specific words, regarding our check into the 'csv' dataset
dseurotop_genislocalID[dseurotop_genislocalID['reviewFullText'].str.contains('Prague')]


# In[47]:


#looking into a dataset with only LocalID 'u', in order to check specific descripitons on 'reviewFullText'
dseurotop_ulocalID = dseurotop_notmag[dseurotop_notmag['localID'].str.contains('u')]

#looking for specific words, regarding our check into the 'csv' dataset
dseurotop_ulocalID[dseurotop_ulocalID['reviewFullText'].str.contains('Edinburgh')]


# In[48]:


#looking for specific words, regarding our check into the 'csv' dataset
dseurotop_ulocalID[dseurotop_ulocalID['reviewFullText'].str.contains('castle')]


# In[49]:


#Conclusion> Adjusting LocalID in the original dataset

# "u" = 'MAG006' = Name: Staromestske namesti = Country:Czech Republic = ISO:CZ
# "genis" = 'MAG005' = Name: Edinburgh Castle = Country:Scotland = ISO:UK


dseurotop_prep['localID'].replace(to_replace = 'u',value = 'MAG006', inplace = True)
dseurotop_prep['localID'].replace(to_replace = 'genis',value = 'MAG005', inplace = True)


# In[50]:


#Drop the columns merge before
dseurotop_prepreset = dseurotop_prep.drop(['Name','Country','ISO'], axis=1)

#Merge again with correct LocalIDs
dseurotop_prep1 = dseurotop_prepreset.merge(dfAttractions, on='localID', how='left', validate='m:1')

#Original Dataset Updated
dseurotop_prep1.info()


# <a class="anchor" id="SetIndex">
# 
# ## 3.1.3. Set an index

# In[51]:


dseurotop_prep1.set_index('localID', inplace = True)
dseurotop_prep1.info()


# <a class="anchor" id="DataCleaning">
# 
# ## 3.2  Data Cleaning
# 
# </a>
# 
# 
# - Check missing values; <br>
# - Filter out missing values; <br>
# - Fill in missing values and drop remaining rows; <br>
# - Convert datatypes; <br>
# - Check Correlations; <br>
# - Detect and remove outliers; <br>

# <a class="anchor" id="cmv">
# 
# ### 3.2.1. Checking Missing Values

# In[52]:


# Check for missing values 
dseurotop_prep1.isnull().sum()


# In[53]:


# Check for missing values %
dseurotop_prep1.isna().sum()/len(dseurotop_prep)*100


# <a class="anchor" id="fmv">
# 
# ### 3.2.2 Filter out missing values

# In[54]:


# drop any row containing missing values
dseurotopRows_no_missing = dseurotop_prep1.dropna()
print("The number of rows in the original dseurotop dataset is", dseurotop_prep1.shape[0])
print("The number of rows in the dseurotop dataset with no missing values is", dseurotopRows_no_missing.shape[0])
print("There were", dseurotop_prep1.shape[0]-dseurotopRows_no_missing.shape[0], "rows containing missing values.")


# In[55]:


# drop rows where all cells in that row are NA
dseurotopRows_cleaned = dseurotop_prep1.dropna(how='all')
print("The number of rows in the tugas dataset with no missing rows is", dseurotopRows_cleaned.shape[0])


# In[56]:


# Filtering on missing values (mv)
import matplotlib.ticker as mtick
dseurotop_mv=dseurotop_prep1.copy()
dseurotop_mv = pd.DataFrame(dseurotop_prep1.isna().sum(),
                        columns = ['Missings']).sort_values(by = 'Missings', ascending = False)

dseurotop_mv = dseurotop_mv[dseurotop_mv['Missings'] > 0].copy()
dseurotop_mv


# In[57]:


dseurotop_mv_perc = pd.DataFrame(dseurotop_prep1.isna().sum()/dseurotop_prep1.shape[0],
                        columns = ['Missings (%)']).sort_values(by = 'Missings (%)', ascending = False)
dseurotop_mv_perc = dseurotop_mv_perc[dseurotop_mv_perc['Missings (%)'] > 0].copy()
dseurotop_mv_perc


# In[58]:


import matplotlib.ticker as mtick

dseurotop_mv = pd.DataFrame(dseurotop_prep1.isna().sum()/dseurotop_prep1.shape[0],
                        columns = ['Missings (%)']).sort_values(by = 'Missings (%)', ascending = False)
dseurotop_mv = dseurotop_mv[dseurotop_mv['Missings (%)'] > 0].copy()

plt.figure(figsize = (20,6.5))

ax = sns.barplot(data = dseurotop_mv, x = dseurotop_mv.index, y = 'Missings (%)', palette="husl" )
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

for p in ax.patches:
    ax.annotate(format(p.get_height()*100, '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


plt.title('Missing Values (%)')
plt.show()


# By calculating the relative frequency of missing values is possible to see missing in "username" column representing 0.02% of the dataset, for this small percentage we believe that we can exclude those variables. 
# Moreover, for the missing values in the column "reviewVisited", even it is much less than 1% of the dataset, the group has decided to keep them using  "reviewWritten" column.
# 
# Because there are many missing values for tripType and userLocation and since it will make a difference in our analysis, we decided to attribute to tripType missing values the word 'Other' and userLocation missing values the word 'Unknown'.

# <a class="anchor" id="fmvd">
# 
# ### 3.2.3. Fill in missing values and drop remaining rows

# In[59]:


# Drop missing values from column "userName"
dseurotop_fill = dseurotop_prep1.dropna(subset=["userName", "reviewVisited"])
dseurotop_fill.shape


# In[60]:


#Filling missing values: userLocation with Unknown & tripType with Other

dseurotop_prep2 = dseurotop_fill.fillna({
    'userLocation': "Unknown",
    'tripType': "Other"    
})
dseurotop_prep2


# In[61]:


dseurotop_prep2.isna().sum().sum()


# In[62]:


dseurotop_prep2.shape


# <a class="anchor" id="dto">
# 
# ### 3.2.4. Detect and Treat the outliers

# In[63]:


dseurotop_prep2.shape


# In[64]:


dseurotop_prep2.describe().T


# From the statiscal summary above, when we look and compare the minimum values against the maximum values, but also through the standard deviation, we can instantly tell that we have some outliers in the "totalReviews", "usercontributions","sitesOnRanking" and finally "positionOnRanking" variables. We will get deeper with the graphs help.

# In[65]:


numerical=dseurotop_prep2.select_dtypes(include = [np.number]).columns.tolist()

fig, ax = plt.subplots(2, 3, figsize = (23,11))
for var, subplot in zip(dseurotop_prep2[numerical], ax.flatten()):
    g = sns.boxplot(data = dseurotop_prep2,
                 x = var,
                 color = 'lightblue',
                 ax = subplot)

plt.rc('axes', labelsize = subPlots_label_fontSize)
fig.suptitle("Boxplots of all numeric variables", fontsize=plots_Title_fontSize);


# Just as we thought, with the help of the boxplots we can see that the already mentioned variables have some outliers that we need to look and handle closely.
# Also, although it seems that we have some outliers in globalRating and reviewRating, we will not treat them as such, for they are rating variables.

# Position on Ranking
# 
# Although it seems that we have some outliers in globalRating and reviewRating, we will not treat them as such, for they are rating variables.

# In[66]:


#positionOnRanking
dseurotop_outliers = dseurotop_prep2.copy()

Q1_positionOnRanking = dseurotop_outliers["positionOnRanking"].quantile(0.25)
Q3_positionOnRanking = dseurotop_outliers["positionOnRanking"].quantile(0.75)

IQR_positionOnRanking = Q3_positionOnRanking - Q1_positionOnRanking

Lower_positionOnRanking = Q1_positionOnRanking - (1.5 * IQR_positionOnRanking)
Upper_positionOnRanking = Q3_positionOnRanking + (1.5 * IQR_positionOnRanking)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))
sns.boxplot(x=dseurotop_outliers["positionOnRanking"], data=dseurotop_outliers, color = 'lightblue',ax = ax[1])
dseurotop_outliers["positionOnRanking"].plot(kind = 'hist', 
                      bins = 30, 
                      title = 'Position On Ranking', 
                      color = 'lightblue',
                      ec = 'black', ax=ax[0])

ax[1].title.set_text('Boxplot representing the Position On Ranking')
ax[0].title.set_text('Histogram representing the Position On Ranking')

plt.show()

#positionOnRanking OUTLIERS SHAPE
positionOnRanking_shape = dseurotop_outliers[['positionOnRanking']]
positionOnRanking_shape = positionOnRanking_shape.sort_values(by=['positionOnRanking'], ascending=False)

UC = positionOnRanking_shape.loc[positionOnRanking_shape['positionOnRanking']>Upper_positionOnRanking].shape[0]
UC


# It seems we have some outliers but for now we'll just take any value > than 33 before the biggest gap.

# In[67]:


outliers = dseurotop_prep2[(dseurotop_prep2.positionOnRanking > 33)].copy()
dseurotop_prep2 = dseurotop_prep2[(dseurotop_prep2.positionOnRanking < 33) | (dseurotop_prep2.positionOnRanking.isnull())]


# In[68]:


sns.boxplot(x = 'positionOnRanking', data = dseurotop_prep2, color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Position On Ranking Boxplot", fontsize=plots_Title_fontSize);


# In[69]:


print("We deleted " + str(91410 - dseurotop_prep2.shape[0]) + " observations from the dseurotop dataset, that is " + str(((91410 - dseurotop_prep2.shape[0])/91410)*100) + "% of the total observations.")


# In[70]:


#sitesOnRanking
dseurotop_outliers = dseurotop_prep2.copy()

Q1_sitesOnRanking = dseurotop_outliers["sitesOnRanking"].quantile(0.25)
Q3_sitesOnRanking = dseurotop_outliers["sitesOnRanking"].quantile(0.75)

IQR_sitesOnRanking = Q3_sitesOnRanking - Q1_sitesOnRanking

Lower_sitesOnRanking = Q1_sitesOnRanking - (1.5 * IQR_sitesOnRanking)
Upper_sitesOnRanking = Q3_sitesOnRanking + (1.5 * IQR_sitesOnRanking)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))
sns.boxplot(x=dseurotop_outliers["sitesOnRanking"], data=dseurotop_outliers, color = 'lightblue',ax = ax[1])
dseurotop_outliers["sitesOnRanking"].plot(kind = 'hist', 
                      bins = 30, 
                      title = 'Sites On Ranking', 
                      color = 'lightblue',
                      ec = 'black', ax=ax[0])

ax[1].title.set_text('Boxplot representing the Sites On Ranking')
ax[0].title.set_text('Histogram representing the Sites On Ranking')

plt.show()

#sitesOnRanking OUTLIERS SHAPE
sitesOnRanking_shape = dseurotop_outliers[['sitesOnRanking']]
sitesOnRanking_shape = sitesOnRanking_shape.sort_values(by=['sitesOnRanking'], ascending=False)

UC = sitesOnRanking_shape.loc[sitesOnRanking_shape['sitesOnRanking']>Upper_sitesOnRanking].shape[0]
UC


# Sites on Ranking- We see that we have two outliers, so we will remove them and check how many observations we are deleting taking into consideration the dataset size.

# In[71]:


outliers = dseurotop_prep2[(dseurotop_prep2.sitesOnRanking > 3200)].copy()
dseurotop_prep2 = dseurotop_prep2[(dseurotop_prep2.sitesOnRanking < 3200) | (dseurotop_prep2.sitesOnRanking.isnull())]


# In[72]:


sns.boxplot(x = 'sitesOnRanking', data = dseurotop_prep2, color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Sites On Ranking Boxplot", fontsize=plots_Title_fontSize);


# In[73]:


print("We deleted " + str(91410 - dseurotop_prep2.shape[0]) + " observations from the dseurotop dataset, that is " + str(((91410 - dseurotop_prep2.shape[0])/91410)*100) + "% of the total observations.")


# In[74]:


#totalReviews
dseurotop_outliers = dseurotop_prep2.copy()

Q1_totalReviews = dseurotop_outliers["totalReviews"].quantile(0.25)
Q3_totalReviews = dseurotop_outliers["totalReviews"].quantile(0.75)

IQR_totalReviews = Q3_totalReviews - Q1_totalReviews

Lower_totalReviews = Q1_totalReviews - (1.5 * IQR_totalReviews)
Upper_totalReviews = Q3_totalReviews + (1.5 * IQR_totalReviews)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))
sns.boxplot(x=dseurotop_outliers["totalReviews"], data=dseurotop_outliers, color = 'lightblue',ax = ax[1])
dseurotop_outliers["totalReviews"].plot(kind = 'hist', 
                      bins = 30, 
                      title = 'Total Reviews', 
                      color = 'lightblue',
                      ec = 'black', ax=ax[0])

ax[1].title.set_text('Boxplot representing the Total Reviews')
ax[0].title.set_text('Histogram representing the Total Reviews')

plt.show()

#Total Reviews OUTLIERS SHAPE
totalReviews_shape = dseurotop_outliers[['totalReviews']]
totalReviews_shape = totalReviews_shape.sort_values(by=['totalReviews'], ascending=False)

UC = totalReviews_shape.loc[totalReviews_shape['totalReviews']>Upper_totalReviews].shape[0]
UC


# As we understand the total reviews per city, and the one showed as an outliers is the first one on the ranking of visits so after some discussion we have decided to keep this outlier.
# 
# 

# As we understand the total reviews per city, and the one showed as an outliers is the first one on the ranking of visits, we have decided to keep the lines.

# In[75]:


sns.boxplot(x = 'totalReviews', data = dseurotop, color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Total Reviews Boxplot", fontsize=plots_Title_fontSize);


# In[76]:


#User contributions
dseurotop_outliers = dseurotop_prep2.copy()

Q1_userContributions = dseurotop_outliers["userContributions"].quantile(0.25)
Q3_userContributions = dseurotop_outliers["userContributions"].quantile(0.75)

IQR_userContributions = Q3_userContributions - Q1_userContributions

Lower_userContributions = Q1_userContributions - (1.5 * IQR_userContributions)
Upper_userContributions = Q3_userContributions + (1.5 * IQR_userContributions)


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))
sns.boxplot(x=dseurotop_outliers["userContributions"], data=dseurotop_outliers, color = 'lightblue',ax = ax[1])
dseurotop_outliers["userContributions"].plot(kind = 'hist', 
                      bins = 30, 
                      title = 'User Contributions', 
                      color = 'lightblue',
                      ec = 'black', ax=ax[0])

ax[1].title.set_text('Boxplot representing the User Contributions')
ax[0].title.set_text('Histogram representing the User Contributions')

plt.show()

#User Contributions OUTLIERS SHAPE
userContributions_shape = dseurotop_outliers[['userContributions']]
userContributions_shape = userContributions_shape.sort_values(by=['userContributions'], ascending=False)

UC = userContributions_shape.loc[userContributions_shape['userContributions']>Upper_userContributions].shape[0]
UC


# In[77]:


outliers = dseurotop_prep2[(dseurotop_prep2.userContributions > 20000)].copy()
dseurotop_prep2 = dseurotop_prep2[(dseurotop_prep2.userContributions < 20000) | (dseurotop_prep2.userContributions.isnull())]


# In[78]:


sns.boxplot(x = 'userContributions', data = dseurotop_prep2, color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("User Contributions Boxplot", fontsize=plots_Title_fontSize);


# In[79]:


print("We deleted " + str(91410 - dseurotop_prep2.shape[0]) + " observations from the dseurotop dataset, that is " + str(((91410 - dseurotop_prep2.shape[0])/91410)*100) + "% of the total observations.")


# In[80]:


dseurotop_prep2.kurt().median()


# <a class="anchor" id="sk">
# 
# ### 3.2.5 Skewness And Kurtosis

# In[81]:


dseurotop_prep2.skew()


# Skew = 0 - Normal Distribution;
# Skew < 0 - Negative/Left Skewness -  if we have outliers they will most likely be on the left side and the right side will have more observations;
# Skew > 0 - Positive/Right Skewness - if we have outliers they will most likely be on the right side and the left side will have more observations.

# By looking at the histograms we can now have a sense of the distribution of the variables, to consolidate that knowledge will be analyzed with the variable's skewness and kurtosis.

# In[82]:


figure = plt.figure(figsize=(7,5))
sns.histplot(dseurotop_prep2["globalRating"], bins=40, ec='lightblue', kde=True)
plt.title('Global Rating Distribution')


# In[83]:


# Example of Skewness / totalReviews 2.034448

figure = plt.figure(figsize=(7,5))
sns.histplot(dseurotop_prep2["totalReviews"], bins=40, ec='lightblue', kde=True)
plt.title('totalReviews Distribution')


# In[84]:


# Example of Skewness / sitesOnRanking 1.566013

figure = plt.figure(figsize=(7,5))
sns.histplot(dseurotop_prep2["sitesOnRanking"], bins=40, ec='lightblue', kde=True)
plt.title('sitesOnRanking Distribution')


# In[85]:


# Example of Skewness / positionOnRanking 4.096109

figure = plt.figure(figsize=(7,5))
sns.histplot(dseurotop_prep2["positionOnRanking"], bins=35, ec='lightblue', kde=True)
plt.title('positionOnRanking Distribution')


# In[86]:


dseurotop_prep2.skew().mean()


# In[87]:


dseurotop_prep2.kurt()


# In[88]:


dseurotop_prep2.kurt().mean()


# Kurtosis measures extreme values in both tails, large kurtosis indicates that tail data exceeds the tails of the normal distribution making room for more outliers. 
# 
# - In this dataset we can see that unless the variable userContributions the other ones has small Kurtosis.
# 
# We will apply the cube root transformation to all the variables except for the rating variables (since we tried and concluded that this would represent a negative impact in the overall analysis), because we can see they all have a high positive skewness. We want to achieve the closest to a gaussian distribution as we can for our statistics and analysis.

# In[89]:


dseurotop_prep2['userContributions_tr'] = np.cbrt(dseurotop_prep2['userContributions'])


# In[90]:


plt.hist(dseurotop_prep2['userContributions_tr'], edgecolor='black', color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("User Contributions with Cube Root Tranformation Histogram", fontsize=plots_Title_fontSize);


# In[91]:


sns.histplot(dseurotop_prep2['userContributions_tr'], color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("User Contributions with Cube Root Tranformation ", fontsize=plots_Title_fontSize);


# In[92]:


dseurotop_prep2['positionOnRanking_tr'] = np.cbrt(dseurotop_prep2['positionOnRanking'])


# In[93]:


plt.hist(dseurotop_prep2['positionOnRanking_tr'], edgecolor='black', color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Position On Ranking with Cube Root Tranformation Histogram", fontsize=plots_Title_fontSize);


# In[94]:


sns.histplot(dseurotop_prep2['positionOnRanking_tr'], color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Position On Ranking with Cube Root Tranformation", fontsize=plots_Title_fontSize);


# In[95]:


dseurotop_prep2['totalReviews_tr'] = np.cbrt(dseurotop_prep2['totalReviews'])


# In[96]:



plt.hist(dseurotop_prep2['totalReviews_tr'], edgecolor='black', color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("total Reviews with Cube Root Tranformation Histogram", fontsize=plots_Title_fontSize);


# In[97]:


sns.histplot(dseurotop_prep2['totalReviews_tr'], color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Total Reviews with Cube Root Tranformation ", fontsize=plots_Title_fontSize);


# In[98]:


dseurotop_prep2['sitesOnRanking_tr'] = np.cbrt(dseurotop_prep2['sitesOnRanking'])


# In[99]:



plt.hist(dseurotop_prep2['sitesOnRanking_tr'], edgecolor='black', color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Sites On Ranking with Cube Root Tranformation Histogram", fontsize=plots_Title_fontSize);


# In[100]:


sns.histplot(dseurotop_prep2['sitesOnRanking_tr'], color = 'lightblue')

plt.rc('axes', labelsize = subPlots_label_fontSize)
plt.title("Sites On Ranking with Cube Root Tranformation ", fontsize=plots_Title_fontSize);


# <a class="anchor" id="cc">
# 
# ### 3.2.6. Check Correlations

# In[101]:


#Create correlation matrix
corr = dseurotop_prep2.corr(method = 'spearman')
mask = np.zeros_like(corr, dtype = bool)
mask[np.triu_indices_from(mask)] = True

#Draw
cmap = sns.diverging_palette(220, 20, as_cmap = True) #aqui da para mudar a cor
fig , ax = plt.subplots(figsize = (18, 20))
heatmap = sns.heatmap(corr,
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = cmap,
                      cbar_kws = {'shrink': .4,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      fmt = '.2f',
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': heatmaps_text_fontSize})

#Decoration
plt.title("Spearman correlation", fontsize = plots_Title_fontSize)
ax.set_yticklabels(corr.columns, rotation = 0)
ax.set_xticklabels(corr.columns, rotation = 90)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})


# In[102]:


#Identify Correlations

dseurotop_corr = dseurotop_prep2.copy()
corr = dseurotop_corr.corr()

figure = plt.figure(figsize=(20,15))
heatmap1=sns.heatmap(data = corr, annot = True, cmap="YlGnBu")
plt.show()


# <a class="anchor" id="fe">
# 
# # 4. Feature Engineering

# - Check coherence on the dataset;
# - Make transformations to the dataset;

# <a class="anchor" id="cv">
# 
# ## 4.1. Changing Variables

# In[103]:


#split countries and cities

dseurotop_prep2['userLocation'] = dseurotop_prep2['userLocation'].astype(str)


# In[104]:


dseurotop_prep2['userLocation'].value_counts()


# In[105]:


dseurotop_prep2['userLocation'][0].split(', ')


# In[106]:


field1 = []
field2 = []

for i in range(dseurotop_prep2.shape[0]):
    a = dseurotop_prep2['userLocation'][i].split(', ')
    field1.append(a[0])
    if len(a)>1:
        field2.append(a[1])
    else: 
        field2.append('Unknown')


# In[107]:


dseurotop_prep2['userCity'] = field1


# In[108]:


dseurotop_prep2['userCountry'] = field2


# In[109]:


dseurotop_prep2.head(5)


# In[110]:


dseurotop_prep2['Covid time'] = dseurotop_prep2['reviewVisited'] > '01-03-2020'


# In[111]:


# Create column indicating "before covid" / "after covid"
def Covid_time(dseurotop_prep2):
    if dseurotop_prep2['Covid time'] == False:
           return "Before_Covid"
    else:
           return "After_Covid"
dseurotop_prep2['Covid_time'] = dseurotop_prep2.apply(Covid_time, axis=1)


# In[112]:


dseurotop_prep2.head()


# In[113]:


#transform the userCountry values in uppercase

dseurotop_prep2['userCountry'] = dseurotop_prep2['userCountry'].str.upper()


# In[114]:


#eliminate spaces from strings

dseurotop_prep2['userCountry']=dseurotop_prep2['userCountry'].str.strip()


# In[122]:


#delete characters

dseurotop_prep2['userCountry']=dseurotop_prep2['userCountry'].replace('.','')


# In[123]:


pd.set_option('display.max_rows', None)


# In[124]:


#list states/cities

brazil = ['Brasil', 'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
canada = ['ONTARIO /CANADA', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU', 'NEWFOUNDLAND AND LABRADOR', 'PRINCE EDWARD ISLAND', 'NOVA SCOTIA', 'NEW BRUNSWICK', 'QUEBEC', 'ONTARIO', 'MANITOBA', 'SASKATCHEWAN', 'ALBERTA', 'BRITISH COLUMBIA', 'YUKON', 'NORTHWEST TERRITORIES', 'NUNAVUT', 'BASHKORTOSTAN']
usa = ['U.S.A', 'U.S.A.', 'UNITED STATES', 'TX   USA', 'OH USA', 'MASS', 'N CAROLINA', 'MONTANA USA', 'MARIANA ISLANDS', 'DELAWARE USA', 'CO USA', 'D.C','D.C.', 'CO. USA', 'USA', 'AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT','VT', 'VA', 'VI', 'WA', 'WV', 'WI', 'WY', 'ALABAMA', 'ALASKA', 'AMERICAN SAMOA', 'ARIZONA', 'ARKANSAS', 'BERKS', 'CAROLINE DU NORD', 'CALIFORNIA', 'CALLIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE', 'DISTRICT OF COLUMBIA', 'COLUMBIA', 'FLORIDA', 'GEORGIA', 'GUAM', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NUEVA YORK', 'NORTH CAROLINA', 'N. CAROLINA', 'NORTH DAKOTA', 'NORTHERN MARIANA', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', "TEXAS YA'LL", 'UTAH', 'VERMONT', 'VIRGINIA', 'VIRGIN ISLANDS', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING']
netherlands = ['NL', 'UTRECHT', 'LIMBURG', 'NOORD-HOLLAND', 'UTRECHT']
norway = ['AKERSHUS', 'GÄVLEBORG COUNTY']
newzeland = ['AUCKLAND', 'WELLINGTON']
philippines = ['BACOLOD', 'CEBU CITY']
greece = ['ΕΛΛΆΔΑ', 'ATTIKI']
russia = ['РОССИЯ', 'MOSCOW CITY' ]
uk = ['UNITED KINGDOM', 'WILTSHIRE', 'WEST MIDLANDS', 'WEST YORKS', 'WEST YORKSHIRE', 'WALES', 'WARWICKSHIRE', 'UK & GRASSE', 'SWALES', 'SUFFOLK', 'SURREY', 'SUSSEX', 'STH YORKSHIRE', 'STAFFS', 'SOUTH WALES', 'SOMERSET', 'READING', 'SCOTLAND UK', 'SCOTLAND', 'OXFORDSHIRE', 'OXON', 'POWYS', 'NORTH HUMBERSIDE', 'NORTH WALES', 'NORTHAMPTON', 'MIDDLESEX', 'MIDLETON', 'MERSEYSIDE', 'LINCOLNSHIRE', 'LANCASHIRE','LEICESTERSHIRE', 'KENT', 'GLANNAU MERSI', 'DEVON','DURHAM','ENGLAND', 'ESSEX', 'DORSET', 'UK & GRASSE', 'CO ANTRIM', 'CHESHIRE', 'WEST SUSSEX', 'HERTS', 'HERTFORDSHIRE', 'HERTFORSHIRE', 'SOMERSET', 'BELFAST', 'CAMBRIDGSHIRE', 'HAMPSHIRE']
israel = ['HATZAFON', 'YERUSHALAYIM', 'HAMERKAZ', 'TEL AVIV']
germany = ['DEUTSCHLAND', 'BADEN-WÜRTTEMBERG', 'HESSEN' ]
egipt = ['AL QAHIRAH', '' ]
france = ['NEW CALEDONIA ', 'ALSACE', 'AQUITAINE', 'BRETAGNE', 'ILE-DE-FRANCE', 'LIMOUSIN']
turkey = ['ANKARA', 'MUGLA PROVINCE']
australia = ['AUSTRALIA', 'WESTERN AUSTRALIA', 'TASMANIA', 'VICTORIA', 'SYDNEY', 'AUSTRALIAN CAPITAL TERRITORY', 'NEW SOUTH WALES', 'NSW', 'QLD', 'QUEENSLAND', 'SOUTH AUSTRALIA']
italy = ['VENETO', 'CAMPANIA', 'EMILIA-ROMAGNA', 'ITALIA', 'MARCHE', 'SICILY', 'VENETO']
mexico = ['STATE OF MEXICO']
portugal = ['AVEIRO', 'LISBOA']
romania = ['BRASOV', 'BUCURESTI', 'PRAHOVA']
spain = ['CANARIAS']
ireland = ['DUBLIN', 'COUNTY WATERFORD', 'WEXFORD']
denmark = ['FYN']
southafrica = ['GAUTENG', 'WESTERN CAPE']
caribbean = ['GRAND CAYMAN', 'STVINCENT', 'CAYMAN ISLANDS', 'GRENADA', 'GUADELOUPE', 'NEW PROVIDENCE ISLAND', 'ST MARTIN / ST MAARTEN', 'ST VINCENT']
ghana = ['GREATER ACCRA']
ecuador = ['GUAYAS', 'ISABELA']
indonesia = ['JAKARTA RAYA']
india = ['MAHARASHTRA', 'WEST BENGAL']
taiwan = ['NEW TAIPEI', 'TAIWAN', 'TAIPEI']
malaysia = ['SELANGOR', 'WILAYAH PERSEKUTUAN']
sweden = ['SVERIGE']
tobago = ['TOBAGO', 'TRINIDAD']
uae = ['UNITED ARAB EMIRATES', 'UNI EMIRAT ARAB']
switzerland = ['ZUG']
venezuela = ['CARONI']
argentina = ['CIUDAD AUTÓNOMA DE BUENOS AIRES']
tanzania = ['DAR ES SALAAM']


# In[125]:


#renaming userCountry to United States

for i in range(dseurotop_prep2.shape[0]):
    if dseurotop_prep2['userCountry'][i] in usa:
        dseurotop_prep2['userCountry'][i] = 'USA'
    if dseurotop_prep2['userCountry'][i] in canada:
        dseurotop_prep2['userCountry'][i] = 'CANADA'
    if dseurotop_prep2['userCountry'][i] in uk:
        dseurotop_prep2['userCountry'][i] = 'UK'
    if dseurotop_prep2['userCountry'][i] in brazil:
        dseurotop_prep2['userCountry'][i] = 'BRAZIL'
    if dseurotop_prep2['userCountry'][i] in netherlands:
        dseurotop_prep2['userCountry'][i] = 'NETHERLANDS'
    if dseurotop_prep2['userCountry'][i] in newzeland:
        dseurotop_prep2['userCountry'][i] = 'NEW ZELAND'
    if dseurotop_prep2['userCountry'][i] in norway:
        dseurotop_prep2['userCountry'][i] = 'NORWAY'
    if dseurotop_prep2['userCountry'][i] in philippines:
        dseurotop_prep2['userCountry'][i] = 'PHILIPPINES'
    if dseurotop_prep2['userCountry'][i] in greece:
        dseurotop_prep2['userCountry'][i] = 'GREECE'
    if dseurotop_prep2['userCountry'][i] in russia:
        dseurotop_prep2['userCountry'][i] = 'RUSSIA'
    if dseurotop_prep2['userCountry'][i] in israel:
        dseurotop_prep2['userCountry'][i] = 'ISRAEL'
    if dseurotop_prep2['userCountry'][i] in germany:
        dseurotop_prep2['userCountry'][i] = 'GERMANY'
    if dseurotop_prep2['userCountry'][i] in egipt:
        dseurotop_prep2['userCountry'][i] = 'EGIPT'
    if dseurotop_prep2['userCountry'][i] in france:
        dseurotop_prep2['userCountry'][i] = 'FRANCE'
    if dseurotop_prep2['userCountry'][i] in turkey:
        dseurotop_prep2['userCountry'][i] = 'TURKEY'
    if dseurotop_prep2['userCountry'][i] in australia:
        dseurotop_prep2['userCountry'][i] = 'AUSTRALIA'
    if dseurotop_prep2['userCountry'][i] in italy:
        dseurotop_prep2['userCountry'][i] = 'ITALY'
    if dseurotop_prep2['userCountry'][i] in mexico:
        dseurotop_prep2['userCountry'][i] = 'MEXICO'
    if dseurotop_prep2['userCountry'][i] in portugal:
        dseurotop_prep2['userCountry'][i] = 'PORTUGAL'
    if dseurotop_prep2['userCountry'][i] in romania:
        dseurotop_prep2['userCountry'][i] = 'ROMANIA'
    if dseurotop_prep2['userCountry'][i] in spain:
        dseurotop_prep2['userCountry'][i] = 'SPAIN'    
    if dseurotop_prep2['userCountry'][i] in ireland:
        dseurotop_prep2['userCountry'][i] = 'IRELAND'     
    if dseurotop_prep2['userCountry'][i] in denmark:
        dseurotop_prep2['userCountry'][i] = 'DENMARK'
    if dseurotop_prep2['userCountry'][i] in southafrica:
        dseurotop_prep2['userCountry'][i] = 'SOUTH AFRICA'
    if dseurotop_prep2['userCountry'][i] in caribbean:
        dseurotop_prep2['userCountry'][i] = 'CARIBBEAN'
    if dseurotop_prep2['userCountry'][i] in ghana:
        dseurotop_prep2['userCountry'][i] = 'GHANA'
    if dseurotop_prep2['userCountry'][i] in ecuador:
        dseurotop_prep2['userCountry'][i] = 'ECUADOR'    
    if dseurotop_prep2['userCountry'][i] in indonesia:
        dseurotop_prep2['userCountry'][i] = 'INDONESIA'        
    if dseurotop_prep2['userCountry'][i] in india:
        dseurotop_prep2['userCountry'][i] = 'INDIA'    
    if dseurotop_prep2['userCountry'][i] in taiwan:
        dseurotop_prep2['userCountry'][i] = 'TAIWAN'
    if dseurotop_prep2['userCountry'][i] in malaysia:
        dseurotop_prep2['userCountry'][i] = 'MALAYSIA'
    if dseurotop_prep2['userCountry'][i] in sweden:
        dseurotop_prep2['userCountry'][i] = 'SWEDEN'
    if dseurotop_prep2['userCountry'][i] in tobago:
        dseurotop_prep2['userCountry'][i] = 'TOBAGO'
    if dseurotop_prep2['userCountry'][i] in uae:
        dseurotop_prep2['userCountry'][i] = 'UAE'
    if dseurotop_prep2['userCountry'][i] in switzerland:
        dseurotop_prep2['userCountry'][i] = 'SWITZERLAND'
    if dseurotop_prep2['userCountry'][i] in venezuela:
        dseurotop_prep2['userCountry'][i] = 'VENEZUELA'
    if dseurotop_prep2['userCountry'][i] in argentina:
        dseurotop_prep2['userCountry'][i] = 'ARGENTINA'
    if dseurotop_prep2['userCountry'][i] in tanzania:
        dseurotop_prep2['userCountry'][i] = 'TANZANIA'


# In[126]:


# Create new variable to check when the customer visited the attraction. 
#Starting with identifying the month of the visit
#Then change for the season

dseurotop_prep2["seasonVisited"] = dseurotop_prep2['reviewVisited'].dt.month_name()

dseurotop_prep2.head()


# In[127]:


#Create a list of Seasons

summer = ['June', 'July', 'August', 'September']
winter = ['January', 'February', 'December']
autumn = ['October', 'November']
spring = ['March', 'April', 'May']


# In[128]:


# renaming monthVisited

for i in range(dseurotop_prep2.shape[0]):
    if dseurotop_prep2['seasonVisited'][i] in summer:
        dseurotop_prep2['seasonVisited'][i] = 'Summer'
    if dseurotop_prep2['seasonVisited'][i] in winter:
        dseurotop_prep2['seasonVisited'][i] = 'Winter'
    if dseurotop_prep2['seasonVisited'][i] in autumn:
        dseurotop_prep2['seasonVisited'][i] = 'Autumn'
    if dseurotop_prep2['seasonVisited'][i] in spring:
        dseurotop_prep2['seasonVisited'][i] = 'Spring'


# <a class="anchor" id="m">
# 
# # 5. Modeling

# <a class="anchor" id="rfm">
# 
# ## 5.1. RFM Model

# In[129]:


#create subset Portugal

dseurotop_portugal = dseurotop_prep2.copy()

dseurotop_portugal = dseurotop_prep2.loc[dseurotop_prep2['Country'] == 'Portugal']


# In[130]:


dseurotop_portugal.head(5)


# In[131]:


# Compute totals per customer
dateMax = dseurotop_portugal.reviewWritten.max()
dspt = dseurotop_portugal.groupby(['userCountry']).agg(Recency=('reviewVisited', lambda date: (dateMax - date.max()).days),
                                   Frequency=('userName', lambda i: len(i.unique())),
                                   Ratings=('reviewRating', 'sum')).fillna(0)


# In[132]:


dspt.hist(column='Frequency')


# In[133]:


dspt['FrequencyLog'] = np.log(dspt['Frequency'])


# In[134]:


dspt.hist(column='FrequencyLog')


# In[135]:


cols = ['Recency','Frequency','Ratings']
table =dspt[cols].describe()
table


# In[136]:


# Calculate RMF scores

# Function
def RFMScore(x, col):
    if x <= dspt.quantile(0.25)[col]:
        return '1'
    elif x <= dspt.quantile(0.5)[col]:
        return '2'
    elif x <= dspt.quantile(0.75)[col]:
        return '3'
    else:
        return '4'

# Process
dspt['RScore'] = dspt['Recency'].apply(RFMScore, col='Recency')
dspt['FScore'] = dspt['Frequency'].apply(RFMScore, col='Frequency')
dspt['MScore'] = dspt['Ratings'].apply(RFMScore, col='Ratings')


# In[137]:


# Show first 5
dspt.head(5)


# In[138]:


# Create a column with full RMF score and sort the results

# Transform to string
cols = ['RScore','FScore','MScore']
dspt[cols] = dspt[cols].astype(str)

# Concatenate
dspt['RFMScore'] = dspt['RScore'] + dspt['FScore'] + dspt['MScore']

# Sort
dspt = dspt.sort_values(by=['RFMScore'])


# In[139]:


# Show first 5
dspt.head(5)


# ## Evaluation
# 
# ### General

# Now we have our segment "most valued customers" after the RFM analysis, we should explore some patterns with some statistical analysis.
# 
# We will try to get some insights amongst those customers, such as the most popular Portuguese attractions, the weight of each trip type, and when they prefer to visit.

# In[140]:


# Calculate statistics per RFM segment
RFMStats = dspt.reset_index().groupby(['RFMScore']).agg(nrCountry=('userCountry', lambda i: len(i.unique()) ),
                                                     avgRecency=('Recency', 'mean'),
                                                     avgFrequency=('Frequency', 'mean'),
                                                     avgRatings=('Ratings', 'mean')).fillna(0)


# In[141]:


# Show statistics
RFMStats.head(10)


# In[142]:


# Check who are the countries from a specific segment (e.g."144")
dspt[dspt['RFMScore']=='144']


# In[143]:


# Interactive 3D scatter plot of each customer's RFM values
get_ipython().run_line_magic('matplotlib', 'widget')

# Draw
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection = '3d')

# Decoration
ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("Ratings")

"""
color = []
for x in dspt['RFMScore']:
    if x[0]<'3':
        color.append('green')
    elif x[0]=='3':
        color.append('yellow')
    else:
        color.append('red')
"""

# Define color according to Recency (1 and 2:Green, 3:Yellow, 4:Red)
color = ['green' if x[0]<'3' else ('yellow' if x[0]=='3' else 'red') for x in dspt['RFMScore']]

# Plot
ax.scatter(dspt['Recency'], dspt['Frequency'], dspt['Ratings'], c=color)
plt.show()


# In[144]:


# Histogram of RFM
get_ipython().run_line_magic('matplotlib', 'inline')
cols = ['Recency','Frequency','Ratings']

# Draw
fig, ax = plt.subplots(1, 3, figsize=(10,4))
for var, subplot in zip(dspt[cols], ax.flatten()):
    g = sns.histplot(data=dspt,
                bins=10,
                 x=var,
                 ax=subplot,
                 kde=False)

# Decoration
sns.despine()
plt.rc('axes', labelsize=subPlots_label_fontSize)
fig.suptitle("RFM histograms", fontsize=plots_Title_fontSize);


# In[145]:


RFMStats.index


# In[146]:


# Treemap with number of customers by segment

# Define colors for levels
def assignColor(rfm):
    if (rfm=='144'):
        hex='lightpink'   # pink
    elif (rfm in ['142','143','133','134','124']):
        hex='orange'   # orange
    elif (rfm in ['141','131','132','122','123','113','114']):
        hex='lightyellow'   # yellow
    else:
        hex='lightblue'   # blue
    return hex

color = [assignColor(x) for x in RFMStats.index]

# Draw
fig, ax = plt.subplots(figsize=(10,7))

# Plot
squarify.plot(sizes=RFMStats['nrCountry'], 
              label=RFMStats.index,
              color = color,
              alpha=.9,
              pad=True)                    

# Decoration
plt.title("Country by RFM segment",fontsize=plots_Title_fontSize)
plt.axis('off')
plt.show()


# In[147]:


tempDF = RFMStats
tempDF.sort_values(by=['nrCountry'], ascending=False)


# In[148]:


# RFM Heatmap

# Prepare data
tempDF = RFMStats
tempDF['Frequency'] = tempDF.index.str[1]
tempDF['Ratings'] = tempDF.index.str[2]
pt = pd.pivot_table(tempDF, values='avgRecency', 
                     index=['Frequency'], 
                     columns='Ratings')

# Draw
fig , ax = plt.subplots(figsize=(6, 8))
heatmap = sns.heatmap(pt,
                      square = True,
                      linewidths = .5,
                      cmap = 'Blues',
                      cbar=False,
                      fmt='.0f',
                      annot = True,
                      annot_kws = {'size': heatmaps_text_fontSize+2})

# Decoration
plt.title("Average Recency (days) by Ratings and Frequency levels", fontsize=plots_Title_fontSize)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})


# #### Ranking best visitors per localID

# In[149]:


# Reset Index localID to use as a variable
dseurotop_portugal.reset_index('localID', inplace = True)


# In[150]:


# Copy the remaining variables to the modeling dataset (in this case is only the country)
cols = ['userCountry','localID']
dspt = dspt.merge(dseurotop_portugal[cols], how='left', left_index=True, right_on='userCountry').drop(columns='userCountry')


# In[151]:


# Encode categorical variables to dummy variables
#For more information the different methods to enconde categorical variables check https://contrib.scikit-learn.org/category_encoders/
cols = ['localID']
ce_one_hot = ce.OneHotEncoder(cols = cols, use_cat_names=True)
dspt = ce_one_hot.fit_transform(dspt)


# In[152]:


# Create a copy of the dataset just with the columns to analyze
dspt_analysis = dspt.drop(columns=['RScore','FScore','MScore','RFMScore'])


# In[153]:


# Check the mean values of each segment
segmentsMeanDF = pd.DataFrame(dspt_analysis.groupby(dspt['RFMScore'].values).mean())
segmentsMeanDF.transpose()


# In[154]:


# Analyze one segment in specific
segmentsMeanDF.loc['144']


# #### Ranking best visitors per Triptype

# In[155]:


# Compute totals per customer
dateMax1 = dseurotop_portugal.reviewWritten.max()
dspt1 = dseurotop_portugal.groupby(['userCountry']).agg(Recency=('reviewVisited', lambda date: (dateMax1 - date.max()).days),
                                   Frequency=('userName', lambda i: len(i.unique())),
                                   Ratings=('reviewRating', 'sum')).fillna(0)


# In[156]:


# Calculate RMF scores

# Function
def RFMScore(x, col):
    if x <= dspt1.quantile(0.25)[col]:
        return '1'
    elif x <= dspt1.quantile(0.5)[col]:
        return '2'
    elif x <= dspt1.quantile(0.75)[col]:
        return '3'
    else:
        return '4'

# Process
dspt1['RScore'] = dspt1['Recency'].apply(RFMScore, col='Recency')
dspt1['FScore'] = dspt1['Frequency'].apply(RFMScore, col='Frequency')
dspt1['MScore'] = dspt1['Ratings'].apply(RFMScore, col='Ratings')


# In[157]:


# Transform to string
cols = ['RScore','FScore','MScore']
dspt1[cols] = dspt1[cols].astype(str)

# Concatenate
dspt1['RFMScore'] = dspt1['RScore'] + dspt1['FScore'] + dspt1['MScore']

# Sort
dspt1 = dspt1.sort_values(by=['RFMScore'])


# In[158]:


# Copy the remaining variables to the modeling dataset (in this case is only the tripType)
cols = ['userCountry','tripType']
dspt1 = dspt1.merge(dseurotop_portugal[cols], how='left', left_index=True, right_on='userCountry').drop(columns='userCountry')


# In[159]:


# Encode categorical variables to dummy variables
#For more information the different methods to enconde categorical variables check https://contrib.scikit-learn.org/category_encoders/
cols = ['tripType']
ce_one_hot = ce.OneHotEncoder(cols = cols, use_cat_names=True)
dspt1 = ce_one_hot.fit_transform(dspt1)


# In[160]:


# Create a copy of the dataset just with the columns to analyze
dspt_analysis1 = dspt1.drop(columns=['RScore','FScore','MScore','RFMScore'])


# In[161]:


# Check the mean values of each segment
segmentsMeanDF1 = pd.DataFrame(dspt_analysis1.groupby(dspt1['RFMScore'].values).mean())
segmentsMeanDF1.transpose()


# In[162]:


# Analyze one segment in specific
segmentsMeanDF1.loc['144']


# #### Ranking best visitors per seasonVisited

# In[163]:


# Compute totals per customer
dateMax2 = dseurotop_portugal.reviewWritten.max()
dspt2 = dseurotop_portugal.groupby(['userCountry']).agg(Recency=('reviewVisited', lambda date: (dateMax2 - date.max()).days),
                                   Frequency=('userName', lambda i: len(i.unique())),
                                   Ratings=('reviewRating', 'sum')).fillna(0)


# In[164]:


# Calculate RMF scores

# Function
def RFMScore(x, col):
    if x <= dspt2.quantile(0.25)[col]:
        return '1'
    elif x <= dspt2.quantile(0.5)[col]:
        return '2'
    elif x <= dspt2.quantile(0.75)[col]:
        return '3'
    else:
        return '4'

# Process
dspt2['RScore'] = dspt2['Recency'].apply(RFMScore, col='Recency')
dspt2['FScore'] = dspt2['Frequency'].apply(RFMScore, col='Frequency')
dspt2['MScore'] = dspt2['Ratings'].apply(RFMScore, col='Ratings')


# In[165]:


# Transform to string
cols = ['RScore','FScore','MScore']
dspt2[cols] = dspt2[cols].astype(str)

# Concatenate
dspt2['RFMScore'] = dspt2['RScore'] + dspt2['FScore'] + dspt2['MScore']

# Sort
dspt2 = dspt2.sort_values(by=['RFMScore'])


# In[166]:


# Copy the remaining variables to the modeling dataset (in this case is only the seasonVisited)
cols = ['userCountry','seasonVisited']
dspt2 = dspt2.merge(dseurotop_portugal[cols], how='left', left_index=True, right_on='userCountry').drop(columns='userCountry')


# In[167]:


# Encode categorical variables to dummy variables
#For more information the different methods to enconde categorical variables check https://contrib.scikit-learn.org/category_encoders/
cols = ['seasonVisited']
ce_one_hot = ce.OneHotEncoder(cols = cols, use_cat_names=True)
dspt2 = ce_one_hot.fit_transform(dspt2)


# In[168]:


# Create a copy of the dataset just with the columns to analyze
dspt_analysis2 = dspt2.drop(columns=['RScore','FScore','MScore','RFMScore'])


# In[169]:


# Check the mean values of each segment
segmentsMeanDF2 = pd.DataFrame(dspt_analysis2.groupby(dspt2['RFMScore'].values).mean())
segmentsMeanDF2.transpose()


# In[170]:


# Analyze one segment in specific
segmentsMeanDF2.loc['144']


# 
# 
# 
# 
# Now we proceed with the dataset analysis using the segment of most valued customers that we found with the RFM model. Where we try to understand some patterns such as: what's the most visited or reviewed attractions, which ones our segments visit to find out what are our main competitors.
# 
# First, we do the analysis for the general dataset. Then we will use with our segment "most valued customers" to analize the Portugal context only.
# 
# 

# Main general insigths:
# 

# In[171]:


#create a new copy of the dataset
dsAT = dseurotop_prep2.copy()


# #### What are the top20 most reviewed attractions?

# In[172]:


dsAT.groupby(['Name'])['totalReviews'].median().nlargest(20)


# #### What are the top20 most visited countries? 

# In[173]:


dsAT['Country'].value_counts().nlargest(20)


# In[174]:


#top20 visitors countries? (weight in the dataset)


# In[175]:


#delete UNKNOWN values

dsATwithoutUnknown = dsAT[dsAT['userCountry'] != 'UNKNOWN']


# In[176]:


#top20 visitors countries? without 'UNKNOWN' (weight in the dataset)

dsATwithoutUnknown['userCountry'].value_counts(normalize=True).nlargest(20)


# Portugal Context insigths:

# #### The impact of covid in Portugal - countries with more visits after Covid

# In[177]:


# Portugal: impact of covid - countries with more visits after Covid

covid = pd.crosstab(index = dsATwithoutUnknown['userCountry'], columns=dsATwithoutUnknown['Covid_time'])
covid['Variation (%)'] = ((covid['After_Covid']/covid['Before_Covid']-1)*100).round(1)
covid=covid.sort_values('After_Covid', ascending=False)
covid.head(20)


# #### General insights for most valued customers (based in our RFM model)

# In[178]:


#subset with our main segments/markets

dsRFM = dsAT[dsAT['userCountry'].isin(['PORTUGAL', 'SWITZERLAND', 'ITALY', 'GREECE', 'GERMANY', 'FRANCE', 'THE NETHERLANDS', 'ROMANIA', 'USA', 'CANADA', 'SOUTH AFRICA', 'UK', 'BELGIUM', 'SPAIN'])]


# In[179]:


dsRFM.describe()


# In[180]:


dsRFM.groupby(['Name'])['totalReviews'].median().nlargest(20)


# #### top20 countries for RFM segments

# In[181]:


dsRFM['Country'].value_counts().nlargest(20)


# <a class="anchor" id="dv">
# 
# # 6. Data Visualization

# <a class="anchor" id="ga">
# 
# ## 6.1. General analysis

# In[182]:


#reviewRating Analysis

ax = sns.countplot(y="reviewRating", data=dseurotop_prep2, palette = 'Blues')
plt.title('Distribution of Review Rating')
plt.xlabel('Count')
plt.ylabel('Review Rating')

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.yaxis.grid(False)
ax.xaxis.grid(True)
plt.xticks(np.arange(0,80000,10000))

total = len(dseurotop_prep2['reviewRating'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_width()/total) 
    x = p.get_x() + p.get_width() + 50
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x, y))

plt.show()


# In[183]:


#globalRate Analysis

ax = sns.countplot(y="globalRating", data=dseurotop_prep2, palette = 'Blues')
plt.title('Distribution of Global Rating')
plt.xlabel('Count')
plt.ylabel('Global Rating')

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.yaxis.grid(False)
ax.xaxis.grid(True)
plt.xticks(np.arange(0,100000,10000))

total = len(dseurotop_prep2['globalRating'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_width()/total) 
    x = p.get_x() + p.get_width() + 50
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x, y))

plt.show()


# In[184]:


#TripType Analysis

ax = sns.countplot(y="tripType", data=dseurotop_prep2, palette = 'Blues')
plt.title('Distribution of Trip Type')
plt.xlabel('Count')
plt.ylabel('Trip Type')
plt.xticks(np.arange(0,40000,5000))
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.yaxis.grid(False)
ax.xaxis.grid(True)

total = len(dseurotop_prep2['tripType'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_width()/total) 
    x = p.get_x() + p.get_width() + 50
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x, y))

plt.show()


# In[185]:


# Frequency of Review Rating

# Draw
fig, ax = plt.subplots(figsize=(8,5))
g = sns.countplot(data=dseurotop_prep2, x=dseurotop_prep2['reviewRating'])

# Decoration
fmt = "{x:,.0f}"
tick = ticker.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
sns.despine()
plt.title("Frequency of Review Rating", fontsize=plots_Title_fontSize)
plt.xlabel("reviewRating")
plt.ylabel("Frequency")
plt.rc('axes', labelsize=subPlots_label_fontSize)

# Save to file
#fig.savefig(fname=exportsFolder+'CountPlot.svg', bbox_inches="tight")


# In[186]:


#Check distribution of globalRating and tripType
sns.countplot(y='tripType', hue='globalRating',data = dseurotop_prep2, palette = 'Blues', 
              order=['Couples','Family','Friends','Solo','Business'])
plt.ylabel("Trip per type", labelpad = 40)
plt.xlabel("Number of people", labelpad = 40)
plt.title('Distribution of Review and Global rating', fontsize = 15)
plt.show()


# In[187]:


# Frequency of Trips per type

# Draw
fig, ax = plt.subplots(figsize=(8,5))
g = sns.countplot(data=dseurotop_prep2, x=dseurotop_prep2['tripType'])

# Decoration
fmt = "{x:,.0f}"
tick = ticker.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
sns.despine()
plt.title("Frequency of Trips per Type", fontsize=plots_Title_fontSize)
plt.xlabel("tripType")
plt.ylabel("Frequency")
plt.rc('axes', labelsize=subPlots_label_fontSize)

# Save to file
#fig.savefig(fname=exportsFolder+'CountPlot.svg', bbox_inches="tight")


# In[188]:


# COUNT PLOT

# Draw
fig, ax = plt.subplots(figsize=(8,5))
g = sns.countplot(data=dseurotop_prep2, x=dseurotop_prep2['reviewRating'])

# Decoration
fmt = "{x:,.0f}"
tick = ticker.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
sns.despine()
plt.title("Ratings Frequency", fontsize=plots_Title_fontSize)
plt.xlabel("reviewRating")
plt.ylabel("Frequency")
plt.rc('axes', labelsize=subPlots_label_fontSize)


# Save to file
#fig.savefig(fname=exportsFolder+'CountPlot.svg', bbox_inches="tight")


# In[189]:


#Check reviews distribution according to tripType

sns.boxplot(x = 'tripType', y = 'reviewRating', data = dseurotop_prep2, palette = 'Reds')

axes = plt.gca()
axes.yaxis.grid(color = '0.5',linestyle='dotted')
plt.ylabel("Review Rating", labelpad = 20)
plt.xlabel("Trip Type", labelpad = 20)
plt.title('Rating Distribution according to Trip Type', fontsize = 15)
plt.yticks(np.arange(0,6,1))
axes.set_yticklabels(['{:,}'.format(int(x)) for x in axes.get_yticks().tolist()])


# #### Now we will analyze the portuguese tourism context, focusing in our most value customers from RFM model and take in consideration the pandemic situation since march '21

# In[190]:


#impact of covid - before and after

ncount = len(dsRFM)
plt.figure(figsize=(20,16))
ax = sns.countplot(x="userCountry", hue="Covid_time", data=dsRFM,  palette = 'Blues', edgecolor = 'w')
plt.xlabel('Visitors')
ax2=ax.twinx()
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('Frequency [%]')
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
ax2.set_ylim(0,600)
ax.set_ylim(0,25000)
ax2.grid(None)


# In[191]:


dsRFMtop3 = dsRFM[dsRFM['userCountry'].isin(['USA', 'UK', 'CANADA'])]


# In[192]:


#best value customers vs tripType

ncount = len(dsRFMtop3)
plt.figure(figsize=(20,16))
ax = sns.countplot(x="userCountry", hue="tripType", data=dsRFMtop3,  palette = 'Blues', edgecolor = 'w')
plt.xlabel('Visitors')
ax2=ax.twinx()
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('Frequency [%]')
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
ax2.set_ylim(0,300)
ax.set_ylim(0,12500)
ax2.grid(None)


# In[193]:


#best value customers top3 vs Season

ncount = len(dsRFMtop3)
plt.figure(figsize=(20,16))
ax = sns.countplot(x="userCountry", hue="seasonVisited", data=dsRFMtop3,  palette = 'Blues', edgecolor = 'w')
plt.xlabel('Visitors')
ax2=ax.twinx()
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('Frequency [%]')
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
ax2.set_ylim(0,300)
ax.set_ylim(0,12500)
ax2.grid(None)


# In[194]:


#top3 season vs tripType

# STACKED BAR PLOT - DISTRIBUTION

# Aggregate and sort
tempDF = dsRFM.pivot_table(values=['userCountry'], 
                      index='tripType',
                      columns='seasonVisited',
                      aggfunc='count',
                      fill_value=0)
tempDF = tempDF.div(tempDF.sum(1), axis=0)

# Draw
fig, ax = plt.subplots(figsize=(8,5))
g = tempDF.plot(kind='bar', stacked=True, ax=ax)

# Decoration
vals = ax.get_yticks().tolist()
ax.yaxis.set_major_locator(ticker.FixedLocator(vals))
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
sns.despine()
plt.title("Trip Type vs Season for Top3 countries", fontsize=plots_Title_fontSize)
plt.xlabel("Season Visited")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.rc('axes', labelsize=subPlots_label_fontSize)
handles, labels = ax.get_legend_handles_labels()
labels = ['Autumn', 'Spring', 'Summer', 'Winter']
ax.legend(handles=handles, labels=labels, loc='upper center', 
          ncol=4, bbox_to_anchor=(0.47, 1.03), frameon=False)

# Save to file
#fig.savefig(fname=exportsFolder+'StackedBar100Percent.svg', bbox_inches="tight")


# <a class="anchor" id="rs">
# 
# # 7. Recommendation  systems

# After analysing our data, we proceed with a creation of a recommendation system based on the attractions that our best value customers visited. Our goal is to recommend portuguese attractions take in consideration the attractions from popular countries, season and trip type.

# #### RECOMMENDATION FOR OUR MOST IMPORTANT SEGMENT
# 
# The follower model has the purpose to identify a sort of specific recommendations to our number one and most value segment. (althought, it is possible to reproduce all the steps for other countries in order to disclosure the respetive recommendation for each country)

# In[195]:


#create new variable to identify each review
dsRFM_rec = dsRFM.copy()
dsRFM_rec['ReviewID'] = np.arange(dsRFM.shape[0])


# In[196]:


# From the customers with more purchases, let's select one
topCountries = dsRFM_rec.groupby('userCountry')['userName'].nunique().sort_values(ascending=False)
topCountries.head(20)


# In[197]:


# Check the visits of the first country
country1 = 'UK'
dsRFM_rec[dsRFM_rec['userCountry']==country1].head()


# In[198]:


# Create a pivot table with visitors per attractions

ptvisitorsAttraction = pd.pivot_table(dsRFM_rec[dsRFM_rec['userCountry']==country1][['userName','Name']],
                    index='userName',                             # Each row will be a visit (transaction)
                    columns='Name',                  # Each attraction will be a column
                    aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)
                            
print(ptvisitorsAttraction.shape)
ptvisitorsAttraction.head(10)


# In[199]:


# Compute the distance and transform it to a dataframe for visualization
# Info on the "dice" implementation https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.dice.html

visits_rec = ptvisitorsAttraction.to_numpy().T
distMatrix = pairwise_distances(visits_rec,metric='dice')
distMatrixDF = pd.DataFrame(distMatrix, columns=ptvisitorsAttraction.columns, index=ptvisitorsAttraction.columns)
distMatrixDF = distMatrixDF.apply(lambda x: 1-x, axis=1) # Transform dissimilarity to similarity
distMatrixDF


# In[200]:


# Let's define the portuguese attractions to simulate
ptAttractions = ['Torre de Belém','Mosteiro dos Jeronimos', 'Ponte de Dom Luís I', 'Park and National Palace of Pena', 'Quinta da Regaleira', 'Cais da Ribeira','Bom Jesus do Monte']


# In[201]:


# Let's create a list of products not in the current basket
EuropeanAttractions = [x for x in distMatrixDF.columns if x not in ptAttractions]

# Create a temporary dataframe with rows in basket and columns not in basket and 
tempDF = distMatrixDF[distMatrixDF.index.isin(ptAttractions)][EuropeanAttractions]

# Create a max similarity dataframe do store the results
maxSimilarDF = pd.DataFrame(columns=['European Attractions','Portuguese Attractions','Similarity'])

# For each item not in the basket
for c in EuropeanAttractions:
    # Get maximum similarity value of the item (index of the row of the item in the basket)
    indexOfMax = tempDF[c].argmax()
    # Add the results to the dataframe
    maxSimilarDF = maxSimilarDF.append(
                   {'European Attractions':c,
                    'Portuguese Attractions':tempDF.iloc[indexOfMax].name,
                    'Similarity':tempDF.iloc[indexOfMax][c]
                    }, ignore_index=True
    )

# Sort results by similarity
maxSimilarDF = maxSimilarDF.sort_values(by='Similarity', ascending=False)


# In[202]:


# Simulate presentation of top n recommendations (Recommendations are the items "NotInBaskket")
n = 10
maxSimilarDF.head(n)


# #### RECOMMENDATION BASED ON ALL PREVIOUS VISITS
# 
# We also thought of developing a model capable of offering recommendations based on the previous visits from our potential customers. For this model, we want to use a set of attractions from the country with more attractions.

# In[203]:


# List the number of attractions by country #top10

dsRFM_rec['Country'].value_counts().head(10)


# In[204]:


# Let's define one popular spanish attraction to simulate the problem
spanishattractions = ['Basilica of the Sagrada Familia', 'Parc Guell', 'The Alhambra', 'Real Alcazar de Sevilla', 'Acueduct of Segovia']

# Families to recomend
attractionsToRecommend = ['Portugal']


# In[205]:


# Create a pivot table with products per document (only desired products)
ptSpainRec = pd.pivot_table(dsRFM_rec[(dsRFM_rec['Name'].isin(spanishattractions)) | (dsRFM_rec['Country'].isin(attractionsToRecommend))][['userName','Name']],
                    index='userName',                             # Each row will be a document (transaction)
                    columns='Name',                  # Each product will be a column
                    aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)
                            
print(ptSpainRec.shape)
ptSpainRec.head()


# In[206]:


# Compute the distance and transform it to a dataframe for visualization
SpainVisitors = ptSpainRec.to_numpy().T
distMatrix = pairwise_distances(SpainVisitors,metric='dice')
distMatrixDF = pd.DataFrame(distMatrix, columns=ptSpainRec.columns, index=ptSpainRec.columns)
distMatrixDF = distMatrixDF.apply(lambda x: 1-x, axis=1) # Transform dissimilarity to similarity
distMatrixDF


# In[207]:


# Let's create a list of products not in the current basket
attractionsToRecommend = [x for x in distMatrixDF.columns if x not in spanishattractions]

# Create a temporary dataframe with rows in basket and columns not in basket and 
tempDF = distMatrixDF[distMatrixDF.index.isin(spanishattractions)][attractionsToRecommend]

# Create a max similarity dataframe do store the results
maxSimilarDF = pd.DataFrame(columns=['Portuguese attractions','Spanish attractions','Similarity'])

# For each item not in the basket
for c in attractionsToRecommend:
    # Get maximum similarity value of the item (index of the row of the item in the basket)
    indexOfMax = tempDF[c].argmax()
    # Add the results to the dataframe
    maxSimilarDF = maxSimilarDF.append(
                   {'Portuguese attractions':c,
                    'Spanish attractions':tempDF.iloc[indexOfMax].name,
                    'Similarity':tempDF.iloc[indexOfMax][c]
                    }, ignore_index=True
    )

# Sort results by similarity
maxSimilarDF = maxSimilarDF.sort_values(by='Similarity', ascending=False)


# In[208]:


# Simulate presentation of top n recommendations (Recommendations are the items "NotInBaskket")
n = 5
maxSimilarDF.head(n)

