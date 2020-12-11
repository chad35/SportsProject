## Question

The prediction question I am trying to answer is:

###### Can we predict if a NCAA College Football QB is going to be drafted in the first round? 

To complete this analysis, I first go collecting and identifying the data needed. ESPN has one of the best compiled lists of data, including historical stats, and this is the main data I am drawing from. Using Beautiful Soup and other webscraping techniques, I was able to collect the statistics of the top 50 quarterbacks in the FBS from 2004 to 2020. Then, I found a list of all the first round picks in the same time frame and scraped that data set as well. After cleaning the data, I merged the two together to have a complete data set. I then divided up the dataset , with year 2004-2019 to be the labeled data set on which to train the model. 2020 can be used to generate prediction. Of note: Althouth the data was gathered, predictions were not yet generated at time of making this code due to their being a few weeks left in the season.  However the model was trained with Ridge, Random Forest, and Elastic Net and was scored with the accuarcy displayed.

## Web Scraping and Exporting Data


```python
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
```


```python
#These blocks of code were used to test out the URL and response
url = "https://www.espn.com/college-football/stats/player/_/view/offense/table/passing/sort/passingYards/dir/desc"
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
```


```python
header = soup.find('tr',attrs={'class' : 'Table__sub-header Table__TR Table__even'})
print(header)
```

    <tr class="Table__sub-header Table__TR Table__even"><th class="Table__TH" title=""><span class=""><div>RK</div></span></th><th class="Table__TH" title=""><span class=""><div>Name</div></span></th></tr>



```python
#Trying to parse through all the possible tables to find the headers, 
#I listed them out to easily see which ones I needed
for j in range(0,102):
    column_headers = soup.findAll('tr')[j]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    print(str(j) + ": ")
    print(column_headers)
```

    0: 
    ['RK', 'Name']
    1: 
    []
    2: 
    []
    3: 
    []
    4: 
    []
    5: 
    []
    6: 
    []
    7: 
    []
    8: 
    []
    9: 
    []
    10: 
    []
    11: 
    []
    12: 
    []
    13: 
    []
    14: 
    []
    15: 
    []
    16: 
    []
    17: 
    []
    18: 
    []
    19: 
    []
    20: 
    []
    21: 
    []
    22: 
    []
    23: 
    []
    24: 
    []
    25: 
    []
    26: 
    []
    27: 
    []
    28: 
    []
    29: 
    []
    30: 
    []
    31: 
    []
    32: 
    []
    33: 
    []
    34: 
    []
    35: 
    []
    36: 
    []
    37: 
    []
    38: 
    []
    39: 
    []
    40: 
    []
    41: 
    []
    42: 
    []
    43: 
    []
    44: 
    []
    45: 
    []
    46: 
    []
    47: 
    []
    48: 
    []
    49: 
    []
    50: 
    []
    51: 
    ['POS', 'CMP', 'ATT', 'CMP%', 'YDS', 'AVG', 'LNG', 'TD', 'INT', 'SACK', 'RTG']
    52: 
    []
    53: 
    []
    54: 
    []
    55: 
    []
    56: 
    []
    57: 
    []
    58: 
    []
    59: 
    []
    60: 
    []
    61: 
    []
    62: 
    []
    63: 
    []
    64: 
    []
    65: 
    []
    66: 
    []
    67: 
    []
    68: 
    []
    69: 
    []
    70: 
    []
    71: 
    []
    72: 
    []
    73: 
    []
    74: 
    []
    75: 
    []
    76: 
    []
    77: 
    []
    78: 
    []
    79: 
    []
    80: 
    []
    81: 
    []
    82: 
    []
    83: 
    []
    84: 
    []
    85: 
    []
    86: 
    []
    87: 
    []
    88: 
    []
    89: 
    []
    90: 
    []
    91: 
    []
    92: 
    []
    93: 
    []
    94: 
    []
    95: 
    []
    96: 
    []
    97: 
    []
    98: 
    []
    99: 
    []
    100: 
    []
    101: 
    []



```python
#Getting the column headers for the individual statistics
column_headers = soup.findAll('tr')[51]
column_headers = [i.getText() for i in column_headers.findAll('th')]
column_headers
```




    ['POS', 'CMP', 'ATT', 'CMP%', 'YDS', 'AVG', 'LNG', 'TD', 'INT', 'SACK', 'RTG']




```python
players = soup.find_all('tr', attrs={"class":re.compile("Table__TR Table__TR--sm Table__even")})
len(players)
```




    100




```python
#This is the code that is actually generating the dataset
#using components of the above code

#The players and their stats are technically two different tables. so the headers for each are collected
player_headers = soup.findAll('tr')[0]
player_headers = [i.getText() for i in player_headers.findAll('th')]
player_headers

column_headers = soup.findAll('tr')[51]
column_headers = [i.getText() for i in column_headers.findAll('th')]
column_headers

#initialize dataframe
QB_data = pd.DataFrame()

#For each year (Each year on different webpage)...
for year in range(2004,2021):
    player_df = pd.DataFrame()
    stats_df = pd.DataFrame()
    final_df = pd.DataFrame()
    #Dynamic URL for each web page
    url = "https://www.espn.com/college-football/stats/player/_/view/offense/season/" + str(year) + "/table/passing/sort/passingYards/dir/desc"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    #For each player, pull the name
    players = soup.find_all('tr', attrs={"class":re.compile("Table__TR Table__TR--sm Table__even")})
    i=0
    for player in players:
        stats = [stat.get_text() for stat in player.find_all('td')]
            
        temp = pd.DataFrame(stats).transpose()
        temp.columns = player_headers
        
        player_df = pd.concat([player_df,temp], ignore_index=True)
        if i > 48:
            break
        i=i+1

    i=0
    
    #For each player, pull the stats
    for player in players:
        if i < 50:
            i=i+1
            continue
        i=i+1
        stats = [stat.get_text() for stat in player.find_all('td')]
        
        temp = pd.DataFrame(stats).transpose()
        temp.columns = column_headers
        
        stats_df = pd.concat([stats_df,temp], ignore_index=True)
    
    final_df = pd.concat([player_df,stats_df],axis=1)
    final_df["YEAR"] = year
    #Merge the two data sets together
    QB_data = pd.concat([QB_data,final_df])


```


```python
QB_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RK</th>
      <th>Name</th>
      <th>POS</th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Sonny CumbieTTU</td>
      <td>QB</td>
      <td>421</td>
      <td>642</td>
      <td>65.6</td>
      <td>4,742</td>
      <td>7.4</td>
      <td>80</td>
      <td>32</td>
      <td>18</td>
      <td>26</td>
      <td>138.5</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>Timmy ChangHAW</td>
      <td>QB</td>
      <td>358</td>
      <td>602</td>
      <td>59.5</td>
      <td>4,258</td>
      <td>7.1</td>
      <td>75</td>
      <td>37</td>
      <td>13</td>
      <td>15</td>
      <td>134.8</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>Omar JacobsBGSU</td>
      <td>QB</td>
      <td>309</td>
      <td>462</td>
      <td>66.9</td>
      <td>4,002</td>
      <td>8.7</td>
      <td>58</td>
      <td>41</td>
      <td>4</td>
      <td>10</td>
      <td>167.2</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>Derek AndersonORST</td>
      <td>QB</td>
      <td>279</td>
      <td>515</td>
      <td>54.2</td>
      <td>3,615</td>
      <td>7.0</td>
      <td>55</td>
      <td>29</td>
      <td>17</td>
      <td>37</td>
      <td>125.1</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>Bruce GradkowskiTOL</td>
      <td>QB</td>
      <td>280</td>
      <td>399</td>
      <td>70.2</td>
      <td>3,518</td>
      <td>8.8</td>
      <td>96</td>
      <td>27</td>
      <td>8</td>
      <td>14</td>
      <td>162.6</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1</td>
      <td>Josh BettsM-OH</td>
      <td>QB</td>
      <td>268</td>
      <td>444</td>
      <td>60.4</td>
      <td>3,512</td>
      <td>7.9</td>
      <td>79</td>
      <td>23</td>
      <td>14</td>
      <td>24</td>
      <td>137.6</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1</td>
      <td>Dan OrlovskyCONN</td>
      <td>QB</td>
      <td>288</td>
      <td>457</td>
      <td>63.0</td>
      <td>3,354</td>
      <td>7.3</td>
      <td>90</td>
      <td>23</td>
      <td>15</td>
      <td>14</td>
      <td>134.7</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1</td>
      <td>Matt LeinartUSC</td>
      <td>QB</td>
      <td>269</td>
      <td>412</td>
      <td>65.3</td>
      <td>3,322</td>
      <td>8.1</td>
      <td>69</td>
      <td>33</td>
      <td>6</td>
      <td>23</td>
      <td>156.5</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1</td>
      <td>Jason WhiteOKLA</td>
      <td>QB</td>
      <td>255</td>
      <td>390</td>
      <td>65.4</td>
      <td>3,205</td>
      <td>8.2</td>
      <td>72</td>
      <td>35</td>
      <td>9</td>
      <td>9</td>
      <td>159.4</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1</td>
      <td>Chris LeakFLA</td>
      <td>QB</td>
      <td>238</td>
      <td>399</td>
      <td>59.6</td>
      <td>3,199</td>
      <td>8.0</td>
      <td>81</td>
      <td>29</td>
      <td>12</td>
      <td>19</td>
      <td>145.0</td>
      <td>2004</td>
    </tr>
  </tbody>
</table>
</div>




```python
QB_data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RK</th>
      <th>Name</th>
      <th>POS</th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>45</td>
      <td>46</td>
      <td>Michael Penix Jr.IU</td>
      <td>QB</td>
      <td>124</td>
      <td>220</td>
      <td>56.4</td>
      <td>1,645</td>
      <td>7.5</td>
      <td>68</td>
      <td>14</td>
      <td>4</td>
      <td>7</td>
      <td>136.5</td>
      <td>2020</td>
    </tr>
    <tr>
      <td>46</td>
      <td>47</td>
      <td>Jeff SimsGT</td>
      <td>QB</td>
      <td>124</td>
      <td>226</td>
      <td>54.9</td>
      <td>1,643</td>
      <td>7.3</td>
      <td>59</td>
      <td>11</td>
      <td>11</td>
      <td>18</td>
      <td>122.3</td>
      <td>2020</td>
    </tr>
    <tr>
      <td>47</td>
      <td>48</td>
      <td>Michael PrattTULN</td>
      <td>QB</td>
      <td>128</td>
      <td>229</td>
      <td>55.9</td>
      <td>1,638</td>
      <td>7.2</td>
      <td>52</td>
      <td>18</td>
      <td>5</td>
      <td>25</td>
      <td>137.6</td>
      <td>2020</td>
    </tr>
    <tr>
      <td>48</td>
      <td>49</td>
      <td>Max DugganTCU</td>
      <td>QB</td>
      <td>136</td>
      <td>227</td>
      <td>59.9</td>
      <td>1,635</td>
      <td>7.2</td>
      <td>71</td>
      <td>9</td>
      <td>4</td>
      <td>18</td>
      <td>130.0</td>
      <td>2020</td>
    </tr>
    <tr>
      <td>49</td>
      <td>50</td>
      <td>Alan BowmanTTU</td>
      <td>QB</td>
      <td>150</td>
      <td>232</td>
      <td>64.7</td>
      <td>1,602</td>
      <td>6.9</td>
      <td>48</td>
      <td>10</td>
      <td>7</td>
      <td>2</td>
      <td>130.8</td>
      <td>2020</td>
    </tr>
  </tbody>
</table>
</div>




```python
#This is testing the table where all 1st round draft picks are found
url2 = "http://www.drafthistory.com/index.php/rounds/round_1"
page2 = requests.get(url2)
soup = BeautifulSoup(page2.text, 'html.parser')

#This code is to find where the headers are
#draft_cols = soup.findAll('tr')
#for j in range(0,2000):
#    draft_cols = soup.findAll('tr')[j]
#    draft_cols = [i.getText() for i in draft_cols.findAll('th')]
#    print(str(j) + ": ")
#    print(draft_cols)

#Get the columns
draft_cols = soup.findAll('tr')[1]
draft_cols = [i.getText() for i in draft_cols.findAll('th')]
draft_cols
```




    ['Year', 'Round', 'Pick', 'Player', 'Name', 'Team', 'Position', 'College']




```python
#This is the actual code that pulls in the data using some of the above componets

url2 = "http://www.drafthistory.com/index.php/rounds/round_1"
page2 = requests.get(url2)
soup = BeautifulSoup(page2.text, 'html.parser')

draft_cols = soup.findAll('tr')[1]
draft_cols = [i.getText() for i in draft_cols.findAll('th')]
draft_df = pd.DataFrame()

rd1pl = soup.find_all('tr', attrs={"bgcolor":re.compile("ffffff")})
for player in rd1pl:
    info = [stat.get_text() for stat in player.find_all('td')]
            
    temp = pd.DataFrame(info).transpose()
    temp.columns = draft_cols
        
    draft_df = pd.concat([draft_df,temp], ignore_index=True)

```


```python
#Writing these data sets to csv files which are used in the project

#draft_df.to_csv("draft_df.csv")
#QB_data.to_csv("QB_data.csv")
```


```python

```

## Data Cleaning


```python
qb_stats = pd.read_csv("QB_data.csv")
```

First we will clean up the QB Data...


```python
qb_stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>RK</th>
      <th>Name</th>
      <th>POS</th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Sonny CumbieTTU</td>
      <td>QB</td>
      <td>421</td>
      <td>642</td>
      <td>65.6</td>
      <td>4,742</td>
      <td>7.4</td>
      <td>80</td>
      <td>32</td>
      <td>18</td>
      <td>26</td>
      <td>138.5</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Timmy ChangHAW</td>
      <td>QB</td>
      <td>358</td>
      <td>602</td>
      <td>59.5</td>
      <td>4,258</td>
      <td>7.1</td>
      <td>75</td>
      <td>37</td>
      <td>13</td>
      <td>15</td>
      <td>134.8</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>Omar JacobsBGSU</td>
      <td>QB</td>
      <td>309</td>
      <td>462</td>
      <td>66.9</td>
      <td>4,002</td>
      <td>8.7</td>
      <td>58</td>
      <td>41</td>
      <td>4</td>
      <td>10</td>
      <td>167.2</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>Derek AndersonORST</td>
      <td>QB</td>
      <td>279</td>
      <td>515</td>
      <td>54.2</td>
      <td>3,615</td>
      <td>7.0</td>
      <td>55</td>
      <td>29</td>
      <td>17</td>
      <td>37</td>
      <td>125.1</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>Bruce GradkowskiTOL</td>
      <td>QB</td>
      <td>280</td>
      <td>399</td>
      <td>70.2</td>
      <td>3,518</td>
      <td>8.8</td>
      <td>96</td>
      <td>27</td>
      <td>8</td>
      <td>14</td>
      <td>162.6</td>
      <td>2004</td>
    </tr>
  </tbody>
</table>
</div>




```python
qb_stats.columns
```




    Index(['Unnamed: 0', 'RK', 'Name', 'POS', 'CMP', 'ATT', 'CMP%', 'YDS', 'AVG',
           'LNG', 'TD', 'INT', 'SACK', 'RTG', 'YEAR'],
          dtype='object')




```python
qb_stats.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 850 entries, 0 to 849
    Data columns (total 15 columns):
    Unnamed: 0    850 non-null int64
    RK            850 non-null int64
    Name          850 non-null object
    POS           850 non-null object
    CMP           850 non-null int64
    ATT           850 non-null int64
    CMP%          850 non-null float64
    YDS           850 non-null object
    AVG           850 non-null float64
    LNG           850 non-null int64
    TD            850 non-null int64
    INT           850 non-null int64
    SACK          850 non-null int64
    RTG           850 non-null float64
    YEAR          850 non-null int64
    dtypes: float64(3), int64(9), object(3)
    memory usage: 99.7+ KB



```python
#Taking out any commas
qb_stats.replace(',','', regex=True, inplace=True)
```


```python
qb_stats['YDS'] = qb_stats['YDS'].astype('int')
```


```python
#Don't need rank, position for this project
qb_stats.drop(['Unnamed: 0', 'RK','POS'], axis=1,inplace=True)
```


```python
#Used regex for cleaning and creating new variables
#Need to find a way to extract college for the name (Using capitilization to split)
pattern = '[A-Z]{2,}'
qb_stats['College'] = qb_stats['Name'].apply(lambda x:str(re.findall(pattern, x)[0]))
qb_stats['Name'] = qb_stats['Name'].apply(lambda x: x.replace((re.findall(pattern, x)[0]),""))
```


```python
qb_stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
      <th>College</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Sonny Cumbie</td>
      <td>421</td>
      <td>642</td>
      <td>65.6</td>
      <td>4742</td>
      <td>7.4</td>
      <td>80</td>
      <td>32</td>
      <td>18</td>
      <td>26</td>
      <td>138.5</td>
      <td>2004</td>
      <td>TTU</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Timmy Chang</td>
      <td>358</td>
      <td>602</td>
      <td>59.5</td>
      <td>4258</td>
      <td>7.1</td>
      <td>75</td>
      <td>37</td>
      <td>13</td>
      <td>15</td>
      <td>134.8</td>
      <td>2004</td>
      <td>HAW</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Omar Jacobs</td>
      <td>309</td>
      <td>462</td>
      <td>66.9</td>
      <td>4002</td>
      <td>8.7</td>
      <td>58</td>
      <td>41</td>
      <td>4</td>
      <td>10</td>
      <td>167.2</td>
      <td>2004</td>
      <td>BGSU</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Derek Anderson</td>
      <td>279</td>
      <td>515</td>
      <td>54.2</td>
      <td>3615</td>
      <td>7.0</td>
      <td>55</td>
      <td>29</td>
      <td>17</td>
      <td>37</td>
      <td>125.1</td>
      <td>2004</td>
      <td>ORST</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Bruce Gradkowski</td>
      <td>280</td>
      <td>399</td>
      <td>70.2</td>
      <td>3518</td>
      <td>8.8</td>
      <td>96</td>
      <td>27</td>
      <td>8</td>
      <td>14</td>
      <td>162.6</td>
      <td>2004</td>
      <td>TOL</td>
    </tr>
  </tbody>
</table>
</div>



Now we will clean the draft data to get ready to combine...


```python
draft_stats = pd.read_csv("draft_df.csv")
```


```python
draft_stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Year</th>
      <th>Round</th>
      <th>Pick</th>
      <th>Player</th>
      <th>Name</th>
      <th>Team</th>
      <th>Position</th>
      <th>College</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Joe Burrow</td>
      <td>Bengals</td>
      <td>QB</td>
      <td>Louisiana State</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td></td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>Jeff Okudah</td>
      <td>Lions</td>
      <td>DB</td>
      <td>Ohio State</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td></td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>Tua Tagovailoa</td>
      <td>Dolphins</td>
      <td>QB</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td></td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>Derrick Brown</td>
      <td>Panthers</td>
      <td>DT</td>
      <td>Auburn</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td></td>
      <td>1</td>
      <td>9</td>
      <td>9</td>
      <td>CJ Henderson</td>
      <td>Jaguars</td>
      <td>DB</td>
      <td>Florida</td>
    </tr>
  </tbody>
</table>
</div>




```python
draft_stats.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 997 entries, 0 to 996
    Data columns (total 9 columns):
    Unnamed: 0    997 non-null int64
    Year          997 non-null object
    Round         997 non-null int64
    Pick          997 non-null int64
    Player        997 non-null int64
    Name          997 non-null object
    Team          997 non-null object
    Position      997 non-null object
    College       997 non-null object
    dtypes: int64(4), object(5)
    memory usage: 70.2+ KB



```python
#Finding where everything is year 2003 and on and dropping that data
draft_stats[draft_stats["Year"] == '2003']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Year</th>
      <th>Round</th>
      <th>Pick</th>
      <th>Player</th>
      <th>Name</th>
      <th>Team</th>
      <th>Position</th>
      <th>College</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>271</td>
      <td>271</td>
      <td>2003</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Carson Palmer</td>
      <td>Bengals</td>
      <td>QB</td>
      <td>USC</td>
    </tr>
  </tbody>
</table>
</div>




```python
draft_stats = draft_stats.loc[:270,:]
```


```python
draft_stats.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Year</th>
      <th>Round</th>
      <th>Pick</th>
      <th>Player</th>
      <th>Name</th>
      <th>Team</th>
      <th>Position</th>
      <th>College</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>266</td>
      <td>266</td>
      <td></td>
      <td>1</td>
      <td>23</td>
      <td>23</td>
      <td>Marcus Tubbs</td>
      <td>Seahawks</td>
      <td>DT</td>
      <td>Texas</td>
    </tr>
    <tr>
      <td>267</td>
      <td>267</td>
      <td></td>
      <td>1</td>
      <td>25</td>
      <td>25</td>
      <td>Ahmad Carroll</td>
      <td>Packers</td>
      <td>DB</td>
      <td>Arkansas</td>
    </tr>
    <tr>
      <td>268</td>
      <td>268</td>
      <td></td>
      <td>1</td>
      <td>27</td>
      <td>27</td>
      <td>Jason Babin</td>
      <td>Texans</td>
      <td>DE</td>
      <td>Western Michigan</td>
    </tr>
    <tr>
      <td>269</td>
      <td>269</td>
      <td></td>
      <td>1</td>
      <td>29</td>
      <td>29</td>
      <td>Michael Jenkins</td>
      <td>Falcons</td>
      <td>WR</td>
      <td>Ohio State</td>
    </tr>
    <tr>
      <td>270</td>
      <td>270</td>
      <td></td>
      <td>1</td>
      <td>31</td>
      <td>31</td>
      <td>Rashaun Woods</td>
      <td>49ers</td>
      <td>WR</td>
      <td>Oklahoma State</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Just need the name, since each player plays one position and can match unique on name
draft_stats = draft_stats['Name']
```


```python
draft_stats
```




    0           Joe Burrow
    1          Jeff Okudah
    2       Tua Tagovailoa
    3        Derrick Brown
    4         CJ Henderson
                ...       
    266       Marcus Tubbs
    267      Ahmad Carroll
    268        Jason Babin
    269    Michael Jenkins
    270      Rashaun Woods
    Name: Name, Length: 271, dtype: object



Now to combine the datasets...


```python
round_1 = draft_stats.to_list()

qb_stats['Round_1'] = qb_stats['Name'].apply(lambda x: True if (x in round_1) else False)
```


```python
qb_stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
      <th>College</th>
      <th>Round_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Sonny Cumbie</td>
      <td>421</td>
      <td>642</td>
      <td>65.6</td>
      <td>4742</td>
      <td>7.4</td>
      <td>80</td>
      <td>32</td>
      <td>18</td>
      <td>26</td>
      <td>138.5</td>
      <td>2004</td>
      <td>TTU</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Timmy Chang</td>
      <td>358</td>
      <td>602</td>
      <td>59.5</td>
      <td>4258</td>
      <td>7.1</td>
      <td>75</td>
      <td>37</td>
      <td>13</td>
      <td>15</td>
      <td>134.8</td>
      <td>2004</td>
      <td>HAW</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Omar Jacobs</td>
      <td>309</td>
      <td>462</td>
      <td>66.9</td>
      <td>4002</td>
      <td>8.7</td>
      <td>58</td>
      <td>41</td>
      <td>4</td>
      <td>10</td>
      <td>167.2</td>
      <td>2004</td>
      <td>BGSU</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Derek Anderson</td>
      <td>279</td>
      <td>515</td>
      <td>54.2</td>
      <td>3615</td>
      <td>7.0</td>
      <td>55</td>
      <td>29</td>
      <td>17</td>
      <td>37</td>
      <td>125.1</td>
      <td>2004</td>
      <td>ORST</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Bruce Gradkowski</td>
      <td>280</td>
      <td>399</td>
      <td>70.2</td>
      <td>3518</td>
      <td>8.8</td>
      <td>96</td>
      <td>27</td>
      <td>8</td>
      <td>14</td>
      <td>162.6</td>
      <td>2004</td>
      <td>TOL</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
(qb_stats[qb_stats['Round_1'] == True])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
      <th>College</th>
      <th>Round_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>15</td>
      <td>Alex Smith</td>
      <td>214</td>
      <td>317</td>
      <td>67.5</td>
      <td>2952</td>
      <td>9.3</td>
      <td>78</td>
      <td>32</td>
      <td>4</td>
      <td>18</td>
      <td>176.5</td>
      <td>2004</td>
      <td>UTAH</td>
      <td>True</td>
    </tr>
    <tr>
      <td>26</td>
      <td>Jason Campbell</td>
      <td>188</td>
      <td>270</td>
      <td>69.6</td>
      <td>2700</td>
      <td>10.0</td>
      <td>87</td>
      <td>20</td>
      <td>7</td>
      <td>13</td>
      <td>172.9</td>
      <td>2004</td>
      <td>AUB</td>
      <td>True</td>
    </tr>
    <tr>
      <td>66</td>
      <td>Jay Cutler</td>
      <td>273</td>
      <td>462</td>
      <td>59.1</td>
      <td>3073</td>
      <td>6.7</td>
      <td>47</td>
      <td>21</td>
      <td>9</td>
      <td>24</td>
      <td>126.1</td>
      <td>2005</td>
      <td>VAN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>67</td>
      <td>Vince Young</td>
      <td>212</td>
      <td>325</td>
      <td>65.2</td>
      <td>3036</td>
      <td>9.3</td>
      <td>75</td>
      <td>26</td>
      <td>10</td>
      <td>13</td>
      <td>163.9</td>
      <td>2005</td>
      <td>TEX</td>
      <td>True</td>
    </tr>
    <tr>
      <td>97</td>
      <td>JaMarcus Russell</td>
      <td>188</td>
      <td>311</td>
      <td>60.5</td>
      <td>2443</td>
      <td>7.9</td>
      <td>50</td>
      <td>15</td>
      <td>9</td>
      <td>21</td>
      <td>136.6</td>
      <td>2005</td>
      <td>LSU</td>
      <td>True</td>
    </tr>
    <tr>
      <td>112</td>
      <td>JaMarcus Russell</td>
      <td>232</td>
      <td>342</td>
      <td>67.8</td>
      <td>3129</td>
      <td>9.1</td>
      <td>58</td>
      <td>28</td>
      <td>8</td>
      <td>16</td>
      <td>167.0</td>
      <td>2006</td>
      <td>LSU</td>
      <td>True</td>
    </tr>
    <tr>
      <td>251</td>
      <td>Levi Brown</td>
      <td>321</td>
      <td>504</td>
      <td>63.7</td>
      <td>4254</td>
      <td>8.4</td>
      <td>70</td>
      <td>23</td>
      <td>9</td>
      <td>24</td>
      <td>146.1</td>
      <td>2009</td>
      <td>TROY</td>
      <td>True</td>
    </tr>
    <tr>
      <td>256</td>
      <td>Blaine Gabbert</td>
      <td>262</td>
      <td>445</td>
      <td>58.9</td>
      <td>3593</td>
      <td>8.1</td>
      <td>84</td>
      <td>24</td>
      <td>9</td>
      <td>19</td>
      <td>140.5</td>
      <td>2009</td>
      <td>MIZ</td>
      <td>True</td>
    </tr>
    <tr>
      <td>283</td>
      <td>Jake Locker</td>
      <td>230</td>
      <td>395</td>
      <td>58.2</td>
      <td>2800</td>
      <td>7.1</td>
      <td>51</td>
      <td>21</td>
      <td>11</td>
      <td>28</td>
      <td>129.7</td>
      <td>2009</td>
      <td>WASH</td>
      <td>True</td>
    </tr>
    <tr>
      <td>288</td>
      <td>Christian Ponder</td>
      <td>227</td>
      <td>330</td>
      <td>68.8</td>
      <td>2717</td>
      <td>8.2</td>
      <td>98</td>
      <td>14</td>
      <td>7</td>
      <td>15</td>
      <td>147.7</td>
      <td>2009</td>
      <td>FSU</td>
      <td>True</td>
    </tr>
    <tr>
      <td>302</td>
      <td>Brandon Weeden</td>
      <td>342</td>
      <td>511</td>
      <td>66.9</td>
      <td>4277</td>
      <td>8.4</td>
      <td>81</td>
      <td>34</td>
      <td>13</td>
      <td>8</td>
      <td>154.1</td>
      <td>2010</td>
      <td>OKST</td>
      <td>True</td>
    </tr>
    <tr>
      <td>320</td>
      <td>Blaine Gabbert</td>
      <td>301</td>
      <td>475</td>
      <td>63.4</td>
      <td>3186</td>
      <td>6.7</td>
      <td>68</td>
      <td>16</td>
      <td>9</td>
      <td>23</td>
      <td>127.0</td>
      <td>2010</td>
      <td>MIZ</td>
      <td>True</td>
    </tr>
    <tr>
      <td>351</td>
      <td>Brandon Weeden</td>
      <td>408</td>
      <td>564</td>
      <td>72.3</td>
      <td>4727</td>
      <td>8.4</td>
      <td>67</td>
      <td>37</td>
      <td>13</td>
      <td>12</td>
      <td>159.8</td>
      <td>2011</td>
      <td>OKST</td>
      <td>True</td>
    </tr>
    <tr>
      <td>413</td>
      <td>Teddy Bridgewater</td>
      <td>287</td>
      <td>419</td>
      <td>68.5</td>
      <td>3718</td>
      <td>8.9</td>
      <td>75</td>
      <td>27</td>
      <td>8</td>
      <td>28</td>
      <td>160.5</td>
      <td>2012</td>
      <td>LOU</td>
      <td>True</td>
    </tr>
    <tr>
      <td>460</td>
      <td>Teddy Bridgewater</td>
      <td>303</td>
      <td>427</td>
      <td>71.0</td>
      <td>3970</td>
      <td>9.3</td>
      <td>69</td>
      <td>31</td>
      <td>4</td>
      <td>23</td>
      <td>171.1</td>
      <td>2013</td>
      <td>LOU</td>
      <td>True</td>
    </tr>
    <tr>
      <td>464</td>
      <td>Marcus Mariota</td>
      <td>245</td>
      <td>386</td>
      <td>63.5</td>
      <td>3665</td>
      <td>9.5</td>
      <td>75</td>
      <td>31</td>
      <td>4</td>
      <td>18</td>
      <td>167.7</td>
      <td>2013</td>
      <td>ORE</td>
      <td>True</td>
    </tr>
    <tr>
      <td>468</td>
      <td>Jared Goff</td>
      <td>320</td>
      <td>531</td>
      <td>60.3</td>
      <td>3508</td>
      <td>6.6</td>
      <td>89</td>
      <td>18</td>
      <td>10</td>
      <td>32</td>
      <td>123.2</td>
      <td>2013</td>
      <td>CAL</td>
      <td>True</td>
    </tr>
    <tr>
      <td>502</td>
      <td>Marcus Mariota</td>
      <td>304</td>
      <td>445</td>
      <td>68.3</td>
      <td>4454</td>
      <td>10.0</td>
      <td>80</td>
      <td>42</td>
      <td>4</td>
      <td>31</td>
      <td>181.7</td>
      <td>2014</td>
      <td>ORE</td>
      <td>True</td>
    </tr>
    <tr>
      <td>504</td>
      <td>Jared Goff</td>
      <td>316</td>
      <td>509</td>
      <td>62.1</td>
      <td>3973</td>
      <td>7.8</td>
      <td>92</td>
      <td>35</td>
      <td>7</td>
      <td>26</td>
      <td>147.6</td>
      <td>2014</td>
      <td>CAL</td>
      <td>True</td>
    </tr>
    <tr>
      <td>552</td>
      <td>Jared Goff</td>
      <td>341</td>
      <td>529</td>
      <td>64.5</td>
      <td>4719</td>
      <td>8.9</td>
      <td>80</td>
      <td>43</td>
      <td>13</td>
      <td>26</td>
      <td>161.3</td>
      <td>2015</td>
      <td>CAL</td>
      <td>True</td>
    </tr>
    <tr>
      <td>567</td>
      <td>Baker Mayfield</td>
      <td>269</td>
      <td>395</td>
      <td>68.1</td>
      <td>3700</td>
      <td>9.4</td>
      <td>76</td>
      <td>36</td>
      <td>7</td>
      <td>39</td>
      <td>173.3</td>
      <td>2015</td>
      <td>OKLA</td>
      <td>True</td>
    </tr>
    <tr>
      <td>608</td>
      <td>Baker Mayfield</td>
      <td>254</td>
      <td>358</td>
      <td>70.9</td>
      <td>3965</td>
      <td>11.1</td>
      <td>88</td>
      <td>40</td>
      <td>8</td>
      <td>18</td>
      <td>196.4</td>
      <td>2016</td>
      <td>OKLA</td>
      <td>True</td>
    </tr>
    <tr>
      <td>631</td>
      <td>Josh Allen</td>
      <td>209</td>
      <td>373</td>
      <td>56.0</td>
      <td>3203</td>
      <td>8.6</td>
      <td>54</td>
      <td>28</td>
      <td>15</td>
      <td>27</td>
      <td>144.9</td>
      <td>2016</td>
      <td>WYO</td>
      <td>True</td>
    </tr>
    <tr>
      <td>635</td>
      <td>Sam Darnold</td>
      <td>246</td>
      <td>366</td>
      <td>67.2</td>
      <td>3086</td>
      <td>8.4</td>
      <td>67</td>
      <td>31</td>
      <td>9</td>
      <td>6</td>
      <td>161.1</td>
      <td>2016</td>
      <td>USC</td>
      <td>True</td>
    </tr>
    <tr>
      <td>651</td>
      <td>Baker Mayfield</td>
      <td>285</td>
      <td>404</td>
      <td>70.5</td>
      <td>4627</td>
      <td>11.5</td>
      <td>84</td>
      <td>43</td>
      <td>6</td>
      <td>26</td>
      <td>198.9</td>
      <td>2017</td>
      <td>OKLA</td>
      <td>True</td>
    </tr>
    <tr>
      <td>654</td>
      <td>Sam Darnold</td>
      <td>303</td>
      <td>480</td>
      <td>63.1</td>
      <td>4143</td>
      <td>8.6</td>
      <td>56</td>
      <td>26</td>
      <td>13</td>
      <td>29</td>
      <td>148.1</td>
      <td>2017</td>
      <td>USC</td>
      <td>True</td>
    </tr>
    <tr>
      <td>702</td>
      <td>Kyler Murray</td>
      <td>260</td>
      <td>377</td>
      <td>69.0</td>
      <td>4361</td>
      <td>11.6</td>
      <td>86</td>
      <td>42</td>
      <td>7</td>
      <td>18</td>
      <td>199.2</td>
      <td>2018</td>
      <td>OKLA</td>
      <td>True</td>
    </tr>
    <tr>
      <td>704</td>
      <td>Tua Tagovailoa</td>
      <td>245</td>
      <td>355</td>
      <td>69.0</td>
      <td>3966</td>
      <td>11.2</td>
      <td>81</td>
      <td>43</td>
      <td>6</td>
      <td>13</td>
      <td>199.5</td>
      <td>2018</td>
      <td>ALA</td>
      <td>True</td>
    </tr>
    <tr>
      <td>735</td>
      <td>Joe Burrow</td>
      <td>219</td>
      <td>379</td>
      <td>57.8</td>
      <td>2894</td>
      <td>7.6</td>
      <td>71</td>
      <td>16</td>
      <td>5</td>
      <td>35</td>
      <td>133.2</td>
      <td>2018</td>
      <td>LSU</td>
      <td>True</td>
    </tr>
    <tr>
      <td>750</td>
      <td>Joe Burrow</td>
      <td>402</td>
      <td>527</td>
      <td>76.3</td>
      <td>5671</td>
      <td>10.8</td>
      <td>78</td>
      <td>60</td>
      <td>6</td>
      <td>34</td>
      <td>202.0</td>
      <td>2019</td>
      <td>LSU</td>
      <td>True</td>
    </tr>
    <tr>
      <td>797</td>
      <td>Tua Tagovailoa</td>
      <td>180</td>
      <td>252</td>
      <td>71.4</td>
      <td>2840</td>
      <td>11.3</td>
      <td>85</td>
      <td>33</td>
      <td>3</td>
      <td>10</td>
      <td>206.9</td>
      <td>2019</td>
      <td>ALA</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Exploring and Dividing up Data into Test and Train


```python
train = qb_stats[qb_stats['YEAR'] != 2020]
test = qb_stats[qb_stats['YEAR'] == 2020]
```


```python
print(len(train))
print(len(test))
```

    800
    50



```python
sns.barplot(y='YDS',x='Round_1', data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc673d1aa90>




![png](output_45_1.png)



```python
sns.barplot(y='CMP%',x='Round_1', data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc673e15850>




![png](output_46_1.png)



```python
sns.barplot(y='RTG',x='Round_1', data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc674016f50>




![png](output_47_1.png)



```python
sns.boxplot(y='YDS',x='Round_1',data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc67409c550>




![png](output_48_1.png)



```python
sns.boxplot(y='RTG',x='Round_1',data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc674135bd0>




![png](output_49_1.png)



```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
      <td>800.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>261.383750</td>
      <td>417.613750</td>
      <td>62.467875</td>
      <td>3266.630000</td>
      <td>7.863875</td>
      <td>74.181250</td>
      <td>24.715000</td>
      <td>10.273750</td>
      <td>22.737500</td>
      <td>143.212625</td>
      <td>2011.500000</td>
    </tr>
    <tr>
      <td>std</td>
      <td>52.460739</td>
      <td>71.335987</td>
      <td>4.377063</td>
      <td>592.732288</td>
      <td>0.928283</td>
      <td>11.135569</td>
      <td>7.853558</td>
      <td>3.647663</td>
      <td>8.589285</td>
      <td>15.810460</td>
      <td>4.612656</td>
    </tr>
    <tr>
      <td>min</td>
      <td>162.000000</td>
      <td>252.000000</td>
      <td>48.500000</td>
      <td>2222.000000</td>
      <td>5.600000</td>
      <td>44.000000</td>
      <td>10.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>105.500000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>226.000000</td>
      <td>369.000000</td>
      <td>59.400000</td>
      <td>2842.000000</td>
      <td>7.200000</td>
      <td>67.000000</td>
      <td>19.000000</td>
      <td>8.000000</td>
      <td>16.000000</td>
      <td>132.475000</td>
      <td>2007.750000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>250.500000</td>
      <td>405.000000</td>
      <td>62.400000</td>
      <td>3136.000000</td>
      <td>7.800000</td>
      <td>75.000000</td>
      <td>24.000000</td>
      <td>10.000000</td>
      <td>22.000000</td>
      <td>141.750000</td>
      <td>2011.500000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>287.000000</td>
      <td>454.000000</td>
      <td>65.450000</td>
      <td>3576.000000</td>
      <td>8.400000</td>
      <td>81.000000</td>
      <td>29.000000</td>
      <td>13.000000</td>
      <td>28.000000</td>
      <td>152.125000</td>
      <td>2015.250000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>512.000000</td>
      <td>714.000000</td>
      <td>76.700000</td>
      <td>5705.000000</td>
      <td>11.600000</td>
      <td>99.000000</td>
      <td>60.000000</td>
      <td>23.000000</td>
      <td>54.000000</td>
      <td>206.900000</td>
      <td>2019.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train[train["Round_1"] == True].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>270.516129</td>
      <td>411.709677</td>
      <td>65.787097</td>
      <td>3656.451613</td>
      <td>8.990323</td>
      <td>74.193548</td>
      <td>30.193548</td>
      <td>8.161290</td>
      <td>21.709677</td>
      <td>162.080645</td>
      <td>2012.225806</td>
    </tr>
    <tr>
      <td>std</td>
      <td>57.801310</td>
      <td>82.262261</td>
      <td>4.962979</td>
      <td>759.640654</td>
      <td>1.429068</td>
      <td>13.115943</td>
      <td>10.565413</td>
      <td>3.088654</td>
      <td>8.331441</td>
      <td>24.603583</td>
      <td>4.786855</td>
    </tr>
    <tr>
      <td>min</td>
      <td>180.000000</td>
      <td>252.000000</td>
      <td>56.000000</td>
      <td>2443.000000</td>
      <td>6.600000</td>
      <td>47.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>123.200000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>228.500000</td>
      <td>356.500000</td>
      <td>62.600000</td>
      <td>3054.500000</td>
      <td>8.150000</td>
      <td>67.500000</td>
      <td>22.000000</td>
      <td>6.000000</td>
      <td>15.500000</td>
      <td>145.500000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>262.000000</td>
      <td>395.000000</td>
      <td>67.200000</td>
      <td>3665.000000</td>
      <td>8.900000</td>
      <td>76.000000</td>
      <td>31.000000</td>
      <td>8.000000</td>
      <td>23.000000</td>
      <td>161.100000</td>
      <td>2013.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>303.500000</td>
      <td>477.500000</td>
      <td>69.000000</td>
      <td>4198.500000</td>
      <td>9.750000</td>
      <td>84.000000</td>
      <td>36.500000</td>
      <td>9.500000</td>
      <td>27.500000</td>
      <td>174.900000</td>
      <td>2016.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>408.000000</td>
      <td>564.000000</td>
      <td>76.300000</td>
      <td>5671.000000</td>
      <td>11.600000</td>
      <td>98.000000</td>
      <td>60.000000</td>
      <td>15.000000</td>
      <td>39.000000</td>
      <td>206.900000</td>
      <td>2019.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train[train["Round_1"] == False].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CMP</th>
      <th>ATT</th>
      <th>CMP%</th>
      <th>YDS</th>
      <th>AVG</th>
      <th>LNG</th>
      <th>TD</th>
      <th>INT</th>
      <th>SACK</th>
      <th>RTG</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
      <td>769.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>261.015605</td>
      <td>417.851756</td>
      <td>62.334070</td>
      <td>3250.915475</td>
      <td>7.818466</td>
      <td>74.180754</td>
      <td>24.494148</td>
      <td>10.358908</td>
      <td>22.778934</td>
      <td>142.452016</td>
      <td>2011.470741</td>
    </tr>
    <tr>
      <td>std</td>
      <td>52.241823</td>
      <td>70.911413</td>
      <td>4.302002</td>
      <td>580.164324</td>
      <td>0.873747</td>
      <td>11.058313</td>
      <td>7.651612</td>
      <td>3.644513</td>
      <td>8.602206</td>
      <td>14.881717</td>
      <td>4.606324</td>
    </tr>
    <tr>
      <td>min</td>
      <td>162.000000</td>
      <td>257.000000</td>
      <td>48.500000</td>
      <td>2222.000000</td>
      <td>5.600000</td>
      <td>44.000000</td>
      <td>10.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>105.500000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>226.000000</td>
      <td>370.000000</td>
      <td>59.400000</td>
      <td>2838.000000</td>
      <td>7.200000</td>
      <td>67.000000</td>
      <td>19.000000</td>
      <td>8.000000</td>
      <td>16.000000</td>
      <td>132.200000</td>
      <td>2007.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>250.000000</td>
      <td>405.000000</td>
      <td>62.300000</td>
      <td>3131.000000</td>
      <td>7.700000</td>
      <td>75.000000</td>
      <td>24.000000</td>
      <td>10.000000</td>
      <td>22.000000</td>
      <td>141.200000</td>
      <td>2011.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>286.000000</td>
      <td>453.000000</td>
      <td>65.300000</td>
      <td>3557.000000</td>
      <td>8.400000</td>
      <td>81.000000</td>
      <td>29.000000</td>
      <td>13.000000</td>
      <td>28.000000</td>
      <td>151.100000</td>
      <td>2015.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>512.000000</td>
      <td>714.000000</td>
      <td>76.700000</td>
      <td>5705.000000</td>
      <td>11.300000</td>
      <td>99.000000</td>
      <td>58.000000</td>
      <td>23.000000</td>
      <td>54.000000</td>
      <td>191.800000</td>
      <td>2019.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, square = True, cmap="YlGnBu")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc678abea50>




![png](output_53_1.png)


## Building the models


```python
X = train.drop(['Name', "Round_1","College"], axis = 1)
y = train['Round_1']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)
```

#### Using Ridge


```python
ridgecv = RidgeClassifierCV(cv=5).fit(X_train, np.ravel(y_train))
ridge_cv_alpha=RidgeClassifier(alpha = ridgecv.alpha_,max_iter=100000,random_state=0).fit(X_train,y_train)

print("Ridge accuracy on training set: {:.3f}".format(ridge_cv_alpha.score(X_train, y_train)))
print("Ridge accuracy on test set: {:.3f}".format(ridge_cv_alpha.score(X_test, y_test)))
```

    Ridge accuracy on training set: 0.962
    Ridge accuracy on test set: 0.960


#### Using Random Forest


```python
grid = {'min_samples_leaf': [1,2,3,4,5],
       'max_depth' : [1,2,3,4,5,6,7,8,9]}

grid_search = GridSearchCV(RandomForestClassifier(random_state=0),grid,cv=5,return_train_score=True)
best_model=grid_search.fit(X_train,y_train)

print("RF accuracy on training set: {:.3f}".format(best_model.score(X_train, y_train)))
print("RF accuracy on test set: {:.3f}".format(best_model.score(X_test, y_test)))
```

    RF accuracy on training set: 0.972
    RF accuracy on test set: 0.960


#### Using KNN Classifier


```python
grid = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(KNeighborsClassifier(),grid,cv=5,return_train_score=True)
best_model_knn=grid_search.fit(X_train,y_train)

print("KNN accuracy on training set: {:.3f}".format(best_model.score(X_train, y_train)))
print("KNN accuracy on test set: {:.3f}".format(best_model.score(X_test, y_test)))
```

    KNN accuracy on training set: 0.967
    KNN accuracy on test set: 0.960



```python

```

All models have similar out of sample accuuacy, moving forward with the Random Forest model...


```python
yhat = best_model.predict(X_test)
```


```python
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
```


```python
probabilities = best_model.predict_proba(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, thresh = roc_curve(y_test, y_prob)
```


```python
#Here is the ROC curve, due to the limited amount of test data we get a step like curve
plot_roc_curve(fpr, tpr)
```


![png](output_68_0.png)



```python
roc_auc_score(y_test, y_prob)
```




    0.7578125



#### Conclusion and Limitations

Based of the models, we do have some success in predicting if a quarteback is a future first round pick. The major limitation is the limited number of observations that did go in the first round, which could possibly be fixed in future projects. The reason is that ESPN includes QB stats back to 2004, however the first round draft picks date back to the 1930's, forcing us to get rid a lot of potential positive observations. I am sure the data is out there, but it would likely have to be hand collected and is beyond the time scope of this project. However, there are differences in the first rounders and non firstrounders that the model is picking up on. No one observation is enough to decide if someone is good enough for the first round, and there is some inherit variance due to the fact that sometimes teams take QB's in the first round because they are desperate for a QB, even if the QB isn't clear first round material. I would love to keep looking for more data and building off this with time.
