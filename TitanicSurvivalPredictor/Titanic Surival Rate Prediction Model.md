# The Challenge
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

Acknowledgements:
This dataset has been referred from Kaggle: https://www.kaggle.com/c/titanic/data.

# Data

Overview
The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

# Data Dictionary
survival	Survival	0 = No, 1 = Yes

pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

sex	Sex	

Age	Age in years	

sibsp	# of siblings / spouses aboard the Titanic	

parch	# of parents / children aboard the Titanic	

ticket	Ticket number	

fare	Passenger fare	

cabin	Cabin number	

embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# Variable Notes

pclass: A proxy for socio-economic status (SES)
* 1st = Upper
* 2nd = Middle
* 3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...

Sibling = brother, sister, stepbrother, stepsister

Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...

Parent = mother, father

Child = daughter, son, stepdaughter, stepson

Some children travelled only with a nanny, therefore parch=0 for them.

## Preparing the tools

We're going to use pandas, Matplotlib and NumPy for data analysis and manipulation.


```python
# Import all the tools we need

# Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
```

# Load Data


```python
df = pd.read_csv("data/train.csv")
df.shape
```




    (891, 12)



# Data Exploration


```python
df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>



### As we can see by looking at just the head and tail that We have some NaN values and our target column is "Survived"


```python
df["Survived"].value_counts()
```




    Survived
    0    549
    1    342
    Name: count, dtype: int64




```python
ax = df["Survived"].value_counts().plot(kind="bar",color=["red","blue"]);

ax.set_xticklabels(["Not Survived", "Survived"],rotation=0)

ax.set_xlabel("Survival Status")
ax.set_ylabel("Count")
ax.set_title("Survival Count in Dataset")

plt.show();
```


    
![png](output_10_0.png)
    


### Let's see if how many missing values we have and where?


```python
df.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



### By the first look we can fill the na of age with the median not the mean
see https://stats.stackexchange.com/questions/2547/why-is-median-age-a-better-statistic-than-mean-age#:~:text=Finally%2C%20the%20median%20gives%20us,can't%20say%20as%20much%2C

### We can drop the two missing embarked rows but i am not sure what to do with the cabin yet. let's continue..


```python
df["Sex"].value_counts()
```




    Sex
    male      577
    female    314
    Name: count, dtype: int64




```python
pd.crosstab(df.Sex,df.Survived)
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
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>81</td>
      <td>233</td>
    </tr>
    <tr>
      <th>male</th>
      <td>468</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = pd.crosstab(df.Sex,df.Survived).plot(kind="bar",
                                     figsize=(10,6),
                                     color=["red","blue"]);
ax.set_xticklabels(["Female", "Male"],rotation=0)

ax.set_xlabel("Survival Status")
ax.set_ylabel("Amount")
ax.legend(["Not Survived","Survived"])
ax.set_title("Survival Count in Dataset")

plt.show();
```


    
![png](output_17_0.png)
    


### Interesting, most of the man died, and most of the woman survived, maybe thay evacuated the woman first as in the movie


```python
df.Age.plot.hist()
```




    <Axes: ylabel='Frequency'>




    
![png](output_19_1.png)
    


### Not the best distribution


```python
df.corr()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[75], line 1
    ----> 1 df.corr()
    

    File ~\anaconda3\Lib\site-packages\pandas\core\frame.py:11049, in DataFrame.corr(self, method, min_periods, numeric_only)
      11047 cols = data.columns
      11048 idx = cols.copy()
    > 11049 mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
      11051 if method == "pearson":
      11052     correl = libalgos.nancorr(mat, minp=min_periods)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\frame.py:1993, in DataFrame.to_numpy(self, dtype, copy, na_value)
       1991 if dtype is not None:
       1992     dtype = np.dtype(dtype)
    -> 1993 result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
       1994 if result.dtype is not dtype:
       1995     result = np.asarray(result, dtype=dtype)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:1694, in BlockManager.as_array(self, dtype, copy, na_value)
       1692         arr.flags.writeable = False
       1693 else:
    -> 1694     arr = self._interleave(dtype=dtype, na_value=na_value)
       1695     # The underlying data was copied within _interleave, so no need
       1696     # to further copy if copy=True or setting na_value
       1698 if na_value is lib.no_default:
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:1753, in BlockManager._interleave(self, dtype, na_value)
       1751     else:
       1752         arr = blk.get_values(dtype)
    -> 1753     result[rl.indexer] = arr
       1754     itemmask[rl.indexer] = 1
       1756 if not itemmask.all():
    

    ValueError: could not convert string to float: 'Braund, Mr. Owen Harris'


### The Name column gets an error because it's not float, and it doesn't help us train the model anyway because all the names are unique there is no pattern in between, so that being said let's drop it


```python
df = df.drop("Name",axis=1)
```


```python
df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Sex          891 non-null    object 
     4   Age          714 non-null    float64
     5   SibSp        891 non-null    int64  
     6   Parch        891 non-null    int64  
     7   Ticket       891 non-null    object 
     8   Fare         891 non-null    float64
     9   Cabin        204 non-null    object 
     10  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(4)
    memory usage: 76.7+ KB
    


```python
df.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
df.Embarked.value_counts()
```




    Embarked
    S    644
    C    168
    Q     77
    Name: count, dtype: int64



### Since we are missing only two values for embark, i will fill them with the most common one for good.


```python
df["Embarked"] = df["Embarked"].fillna("S")
```


```python
df.Embarked.value_counts()
```




    Embarked
    S    646
    C    168
    Q     77
    Name: count, dtype: int64




```python
df.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         0
    dtype: int64



### Now let's fill the Na Age's with the median see https://stats.stackexchange.com/questions/2547/why-is-median-age-a-better-statistic-than-mean-age#:~:text=Finally%2C%20the%20median%20gives%20us,can't%20say%20as%20much%2C


```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```


```python
df.Age.plot.hist();
```


    
![png](output_34_0.png)
    



```python
df.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Sex              0
    Age              0
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         0
    dtype: int64



### Now that we handled Age, we're left with the Cabin column, even tho we are missing information on every row about Cabin, it is still an important information. So that being said let's do some feature engineering on cabin..


```python
df.Cabin.value_counts()
```




    Cabin
    B96 B98        4
    G6             4
    C23 C25 C27    4
    C22 C26        3
    F33            3
                  ..
    E34            1
    C7             1
    C54            1
    E36            1
    C148           1
    Name: count, Length: 147, dtype: int64



### The passengers that has cabins were probably wealthier so that preserve that informationf first


```python
df["HasCabin"] = df["Cabin"].notna().astype(int)
```


```python
df["HasCabin"].value_counts()
```




    HasCabin
    0    687
    1    204
    Name: count, dtype: int64



### We know that passengers with Cabins were wealthier but even if they had a cabin, the First Letter of the Cabin matters because some Cabins were worthed more than the others. So let's extract that information and the amount of Cabin that passenger had into new Columns


```python
# If passenger has no Cabin fill it it U as in Unknown if not fill with the first letter of the cabin
df["CabinDeck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")
```


```python
# Count Number of Cabins passenger had if non fill with 0
df["CabinCount"] = df["Cabin"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
```

### Now that we extracted the usefull information from the Cabin column, we don't need it anymore. So that let's drop it


```python
df = df.drop("Cabin", axis=1)
```


```python
df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>HasCabin</th>
      <th>CabinDeck</th>
      <th>CabinCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
      <td>U</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>U</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
      <td>U</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's convert the Sex aswell
df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)
```

### Note that we used 0 as female and 1 for male


```python
df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>HasCabin</th>
      <th>CabinDeck</th>
      <th>CabinCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
      <td>U</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>U</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
      <td>U</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.Ticket.value_counts()
```




    Ticket
    347082      7
    CA. 2343    7
    1601        7
    3101295     6
    CA 2144     6
               ..
    9234        1
    19988       1
    2693        1
    PC 17612    1
    370376      1
    Name: count, Length: 681, dtype: int64



### Ticket columns looks important since it's not unique for every value so that let's keep it as it is


```python
df.isna().sum()
```




    PassengerId    0
    Survived       0
    Pclass         0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Embarked       0
    HasCabin       0
    CabinDeck      0
    CabinCount     0
    dtype: int64



### So far so good, we have no missing values anymore, now let's convert rest of our columns into number


```python
df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Sex              int64
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Embarked        object
    HasCabin         int32
    CabinDeck       object
    CabinCount       int64
    dtype: object



### Let's Turn the string and object values into category values


```python
for label, content in df.items():
    if pd.api.types.is_object_dtype(content):
        df[label] = content.astype("category").cat.as_ordered()
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   PassengerId  891 non-null    int64   
     1   Survived     891 non-null    int64   
     2   Pclass       891 non-null    int64   
     3   Sex          891 non-null    int64   
     4   Age          891 non-null    float64 
     5   SibSp        891 non-null    int64   
     6   Parch        891 non-null    int64   
     7   Ticket       891 non-null    category
     8   Fare         891 non-null    float64 
     9   Embarked     891 non-null    category
     10  HasCabin     891 non-null    int32   
     11  CabinDeck    891 non-null    category
     12  CabinCount   891 non-null    int64   
    dtypes: category(3), float64(2), int32(1), int64(7)
    memory usage: 91.7 KB
    


```python
df.CabinDeck.value_counts()
```




    CabinDeck
    U    687
    C     59
    B     47
    D     33
    E     32
    A     15
    F     13
    G      4
    T      1
    Name: count, dtype: int64




```python
df.Ticket.cat.codes
```




    0      523
    1      596
    2      669
    3       49
    4      472
          ... 
    886    101
    887     14
    888    675
    889      8
    890    466
    Length: 891, dtype: int16



### Now finally let's turn our categorical variables into numbers


```python
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        df[label] = pd.Categorical(content).codes + 1
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Sex          891 non-null    int64  
     4   Age          891 non-null    float64
     5   SibSp        891 non-null    int64  
     6   Parch        891 non-null    int64  
     7   Ticket       891 non-null    int16  
     8   Fare         891 non-null    float64
     9   Embarked     891 non-null    int8   
     10  HasCabin     891 non-null    int32  
     11  CabinDeck    891 non-null    int8   
     12  CabinCount   891 non-null    int64  
    dtypes: float64(2), int16(1), int32(1), int64(7), int8(2)
    memory usage: 69.7 KB
    


```python
df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>HasCabin</th>
      <th>CabinDeck</th>
      <th>CabinCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>524</td>
      <td>7.2500</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>597</td>
      <td>71.2833</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>670</td>
      <td>7.9250</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>50</td>
      <td>53.1000</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>473</td>
      <td>8.0500</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Let's make a correlation matrix to see the relations 


```python
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                 annot = True,
                 linewidths = 0.5,
                 fmt = ".2f",
                 cmap = "YlGnBu")
```


    
![png](output_65_0.png)
    


### We can see from the correlation matrix that having Cabin is very important for surviving and as the Pclass number increases the survival rate drops
What was Pclass again:

pclass: A proxy for socio-economic status (SES)

* 1st = Upper
* 2nd = Middle
* 3rd = Lower
  
Summary, more of the wealthier people survived.

# Great now that we are done with exploring and engineering our data, let's dive into modelling

# Modelling


```python
# Split data into X & y

X = df.drop("Survived",axis=1)
y = df["Survived"]
```


```python
# Splitting our data into train and test sets
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X,y)
len(X_train), len(X_test), len(y_train), len(y_test)
```




    (668, 223, 668, 223)



### Time to find a good algorithm to train our model
See: https://scikit-learn.org/1.5/machine_learning_map.html#choosing-the-right-estimator

Models are going to be tested
1. Logistic Regression
2. K-Nearest Neighbours Classifier
3. Random Forest Classifier


```python
# Put models in a dictionary

models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Hist Gradient Boost": HistGradientBoostingClassifier(),
          "Extra Trees": ExtraTreesClassifier(),
          "Decision Tree": DecisionTreeClassifier(),
          "SVC rbf": SVC(kernel='rbf')}

# Create a function to fit and score the models in our dict

def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models. with the given train and test data.
    """
    np.random.seed(42)

    # Create an empty dict to keep model scores
    model_scores = {}

    # Loop through models
    for name, model in models.items():
        # Fit the model
        model.fit(X_train, y_train)
        # Evaluate the model and append it's score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
```


```python
model_scores = fit_and_score(models = models,
                             X_train = X_train,
                             X_test = X_test,
                             y_train = y_train,
                             y_test = y_test)
model_scores
```

    C:\Users\asus\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    {'Logistic Regression': 0.7802690582959642,
     'KNN': 0.6681614349775785,
     'Random Forest': 0.8295964125560538,
     'Hist Gradient Boost': 0.8251121076233184,
     'Extra Trees': 0.8295964125560538,
     'Decision Tree': 0.7354260089686099,
     'SVC rbf': 0.6816143497757847}



# Model Comparison


```python
model_compare = pd.DataFrame(model_scores, index=["Accuracy"])
model_compare.T.plot.bar();
```


    
![png](output_76_0.png)
    


# Now let's dive into Hyperparameter tuning


```python
# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {
    "n_estimators": np.arange(50, 1001, 50),  # More granular steps
    "max_depth": [None, 3, 5, 10, 20, 30],  # Added 30 for better control
    "min_samples_split": [2, 5, 10, 15],  # Balanced values
    "min_samples_leaf": [1, 2, 5, 10]  # Prevent overly small leaves
}

# Set up RandomizedSearchCV
gs_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),  # Ensures reproducibility
    param_distributions=rf_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    n_iter=20,  # Number of random samples (reduces computation)
    verbose=2,  # More detailed output
    random_state=42  # Ensures consistent results
)
gs_rf.fit(X_train, y_train)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    




<style>#sk-container-id-6 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-6 {
  color: var(--sklearn-color-text);
}

#sk-container-id-6 pre {
  padding: 0;
}

#sk-container-id-6 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-6 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-6 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-6 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-6 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-6 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-6 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-6 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-6 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-6 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-6 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-6 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-6 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-6 div.sk-label label.sk-toggleable__label,
#sk-container-id-6 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-6 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-6 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-6 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-6 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-6 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-6 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-6 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-6 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-6 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42),
                   n_iter=20, n_jobs=-1,
                   param_distributions={&#x27;max_depth&#x27;: [None, 3, 5, 10, 20, 30],
                                        &#x27;min_samples_leaf&#x27;: [1, 2, 5, 10],
                                        &#x27;min_samples_split&#x27;: [2, 5, 10, 15],
                                        &#x27;n_estimators&#x27;: array([  50,  100,  150,  200,  250,  300,  350,  400,  450,  500,  550,
        600,  650,  700,  750,  800,  850,  900,  950, 1000])},
                   random_state=42, verbose=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomizedSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html">?<span>Documentation for RandomizedSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomizedSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42),
                   n_iter=20, n_jobs=-1,
                   param_distributions={&#x27;max_depth&#x27;: [None, 3, 5, 10, 20, 30],
                                        &#x27;min_samples_leaf&#x27;: [1, 2, 5, 10],
                                        &#x27;min_samples_split&#x27;: [2, 5, 10, 15],
                                        &#x27;n_estimators&#x27;: array([  50,  100,  150,  200,  250,  300,  350,  400,  450,  500,  550,
        600,  650,  700,  750,  800,  850,  900,  950, 1000])},
                   random_state=42, verbose=2)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: RandomForestClassifier</label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(min_samples_leaf=2, n_estimators=1000, random_state=42)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(min_samples_leaf=2, n_estimators=1000, random_state=42)</pre></div> </div></div></div></div></div></div></div></div></div>




```python
gs_rf.score(X_test, y_test)
```




    0.8295964125560538



## Evaluating our tuned machine learning classifier, beyon accuracy

* ROC curve and AUC score
* Confusion matrix
* Classification report
* Precision
* Recall
* F1-Score


```python
y_preds = gs_rf.predict(X_test)
```


```python
RocCurveDisplay.from_estimator(
   gs_rf, X_test, y_test)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x1c0820582f0>




    
![png](output_82_1.png)
    



```python
# Confusion matrix
print(confusion_matrix(y_test, y_preds))
```

    [[120  14]
     [ 24  65]]
    


```python
sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_test, y_preds)
    
    # Plot heatmap with integer formatting
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

# Call function
plot_conf_mat(y_test, y_preds)
```


    
![png](output_84_0.png)
    



```python
print(classification_report(y_test, y_preds))
```

                  precision    recall  f1-score   support
    
               0       0.83      0.90      0.86       134
               1       0.82      0.73      0.77        89
    
        accuracy                           0.83       223
       macro avg       0.83      0.81      0.82       223
    weighted avg       0.83      0.83      0.83       223
    
    


```python
# Check best hyperparameters
gs_rf.best_params_
```




    {'n_estimators': 1000,
     'min_samples_split': 2,
     'min_samples_leaf': 2,
     'max_depth': None}




```python
# Create a new classifier with best parameters
clf = RandomForestClassifier(n_estimators = 1000,
                        min_samples_split =  2,
                        min_samples_leaf = 2,
                        max_depth = None)
```


```python
# Method for Cross-validated scores
scoring_types = {"accuracy score: ":"accuracy",
                 "precision score: ":"precision",
                 "recall score: ":"recall",
                 "f1 score: ":"f1"
                 }
scoring_results = {}
def cross_validator(model,X,y):
    for name, scoring_t in scoring_types.items():
        cv_scoring = cross_val_score(model,
                             X,
                             y,
                             cv=5,
                             scoring=scoring_t)
        cv_scoring = np.mean(cv_acc)
        scoring_results[name] = cv_scoring
    return scoring_results
```


```python
all_scoring_results = cross_validator(clf,X,y)
```


```python
# Visualize cross-validated metrics
cv_metrics = pd.DataFrame(all_scoring_results,
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                      legend=False);
```


    
![png](output_90_0.png)
    



```python
df_test = pd.read_csv("data/test.csv")
df_test.shape
df_test.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
clf.predict(df_test)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[408], line 1
    ----> 1 clf.predict(df_test)
    

    File ~\anaconda3\Lib\site-packages\sklearn\ensemble\_forest.py:904, in ForestClassifier.predict(self, X)
        883 def predict(self, X):
        884     """
        885     Predict class for X.
        886 
       (...)
        902         The predicted classes.
        903     """
    --> 904     proba = self.predict_proba(X)
        906     if self.n_outputs_ == 1:
        907         return self.classes_.take(np.argmax(proba, axis=1), axis=0)
    

    File ~\anaconda3\Lib\site-packages\sklearn\ensemble\_forest.py:946, in ForestClassifier.predict_proba(self, X)
        944 check_is_fitted(self)
        945 # Check data
    --> 946 X = self._validate_X_predict(X)
        948 # Assign chunk of trees to jobs
        949 n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
    

    File ~\anaconda3\Lib\site-packages\sklearn\ensemble\_forest.py:641, in BaseForest._validate_X_predict(self, X)
        638 else:
        639     force_all_finite = True
    --> 641 X = self._validate_data(
        642     X,
        643     dtype=DTYPE,
        644     accept_sparse="csr",
        645     reset=False,
        646     force_all_finite=force_all_finite,
        647 )
        648 if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
        649     raise ValueError("No support for np.int64 index based sparse matrices")
    

    File ~\anaconda3\Lib\site-packages\sklearn\base.py:608, in BaseEstimator._validate_data(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)
        537 def _validate_data(
        538     self,
        539     X="no_validation",
       (...)
        544     **check_params,
        545 ):
        546     """Validate input data and set or check the `n_features_in_` attribute.
        547 
        548     Parameters
       (...)
        606         validated.
        607     """
    --> 608     self._check_feature_names(X, reset=reset)
        610     if y is None and self._get_tags()["requires_y"]:
        611         raise ValueError(
        612             f"This {self.__class__.__name__} estimator "
        613             "requires y to be passed, but the target y is None."
        614         )
    

    File ~\anaconda3\Lib\site-packages\sklearn\base.py:535, in BaseEstimator._check_feature_names(self, X, reset)
        530 if not missing_names and not unexpected_names:
        531     message += (
        532         "Feature names must be in the same order as they were in fit.\n"
        533     )
    --> 535 raise ValueError(message)
    

    ValueError: The feature names should match those that were passed during fit.
    Feature names unseen at fit time:
    - Cabin
    - Name
    Feature names seen at fit time, yet now missing:
    - CabinCount
    - CabinDeck
    - HasCabin
    



```python
## We have to convert the test dataset into the shape of the training dataset, let's create a function for it
def prepare_data(data):
    data = data.drop("Name",axis=1)
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    # If passenger has no Cabin fill it it U as in Unknown if not fill with the first letter of the cabin
    data["CabinDeck"] = data["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")
    # Count Number of Cabins passenger had if non fill with 0
    data["CabinCount"] = data["Cabin"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    data = data.drop("Cabin", axis=1)
    # Let's convert the Sex aswell
    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "male" else 0)
    for label, content in data.items():
        if pd.api.types.is_object_dtype(content):
            data[label] = content.astype("category").cat.as_ordered()
    for label, content in data.items():
        if not pd.api.types.is_numeric_dtype(content):
            data[label] = pd.Categorical(content).codes + 1
    return data
```


```python
df_test = prepare_data(df_test)
```


```python
predictions = clf.predict(df_test)
```


```python
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```

    Your submission was successfully saved!
    


```python

```
