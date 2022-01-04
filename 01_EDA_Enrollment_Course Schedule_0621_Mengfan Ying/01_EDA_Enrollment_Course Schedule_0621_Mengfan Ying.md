# Exploratory Data Analysis (EDA)

## Import packages


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style("darkgrid")
sns.set(rc={'figure.figsize':(8,6)}) # adjust figure size

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

# Table#1 -- Contact Object
**This is an EDA for Table 'Contact Object'**
*Basic contact information about each Fellow*


```python
#read the csv document
contact = pd.read_csv('DP- Contact Object 5.10.21.csv')
```

## 1) Check rows and cols


```python
#contact.head()
# contact.info()
#contact.columns
```

## 1) conclusions:
- there are 1524 rows x 51 columns in this table
- Id is the primary key
- Datatype: object, float, int

***

## 2) Check missing data


```python
sns.set(rc={'figure.figsize':(12,9)}) # adjust figure size
sns.heatmap(contact.isnull(), cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb32445cfd0>




![png](output_10_1.png)



```python
total = contact.isnull().sum().sort_values(ascending=False)
percent = (contact.isnull().sum()/contact.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
missing_data.head(5) 
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
      <th>Total Missing</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Semester_11_Credits__c</td>
      <td>1522</td>
      <td>0.998688</td>
    </tr>
    <tr>
      <td>Semester_12_Credits__c</td>
      <td>1517</td>
      <td>0.995407</td>
    </tr>
    <tr>
      <td>Semester_9_Credits__c</td>
      <td>1497</td>
      <td>0.982283</td>
    </tr>
    <tr>
      <td>Semester_10_Credits__c</td>
      <td>1491</td>
      <td>0.978346</td>
    </tr>
    <tr>
      <td>Latest_FAFSA_Date__c</td>
      <td>1486</td>
      <td>0.975066</td>
    </tr>
  </tbody>
</table>
</div>



## 2) conclusions:
- some cols (especially semester_credits) contain a lot of missing values (90%+)
- some fellows(students) lost most of the info

## 2) questions and further work:
- are semester_credits true missing values? Is it because those students are still in the program?

***

## 3) Basic Visualizations

### - Countplots
__[reference](https://seaborn.pydata.org/generated/seaborn.countplot.html)__


```python
# how many fellows in each ethical group
sns.set(rc={'figure.figsize':(8,6)}) # adjust figure size
ax = sns.countplot(x="Ethnicity__c", data=contact)

plt.xlabel("ethnicity", fontsize= 12)
plt.ylabel("fellow number", fontsize= 12)
plt.title("students in different ethnicities", fontsize= 15)
```




    Text(0.5, 1.0, 'students in different ethnicities')




![png](output_17_1.png)



```python
# fellow population distribution -- gender & ethnicity
sns.set(rc={'figure.figsize':(12,9)}) # adjust figure size
g = sns.catplot(x="Ethnicity__c", hue="Gender__c", col="First_Generation_College_Student__c",
                data=contact, kind="count",
                height=6, aspect=1.2);


g.set_xlabels('ethnicity', fontsize=15) # not set_label
g.set_ylabels('student number', fontsize=15)
```




    <seaborn.axisgrid.FacetGrid at 0x7f8a59f51750>




![png](output_18_1.png)


### - Conclusions from countplot:
- most of the students are African Americans, Hispanic is the second most, very few are Native Americans or White
- first generation college student or not are pretty evenly distributed
- Female students are more than male students

### - Histograms
__[reference](https://seaborn.pydata.org/generated/seaborn.histplot.html)__


```python
# this is a distribution of high school final GPA
sns.set(rc={'figure.figsize':(8,6)}) # adjust figure size
sns.histplot(data=contact, x="HS_Final_GPA__c", kde=True)
# replace with 'Highest_ACT_Score__c'

plt.xlabel("High school Final GPA", fontsize= 12)
plt.ylabel("student number", fontsize= 12)
plt.title("GPA distribution", fontsize= 15)
```




    Text(0.5, 1.0, 'GPA distribution')




![png](output_21_1.png)



```python
# this is a stack distribution of highest ACT score, male vs female
sns.histplot(data=contact, x="Highest_ACT_Score__c", hue="Gender__c", multiple="stack")
# replace with 'Ethnicity__c', 'First_Generation_College_Student__c', 'Active_T_Mobile_Plan__c'

plt.xlabel("Highest ACT", fontsize= 12)
plt.ylabel("student number", fontsize= 12)
plt.title("ACT distribution(gender Included)", fontsize= 15)
```




    Text(0.5, 1.0, 'ACT distribution(gender Included)')




![png](output_22_1.png)



```python
# this is a distribution of how much money family can contribute to college tuition
sns.set(rc={'figure.figsize':(8,6)}) # adjust figure size
sns.histplot(data=contact, x="EFC_from_FAFSA__c", kde=True)

plt.xlabel("EFC number", fontsize= 12)
plt.ylabel("student number", fontsize= 12)
plt.title("EFC number distribution", fontsize= 15)
```




    Text(0.5, 1.0, 'EFC number distribution')




![png](output_23_1.png)


#### An EFC number is the "expected family contribution", or the amount a family is expected to pay for their student's college education. It short, the EFC has an effect on how much federal grant money you will be given for college expenses.

### - Conclusions from histogram:
- the grades (ACT&GPA) are normally distributed
- The normal distribution is not affected greatly by gender, ethnics, first generation or not
- student who gets the lowest and highest ACT are both female
- most students have financial problems that their family cannot afford their college tuition, but there are some outliers

### - Boxplot
__[reference](https://seaborn.pydata.org/generated/seaborn.boxplot.html)__


```python
# high school final GPA in different ethnical groups 
sns.boxplot(x="Ethnicity__c", y="Highest_ACT_Score__c", data=contact)
# replace with 'Highest_ACT_Score__c','HS_Final_GPA__c'

plt.xlabel("ethnicity", fontsize= 12)
plt.ylabel("Highest ACT", fontsize= 12)
plt.title("highest ACT of each ethnical group", fontsize= 15)
```




    Text(0.5, 1.0, 'highest ACT of each ethnical group')




![png](output_27_1.png)



```python
# highest ACT in different ethnical groups 
sns.set(rc={'figure.figsize':(8,6)}) # adjust figure size
sns.boxplot(x="Ethnicity__c", y="Highest_ACT_Score__c", hue="First_Generation_College_Student__c", data=contact, linewidth=2.5)

plt.xlabel("ethnicity", fontsize= 12)
plt.ylabel("Highest ACT", fontsize= 12)
plt.title("ethnicity vs ACT (first generation college student Included)", fontsize= 15)
```




    Text(0.5, 1.0, 'ethnicity vs ACT (first generation college student Included)')




![png](output_28_1.png)


### - Conclusions from boxplot:
- due to lack of data, only look into the first three ethnical groups
- hispanics have better academic performance
- first generation college students have better academic performance

### - Scatterplot
__[reference]( https://seaborn.pydata.org/generated/seaborn.boxplot.html)__


```python
# ACT VS high school GPA
sns.scatterplot(data=contact, x="Highest_ACT_Score__c", y="HS_Final_GPA__c")

plt.xlabel("highest ACT", fontsize= 12)
plt.ylabel("High school final GPA", fontsize= 12)
plt.title("highest ACT VS high school final GPA", fontsize= 15)
```




    Text(0.5, 1.0, 'highest ACT VS high school final GPA')




![png](output_31_1.png)



```python
# credits taken VS college GPA
sns.scatterplot(data=contact, x="Semester_1_Credits__c", y="Semester_1_GPA__c", hue="First_Generation_College_Student__c")

plt.xlabel("semester 1 credits", fontsize= 12)
plt.ylabel("semester 1 GPA", fontsize= 12)
plt.title("credits VS GPA (semester 1)", fontsize= 15)
```




    Text(0.5, 1.0, 'credits VS GPA (semester 1)')




![png](output_32_1.png)


### - Conclusions from scatterplot:
- positive relationship between ACT and high school GPA
- in the first semester: more credits taken, higher GPA in general
- first generation college students work harder (6 first generation students get 4 credits vs 2 non-first generation get 4 credits)

***

# Table#2 -- Contact Note
**This is an EDA for Table 'Contact Note'**
*Documents each contact between coach and Fellow*


```python
#read the csv document
contactNote = pd.read_csv('DP- Contact_Note_c 5.10.21.csv')
```

## 1) Check rows and cols


```python
#contactNote.head()
# contactNote.info()
#contactNote.columns
```

## 1) conclusions:
- there are 25947 rows x 6 columns in this table
- Contact__c is the primary key
- Datatype: object, int(binary)

## 2) Check missing data


```python
#sns.set(rc={'figure.figsize':(12,9)}) # adjust figure size
sns.heatmap(contactNote.isnull(), cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8a46d473d0>




![png](output_41_1.png)



```python
total = contactNote.isnull().sum().sort_values(ascending=False)
percent = (contactNote.isnull().sum()/contactNote.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
missing_data
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
      <th>Total Missing</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Discussion_Category__c</td>
      <td>12524</td>
      <td>0.482676</td>
    </tr>
    <tr>
      <td>How</td>
      <td>1036</td>
      <td>0.039928</td>
    </tr>
    <tr>
      <td>Comm_Status_c__c</td>
      <td>492</td>
      <td>0.018962</td>
    </tr>
    <tr>
      <td>Date_of_Contact__c</td>
      <td>28</td>
      <td>0.001079</td>
    </tr>
    <tr>
      <td>Initiated_by_alum__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Contact__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 2) conclusions:
- some cols contain some missing values
- some fellows(students) lost many cols info

## 3) Basic Visualizations

### - Countplots
__[reference](https://seaborn.pydata.org/generated/seaborn.countplot.html)__


```python
#  if contact between coach and student was successful
ax = sns.countplot(x="Comm_Status_c__c", hue='Initiated_by_alum__c', data=contactNote)

plt.xlabel("communication status", fontsize= 12)
plt.ylabel("count", fontsize= 12)
plt.title("number of different communication status", fontsize= 15)
plt.legend(title='student initiated', loc='upper right')
# 1--> student initiate the contact
```




    <matplotlib.legend.Legend at 0x7f8a4c423690>




![png](output_46_1.png)



```python
#  Nature of conversation
ax = sns.countplot(y="Discussion_Category__c", hue='Initiated_by_alum__c', data=contactNote,
                  order = contactNote['Discussion_Category__c'].value_counts().index) # ordered
#plt.xticks(rotation=20)

plt.ylabel("discussion category", fontsize= 12)
plt.xlabel("count", fontsize= 12)
plt.title("number of conversations in each category", fontsize= 15)
plt.legend(title='student initiated', loc='lower right')
# 1--> student initiate the contact
```




    <matplotlib.legend.Legend at 0x7f8a5aea11d0>




![png](output_47_1.png)



```python
#  How did the coach and fellow make contact
ax = sns.countplot(y="How ", hue='Initiated_by_alum__c', data=contactNote,
                  order = contactNote['How '].value_counts().index) # ordered
#plt.xticks(rotation=45)

plt.ylabel("contact method", fontsize= 12)
plt.xlabel("count", fontsize= 12)
plt.title("number of conversations in each method", fontsize= 15)
plt.legend(title='student initiated', loc='lower right')
# 1--> student initiate the contact
```




    <matplotlib.legend.Legend at 0x7f8a5bfcbf90>




![png](output_48_1.png)


## conclusion from countplot:
- all conversations initiated by students were successful
- the number of contacts initiated by a coach is way more than contacts initiated by a student
- students won't take the initiative to talk about career or family
- academic is the most common topic except for general outreach
- text is the most popular contact method

### - time series data
#### before plotting, extract and create two new variables -- year & month


```python
contactNote['year'] = pd.DatetimeIndex(contactNote['Date_of_Contact__c']).year
contactNote['month'] = pd.DatetimeIndex(contactNote['Date_of_Contact__c']).month

contactNote.head()
# it is float type because due to NAs in this col
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
      <th>Contact__c</th>
      <th>Comm_Status_c__c</th>
      <th>Date_of_Contact__c</th>
      <th>Discussion_Category__c</th>
      <th>Initiated_by_alum__c</th>
      <th>How</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>00346000002iXe4AAE</td>
      <td>Successful communication</td>
      <td>9/7/2018 0:00</td>
      <td>NaN</td>
      <td>0</td>
      <td>Social Networking</td>
      <td>2018.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>00346000002iXdxAAE</td>
      <td>Successful communication</td>
      <td>10/10/2018 0:00</td>
      <td>Academic</td>
      <td>0</td>
      <td>In Person</td>
      <td>2018.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0034600000iXkmFAAS</td>
      <td>Successful communication</td>
      <td>8/30/2018 0:00</td>
      <td>Academic</td>
      <td>0</td>
      <td>Call</td>
      <td>2018.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>00346000002iXduAAE</td>
      <td>Successful communication</td>
      <td>8/31/2018 0:00</td>
      <td>Social &amp; Academic Integration</td>
      <td>0</td>
      <td>Call</td>
      <td>2018.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0034600001EXcSPAA1</td>
      <td>Successful communication</td>
      <td>8/28/2018 0:00</td>
      <td>Academic</td>
      <td>0</td>
      <td>Text</td>
      <td>2018.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#  conversations through out years
ax = sns.countplot(x="year", data=contactNote)

plt.xlabel("year", fontsize= 12)
plt.ylabel("count", fontsize= 12)
plt.title("number of conversations throug out years", fontsize= 15)
```




    Text(0.5, 1.0, 'number of conversations throug out years')




![png](output_52_1.png)



```python
#  conversations through out months
ax = sns.countplot(x="month", data=contactNote)

plt.xlabel("month", fontsize= 12)
plt.ylabel("count", fontsize= 12)
plt.title("number of conversations throug out months", fontsize= 15)
```




    Text(0.5, 1.0, 'number of conversations throug out months')




![png](output_53_1.png)


## conclusion from time series data:
- the records are mostly between 2013-2020
- the contacts took place mostly during spring or fall semesters, not in the summer (July has the fewest contacts)

***

# Table#3 -- Course schedule
**This is an EDA for Table 'Course schedule'**
*Course details collected from Fellows*


```python
#read the csv document
courseSchedule = pd.read_csv('DP- Course_Shedule_c 5.10.21.csv', encoding= 'unicode_escape')
```

## 1) Check rows and cols


```python
# courseSchedule.head()
courseSchedule.info()
# courseSchedule.columns
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5101 entries, 0 to 5100
    Data columns (total 12 columns):
    Student__c              5101 non-null object
    Id                      5101 non-null object
    CreatedDate             5101 non-null object
    Course__c               5081 non-null object
    Course_End_Time__c      2125 non-null object
    Course_Start_Time__c    2127 non-null object
    Course_Title__c         5087 non-null object
    Course_Type__c          3066 non-null object
    Credit_Hours__c         4308 non-null float64
    Gen_Ed_Category__c      1214 non-null object
    Semester_Quarter__c     4660 non-null object
    Date_Course_Taken__c    4405 non-null object
    dtypes: float64(1), object(11)
    memory usage: 478.3+ KB



```python
uniqueCourse = courseSchedule['Course__c'].nunique(dropna=True)
print (f'the number of unique courses is {uniqueCourse}')

uniqueCourseTitle = courseSchedule['Course_Title__c'].nunique(dropna=True)
print (f'the number of unique courses titles is {uniqueCourseTitle}')
```

    the number of unique courses is 3009
    the number of unique courses titles is 2814


## 1) conclusions:
- there are 5101 rows x 12 columns in this table
- Id is the primary key
- Datatype: object, float
- course and course_title require dimension reduction (keep one or fewer categories?)
- semester_quarter needs to be cleaned up

## 2) Check missing data


```python
# sns.set(rc={'figure.figsize':(12,9)}) # adjust figure size
# sns.heatmap(courseSchedule.isnull(), cbar=False)

msno.matrix(courseSchedule)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff924a55390>




![png](output_63_1.png)



```python
total = courseSchedule.isnull().sum().sort_values(ascending=False)
percent = (courseSchedule.isnull().sum()/courseSchedule.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
missing_data
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
      <th>Total Missing</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gen_Ed_Category__c</td>
      <td>3887</td>
      <td>0.762007</td>
    </tr>
    <tr>
      <td>Course_End_Time__c</td>
      <td>2976</td>
      <td>0.583415</td>
    </tr>
    <tr>
      <td>Course_Start_Time__c</td>
      <td>2974</td>
      <td>0.583023</td>
    </tr>
    <tr>
      <td>Course_Type__c</td>
      <td>2035</td>
      <td>0.398941</td>
    </tr>
    <tr>
      <td>Credit_Hours__c</td>
      <td>793</td>
      <td>0.155460</td>
    </tr>
    <tr>
      <td>Date_Course_Taken__c</td>
      <td>696</td>
      <td>0.136444</td>
    </tr>
    <tr>
      <td>Semester_Quarter__c</td>
      <td>441</td>
      <td>0.086454</td>
    </tr>
    <tr>
      <td>Course__c</td>
      <td>20</td>
      <td>0.003921</td>
    </tr>
    <tr>
      <td>Course_Title__c</td>
      <td>14</td>
      <td>0.002745</td>
    </tr>
    <tr>
      <td>CreatedDate</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Id</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Student__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 2) conclusions:
- some cols contain a lot of missing values (50%+)

## 3) Basic Visualizations

### - Countplots
__[reference](https://seaborn.pydata.org/generated/seaborn.countplot.html)__


```python
# different course types
# sns.set(rc={'figure.figsize':(8,6)}) # adjust figure size
ax = sns.countplot(x="Course_Type__c", data=courseSchedule)

plt.xlabel("course type", fontsize= 16)
plt.ylabel("count", fontsize= 16)
plt.title("Course types", fontsize= 18)
```




    Text(0.5, 1.0, 'Course types')




![png](output_68_1.png)


### - Conclusions from countplot:
- most of the courses are traditional type

***

# Table#4 -- Enrollment
**This is an EDA for Table 'Enrollment**
*Tracking each Fellow's enrollment college*


```python
#read the csv document
enrollment = pd.read_csv('DP- Enrollment_c 5.10.21.csv')
```

## 1) Check rows and cols


```python
#enrollment.head()
enrollment.info()
#enrollment.columns
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2061 entries, 0 to 2060
    Data columns (total 11 columns):
    Student__c               2061 non-null object
    Id                       2061 non-null object
    CreatedDate              2061 non-null object
    College__c               2061 non-null object
    Date_Last_Verified__c    1925 non-null object
    Degree_Type__c           1998 non-null object
    End_Date__c              1520 non-null object
    Major_Text__c            162 non-null object
    Start_Date__c            2002 non-null object
    Status__c                2059 non-null object
    Withdrawal_code__c       634 non-null object
    dtypes: object(11)
    memory usage: 177.2+ KB



```python
# check the number of unique rows using 'Student__c'
len(enrollment['Student__c'].drop_duplicates())
```




    1142




```python
# check withdraw reasons
enrollment['Withdrawal_code__c'].unique()
```




    array([nan, 'Summer Academics', 'Academic', 'Academic;Health',
           'Academic;Motivational', 'Family', 'Family;Motivational',
           'Unknown', 'Financial', 'Financial;Social', 'Academic;Financial',
           'Motivational', 'Academic;Family', 'Financial;Motivational',
           'Social', 'Family;Financial;Motivational', 'Health',
           'Family;Financial', 'Health;Motivational',
           'Academic;Financial;Motivational', 'Social;Racial Conflict',
           'Family;Health', 'Academic;Family;Financial',
           'Academic;Family;Motivational', 'Academic;Financial;Social',
           'Academic;Social', 'Suspended (Academic)', 'Expelled (Academic)',
           'Expelled (Behavioral)', 'Social;Suspended (Behavioral)',
           'Financial;Health;Motivational', 'Motivational;Social',
           'Family;Motivational;Social', 'Academic;Family;Financial;Health',
           'Academic;Family;Health', 'Family;Social',
           'Health;Motivational;Social;Racial Conflict',
           'Motivational;Racial Conflict', 'Suspended (Behavioral)',
           'Family;Health;Motivational;Social'], dtype=object)



## 1) conclusions:
- there are 2061 rows x 11 columns in this table
- though the number of studentID = the number of enrollmentID --> some students have more than one enrollment
- Id is the primary key
- Datatype: object
- for end_date and start_date --> need to create a new feature: end-start = duration
- 'College__c' is not a foreign key. useless?
- 'Status__c' may not accurate (according to the dictionary)
- 'Withdrawal_code__c' info overlapped --> split and count the times of each one has been mentioned

## 2) Check missing data


```python
#sns.heatmap(enrollment.isnull(), cbar=False)
import missingno as msno
msno.matrix(enrollment)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff92423a210>




![png](output_79_1.png)



```python
total = enrollment.isnull().sum().sort_values(ascending=False)
percent = (enrollment.isnull().sum()/enrollment.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
missing_data
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
      <th>Total Missing</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Major_Text__c</td>
      <td>1899</td>
      <td>0.921397</td>
    </tr>
    <tr>
      <td>Withdrawal_code__c</td>
      <td>1427</td>
      <td>0.692382</td>
    </tr>
    <tr>
      <td>End_Date__c</td>
      <td>541</td>
      <td>0.262494</td>
    </tr>
    <tr>
      <td>Date_Last_Verified__c</td>
      <td>136</td>
      <td>0.065987</td>
    </tr>
    <tr>
      <td>Degree_Type__c</td>
      <td>63</td>
      <td>0.030568</td>
    </tr>
    <tr>
      <td>Start_Date__c</td>
      <td>59</td>
      <td>0.028627</td>
    </tr>
    <tr>
      <td>Status__c</td>
      <td>2</td>
      <td>0.000970</td>
    </tr>
    <tr>
      <td>College__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>CreatedDate</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Id</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Student__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 2) conclusions:
- some cols (especially Major_Text__c) contain a lot of missing values (90%+), but we can get the info through connecting this table with the table 'contact object' which contains the info about major (I tried through excel and found that some can match)

## 3) Basic Visualizations

### - Countplots
__[reference](https://seaborn.pydata.org/generated/seaborn.countplot.html)__


```python
# degree types
ax = sns.countplot(y="Degree_Type__c", 
                   order = enrollment['Degree_Type__c'].value_counts().index,
                   data=enrollment)

#plt.xticks(rotation=45)
plt.ylabel("degree types", fontsize= 16)
plt.xlabel("student number", fontsize= 16)
plt.title("Degree types", fontsize= 18)
```




    Text(0.5, 1.0, 'Degree types')




![png](output_84_1.png)



```python
# status
ax = sns.countplot(y="Status__c", 
                   order = enrollment['Status__c'].value_counts().index,
                   data=enrollment)

#plt.xticks(rotation=45)
plt.ylabel("status", fontsize= 16)
plt.xlabel("student number", fontsize= 16)
plt.title("Student status", fontsize= 18)
```




    Text(0.5, 1.0, 'Student status')




![png](output_85_1.png)


## 3) conclusions:
- most students are pursuing bachelor's degree
- very few are pursuing Master or certificate(decided)

***

# Table#5 -- Support
**This is an EDA for Table 'Support'**
*Data tracking additional supports for Fellows for one-year (2018)*


```python
#read the csv document
support = pd.read_csv('DP- Academic_Support_c 5.10.21.csv')
```

## 1) Check rows and cols


```python
#support.head()
support.info()
#support.columns
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 259 entries, 0 to 258
    Data columns (total 7 columns):
    Student__c             259 non-null object
    Id                     259 non-null object
    CreatedDate            259 non-null object
    Location__c            259 non-null object
    Role__c                259 non-null object
    Purpose_of_Visit__c    259 non-null object
    Course_type__c         255 non-null object
    dtypes: object(7)
    memory usage: 14.3+ KB



```python
# check purpose of visit
support['Purpose_of_Visit__c'].unique()
```




    array(['Help with homework to be handed in', 'Other',
           'Study for an upcoming quiz or test', 'Career advice or planning',
           'Academic_advice_or_planning', 'General life counseling',
           'Revise or edit a paper', 'Review past homework, quizzes or tests'],
          dtype=object)



## 1) conclusions:
- there are 259 rows x 7 columns in this table
- Id is the primary key
- Datatype: object

## 2) Check missing data


```python
sns.heatmap(support.isnull(), cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8a4be98f50>




![png](output_95_1.png)



```python
total = support.isnull().sum().sort_values(ascending=False)
percent = (support.isnull().sum()/support.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
missing_data
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
      <th>Total Missing</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Course_type__c</td>
      <td>4</td>
      <td>0.015444</td>
    </tr>
    <tr>
      <td>Purpose_of_Visit__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Role__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Location__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>CreatedDate</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Id</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Student__c</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 2) conclusions:
- only 4 missing values in course type

## 3) Basic Visualizations
- mainly categorical variables, plot some countplots


```python
# course support
ax = sns.countplot(x="Course_type__c", hue='Role__c', 
                   order = support['Course_type__c'].value_counts().index,
                   data=support)

plt.xlabel("course type", fontsize= 12)
plt.ylabel("count", fontsize= 12)
plt.title("course support from different roles", fontsize= 15)
plt.legend(loc='upper right')
```




    <matplotlib.legend.Legend at 0x7f8a4b6abe50>




![png](output_99_1.png)



```python
# purpose of visit
ax = sns.countplot(y="Purpose_of_Visit__c", hue='Location__c',
                   order = support['Purpose_of_Visit__c'].value_counts().index,
                   data=support)

plt.ylabel("purpose", fontsize= 12)
plt.xlabel("student number", fontsize= 12)
plt.title("purpose of visit", fontsize= 15)
#plt.legend(loc='upper right')
```




    Text(0.5, 1.0, 'purpose of visit')




![png](output_100_1.png)


## 3) conclusions:
- students needs more help on math and science than social science or humanity classes. For math and science, they prefer to ask tutor and prof for help. For others, they prefer to talk to their advisor.
- Foreign language is the least popular subject when it comes to asking for out-of-class help
- Students tend to ask for help before turning in assignment or taking a test
- when it comes to reviewing a test, they tend to ask prof for help
- they go to a non-academic support office only when they want to ask for general life counseling or for other purpose. 

# Table#6 -- Survey Response
**This is an EDA for Table 'Survey_Response'**
*Data pulled from our Salesforce based survey. First data collected Fall 2020.*


```python
#read the csv document
surveyResponse = pd.read_csv('DP- Survey_Response_c 5.10.21.csv', encoding= 'unicode_escape')
```

## 1) Check rows and cols


```python
# surveyResponse.head()
# surveyResponse.info()
# surveyResponse.columns
```


```python
# check unique students who took the survey
uniqueContact = surveyResponse['Contact__c'].nunique(dropna=True)
print (f'the number of unique contact is {uniqueContact}')
```

    the number of unique contact is 135



```python
# some students took the survey several times in a short time (within a few days) but submit different answers
surveyResponse.loc[surveyResponse['Contact__c'] == '0034p00001f3nlWAAQ']
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
      <th>Contact__c</th>
      <th>Id</th>
      <th>CreatedDate</th>
      <th>19. The key to academic success in college is:</th>
      <th>12.Think about last month. In an average week, how many times did you participate in extracurricular activities?</th>
      <th>18.How true is the statement: I am able to afford all my expenses</th>
      <th>9.How True is the statement: I am confident that I can succeed in college</th>
      <th>25.Please select the statement that best describes your status for your Alumni Coordinator right now</th>
      <th>3. How true is the statement: I feel accepted by other students on campus</th>
      <th>1. How true is the statement: I feel at-home on campus</th>
      <th>...</th>
      <th>22. The key to success in college courses like math and science is:</th>
      <th>Survey__c</th>
      <th>15. Think about last month. In an average week, how many hours did you expand your education in any manner?</th>
      <th>9. How True is the statement: I am confident that I can achieve financial independence</th>
      <th>8. How True is the statement: I am happy at this point in my life</th>
      <th>20. How true is the statement: I am able to find a quiet place to unwind when I need it</th>
      <th>5. How true is the statement: I feel like I have a plan to reach my goals</th>
      <th>16. Think about last month. In an average week, how many hours did you committment to self-improvement? (reading, course work, meditation)</th>
      <th>14. Think about last month. In an average week, how many times were you absent from work or training?</th>
      <th>KPI_Metric_2__c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20</td>
      <td>0034p00001f3nlWAAQ</td>
      <td>a1T4p000002ttaTEAQ</td>
      <td>9/9/2020 21:15</td>
      <td>Equal parts being smart and hard work</td>
      <td>None</td>
      <td>Neither True Nor False</td>
      <td>Mostly True</td>
      <td>I'm ok, but let's check in as usual.</td>
      <td>Neither True Nor False</td>
      <td>Mostly True</td>
      <td>...</td>
      <td>Equal parts being smart and hard work</td>
      <td>College Success Survey</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.00</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0034p00001f3nlWAAQ</td>
      <td>a1T4p000002ttfTEAQ</td>
      <td>9/10/2020 16:39</td>
      <td>Equal parts being smart and hard work</td>
      <td>None</td>
      <td>Neither True Nor False</td>
      <td>Very True</td>
      <td>I'm ok, but let's check in as usual.</td>
      <td>Neither True Nor False</td>
      <td>Mostly True</td>
      <td>...</td>
      <td>Equal parts being smart and hard work</td>
      <td>College Success Survey</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.75</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0034p00001f3nlWAAQ</td>
      <td>a1T4p000002tthUEAQ</td>
      <td>9/10/2020 21:04</td>
      <td>Equal parts being smart and hard work</td>
      <td>None</td>
      <td>Neither True Nor False</td>
      <td>Very True</td>
      <td>I'm ok, but let's check in as usual.</td>
      <td>Mostly True</td>
      <td>Mostly True</td>
      <td>...</td>
      <td>Equal parts being smart and hard work</td>
      <td>College Success Survey</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.50</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 37 columns</p>
</div>



## 1) conclusions:
- there are 589 rows x 37 columns in this table
- Id is the primary key
- Datatype: float, object
- some students took the survey several times in a short time giving different answers

## 2) Check missing data


```python
sns.heatmap(surveyResponse.isnull(), cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8a45db2ad0>




![png](output_110_1.png)


## 2) conclusion:
- for the last a few couples of surveys, questions 15,9,8,20,5,16,14 are changed

## 3) Basic Visualizations

### - Boxplot
__[reference](https://seaborn.pydata.org/generated/seaborn.boxplot.html)__


```python
# KPI
sns.boxplot(y="KPI_Metric_2__c", data=surveyResponse)

plt.ylabel("KPI", fontsize= 12)
plt.title("boxplot for KPI metric", fontsize= 15)
```




    Text(0.5, 1.0, 'boxplot for KPI metric')




![png](output_114_1.png)


#### key performance indicator (KPI) is a type of performance measurement. KPIs evaluate the success of an organization or of a particular activity in which it engages.

## 3) conclusion:
- I guess the KPI is calculated from the survey to indicate student success. The distribution is pretty normal. The median is around 3. The highest score is 5, while the lowest score is around .25.

***

# Table#7 --  App Response
**This is an EDA for Table '*App Response'**
*Data pulled from the UtmostU App monthy survey. The questions shifted over time. Data collected 2017 to Spring 2020.*


```python
#read the csv document
AppSurvey = pd.read_csv('DP- App-Survey_c 5.10.21.csv')
```

## 1) Check rows and cols


```python
# AppSurvey.head()
# AppSurvey.info()
# AppSurvey.columns
```


```python
# check unique students who took the survey
uniqueStudent = AppSurvey['Student__c'].nunique(dropna=True)
print (f'the number of unique student is {uniqueStudent}')
```

    the number of unique student is 297



```python
# many students took the survey many times
# the student '0034600000B1BvoAAF' took the survey 21 in total.
AppSurvey.loc[AppSurvey['Student__c'] == '0034600000B1BvoAAF']
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
      <th>Student__c</th>
      <th>Id</th>
      <th>IsDeleted</th>
      <th>Name</th>
      <th>CreatedDate</th>
      <th>1. How true is the statement: I feel at-home on campus</th>
      <th>2. How true is the statement: I feel physically safe on campus</th>
      <th>3. How true is the statement:  I feel accepted by other students on campus</th>
      <th>4. How true is the statement: I feel respected by my professors</th>
      <th>5. I feel like I made the right decision to attend this school</th>
      <th>...</th>
      <th>22. The key to success in college courses like math and science is</th>
      <th>22.How true is the statement:  I am able to find a quiet place to study when I need it</th>
      <th>15.Think about last month. In an average week, how many hours did you study in a group.3</th>
      <th>23. The key to success in college courses like English, History, and language, is</th>
      <th>23.Please select the statement that best describes your status  for your Alumni Coordinator right now</th>
      <th>23.How true is the statement: I have a place I can hang out comfortably when I?m not studying</th>
      <th>15.Think about last month. In an average week, how many hours did you study in a group.4</th>
      <th>24.How true is the statement:  I am able to find a quiet place to study when I need it</th>
      <th>15.Think about last month. In an average week, how many hours did you study in a group.5</th>
      <th>25.Please select the statement that best describes your status  for your Alumni Coordinator right now</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c46000002ZNjTAAW</td>
      <td>0</td>
      <td>New Survey 2017-09-05 19:01</td>
      <td>9/5/17 19:00</td>
      <td>1.How true is the statement: I feel at-home on...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement: I feel accepted ...</td>
      <td>4. How true is the statement: I feel respected...</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1280</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c46000003nvIYAAY</td>
      <td>0</td>
      <td>Student Survey 2017-09-10 21:56</td>
      <td>9/10/17 21:56</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1700</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c46000005E6QEAA0</td>
      <td>0</td>
      <td>Student Survey 2018-01-10 14:58</td>
      <td>1/10/18 14:57</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1962</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000ApAmiAAF</td>
      <td>0</td>
      <td>Student Survey 2018-12-12 19:00</td>
      <td>12/12/18 18:59</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>1963</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000B4dDDAAZ</td>
      <td>0</td>
      <td>Student Survey 2019-03-26 13:58</td>
      <td>3/26/19 13:58</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>1964</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4p00000BfD17AAF</td>
      <td>0</td>
      <td>Student Survey 2019-09-09 21:57</td>
      <td>9/9/19 21:58</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>1965</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4p00000BfDBuAAN</td>
      <td>0</td>
      <td>Student Survey 2019-09-10 15:55</td>
      <td>9/10/19 15:56</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>1966</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4p00000BfKVvAAN</td>
      <td>0</td>
      <td>Student Survey 2019-09-30 21:49</td>
      <td>9/30/19 21:49</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2120</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000Bnb5lAAB</td>
      <td>0</td>
      <td>Student Survey 2019-05-23 16:02</td>
      <td>5/23/19 16:02</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2166</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000ApAnEAAV</td>
      <td>0</td>
      <td>Student Survey 2018-12-12 19:03</td>
      <td>12/12/18 19:03</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2360</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000B3s1HAAR</td>
      <td>0</td>
      <td>Student Survey 2019-02-25 22:01</td>
      <td>2/25/19 22:01</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2499</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000Ap1JqAAJ</td>
      <td>0</td>
      <td>Student Survey 2018-12-07 22:16</td>
      <td>12/7/18 22:15</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2500</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000ApAnXAAV</td>
      <td>0</td>
      <td>Student Survey 2018-12-12 19:05</td>
      <td>12/12/18 19:04</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2501</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000AP3f0AAD</td>
      <td>0</td>
      <td>Student Survey 2019-04-23 04:10</td>
      <td>4/23/19 4:10</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2572</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000BXca3AAD</td>
      <td>0</td>
      <td>Student Survey 2019-02-04 23:22</td>
      <td>2/4/19 23:22</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2637</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000ApAmnAAF</td>
      <td>0</td>
      <td>Student Survey 2018-12-12 19:01</td>
      <td>12/12/18 19:00</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2766</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000AnwrNAAR</td>
      <td>0</td>
      <td>Student Survey 2018-10-30 20:01</td>
      <td>10/30/18 20:00</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2913</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c46000008RbdTAAS</td>
      <td>0</td>
      <td>Student Survey 2018-09-10 03:14</td>
      <td>9/10/18 3:14</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>2964</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000ApAmxAAF</td>
      <td>0</td>
      <td>Student Survey 2018-12-12 19:02</td>
      <td>12/12/18 19:01</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>3260</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c4600000AlWT5AAN</td>
      <td>0</td>
      <td>Student Survey 2018-10-01 20:26</td>
      <td>10/1/18 20:26</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>22. The key to success in college courses like...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.How true is the statement: I have a place I...</td>
      <td>NaN</td>
      <td>24.How true is the statement:  I am able to fi...</td>
      <td>NaN</td>
      <td>25.Please select the statement that best descr...</td>
    </tr>
    <tr>
      <td>3451</td>
      <td>0034600000B1BvoAAF</td>
      <td>a0c46000008R6G7AAK</td>
      <td>0</td>
      <td>Student Survey 2018-08-28 02:47</td>
      <td>8/28/18 2:47</td>
      <td>1. How true is the statement: I feel at-home o...</td>
      <td>2. How true is the statement: I feel physicall...</td>
      <td>3. How true is the statement:  I feel accepted...</td>
      <td>4. How true is the statement:  I feel respecte...</td>
      <td>5. I feel like I made the right decision to at...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>21 rows × 65 columns</p>
</div>




```python

```

## 1) conclusion:
- there are 3455 rows x 65 columns in this table
- Id is the primary key
- Datatype: object
- some students took the survey several times in a short time giving different answers
- it contains more info from 2017-2020, but needs to be cleaned (use regex to extract answers)
- many students take this svurvey many times, some submission dates are very close

## 2) Check missing data


```python
sns.heatmap(AppSurvey.isnull(), cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8b3742fd50>




![png](output_127_1.png)


## 2) conclusions:
- some questions are repeated with no answers
- need to reorganize the table

***

# Table#8 -- programming
**This is an EDA for Table 'programming'**
*Documents the support programs a Fellows engaged with (Emergency Funds; Scholarships; Recovery Credits))*


```python
#read the csv document
programming = pd.read_csv('DP- UtmostU_Programming_c 5.10.21.csv')
```

## 1) Check rows and cols


```python
# programming.head()
# programming.info()
# programming.columns
```


```python
# check unique students are in the program
uniqueSturent = programming['Student__c'].nunique(dropna=True) # note NAs
print (f'the number of unique students is {uniqueSturent}')
```

    the number of unique students is 222


## 1) conclusions:
- there are 335 rows x 9 columns in this table
- Id is the primary key
- Datatype: num, str
- some students are in the program several times?
- don't quite understand... summer program & non-summer program? see the original table
- only summer school has course location
- only summer school prior 2020 have credits earned (but outliers)
- need to reorganize the columns...

## 2) Check missing data


```python
sns.heatmap(programming.isnull(), cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8a5c99b990>




![png](output_137_1.png)


***
