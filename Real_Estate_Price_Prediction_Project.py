#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# #Read File

# In[91]:


df1 = pd.read_csv("C:/Users/Shweta Patil/code2/bengaluru_house_data.csv")
df1.head()


# # Data Cleaning

# In[92]:


df1.shape


# In[93]:


df1.columns


# In[94]:


df1.groupby('area_type')['area_type'].agg('count')


# In[95]:


# drop column -assume availability & etc not imp for deciding final price


# In[96]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# In[97]:


df2.isnull().sum()


# In[98]:


df3=df2.dropna()
df3.isnull().sum()


# In[99]:


df3['size'].unique()


# In[100]:


df3['bhk']=df3['size'].apply(lambda x:int(x.split(' ')[0]))


# In[101]:


df3.head()


# In[102]:


df3['bhk'].unique()


# In[103]:


df3[df3.bhk>20]


# In[104]:


df3.total_sqft.unique()


# In[105]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[106]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[107]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 


# In[108]:


convert_sqft_to_num('2345')


# In[109]:


convert_sqft_to_num('1123-8976')


# In[110]:


convert_sqft_to_num('4125Perch')


# In[111]:


df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[112]:


df4.loc[30]


# In[113]:


(2100+2850)/2


# # Feature Engineering

# In[114]:


df4.head(4)


# In[115]:


df5=df4.copy()
df5['price_per_sqt']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[116]:


df5.location.unique()


# In[117]:


len(df5.location.unique())


# In[118]:


df5.location=df5.location.apply(lambda x:x.strip())

location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)


# In[119]:


location_stats


# In[120]:


len(location_stats[location_stats<=10])


# In[121]:


location_stats_less_than_10=location_stats[location_stats<=10]
location_stats_less_than_10


# In[122]:


df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[123]:


df5.head(20)


# # Outlier Removal

# In[124]:


df5[df5.total_sqft/df5.bhk <300].head()


# In[125]:


df5.shape


# In[126]:


df6=df5[~(df5.total_sqft/df5.bhk <300)]
df6.shape


# In[127]:


df6.price_per_sqt.describe()


# In[128]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqt)
        st = np.std(subdf.price_per_sqt)
        reduced_df = subdf[(subdf.price_per_sqt>(m-st)) & (subdf.price_per_sqt<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[129]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
plot_scatter_chart(df7,"Rajaji Nagar")   


# In[130]:


plot_scatter_chart(df7,"Hebbal")


# In[131]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqt),
                'std': np.std(bhk_df.price_per_sqt),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqt<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[132]:


plot_scatter_chart(df8,"Hebbal")


# In[133]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqt,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[134]:


df8.bath.unique()


# In[135]:


df8[df8.bath>10]


# In[136]:


plt.hist(df8.bath,rwidth=0.8) #rwidth=bar width
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[137]:


df8[df8.bath>df8.bhk+2]


# In[138]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[139]:


df10 = df9.drop(['size','price_per_sqt'],axis='columns')
df10.head(3)


# In[160]:


dummies = pd.get_dummies(df10.location,dtype=int)
dummies.head(3)


# In[161]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[162]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[163]:


df12.shape


# In[164]:


X = df12.drop(['price'],axis='columns')
X.head(3) # x-independent variables


# In[165]:


y=df12.price
y.head()


# In[166]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[167]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[168]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[175]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[176]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[177]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[178]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[179]:


predict_price('Indira Nagar',1000, 2, 2)


# In[180]:


predict_price('Indira Nagar',1000, 3, 3)


# In[189]:


import pickle
file_path = "C:/Users/Shweta Patil/code2/banglore_home_prices_model.pickle"

with open(file_path, 'wb') as f:
    pickle.dump(lr_clf, f)


# In[191]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
file_path = "C:/Users/Shweta Patil/code2/columns.json" 
with open(file_path,"w") as f:
    json.dump(columns, f, indent=4)

