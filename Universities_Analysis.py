import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df =
pd.read_csv("E:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\groceries_200.csv",header=-
1)
df.head()

df.isnull().sum()

df.fillna(0,inplace=True)
df.head()

df.isnull().sum()

arr = np.array(df.values)
all_items = np.unique(arr[~(arr==0)])
print(all_items)

basket = pd.DataFrame(0,index=np.arange(len(df)),columns=all_items)
basket.head()

for col in df.columns: 
t_f = (df[col].values!=0)
itms = df.iloc[t_f,col]

indx = itms.index 
for i in range(len(itms)):
basket.loc[indx[i]][itms.iloc[i]] += 1

basket.head()

frequent_itemsets = apriori(basket,min_support=0.02,use_colnames=True)
frequent_itemsets.head()

frequent_itemsets.tail()


frequent_itemsets.head()

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets.tail()

rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1)

rules.head()

len(rules)

rules = association_rules(frequent_itemsets,metric="lift", min_threshold=5)
len(rules)

rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1)
rules[rules['confidence']>=0.7]

rules = association_rules(frequent_itemsets,metric="confidence", min_threshold=0.5)
rules.head()

rules = association_rules(frequent_itemsets, min_threshold=0.5)
rules.head()

rules = association_rules(frequent_itemsets, min_threshold=2.0)
rules.head()

rules = association_rules(frequent_itemsets,metric="support", min_threshold=0.01)
rules.head()

import pandas as pd
dict = {'itemsets': [['177', '176'], ['177', '179'],
['176', '178'], ['176', '179'],
['93', '100'], ['177', '178'],
['177', '176', '178']],
'support':[0.253623, 0.253623, 0.217391,
0.217391, 0.181159, 0.108696, 0.108696]}

freq_itemsets = pd.DataFrame(dict)
freq_itemsets