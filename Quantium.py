#!/usr/bin/env python
# coding: utf-8

# In[320]:


#import necessary libraries
import pandas as pd
import numpy as np


# In[462]:


#loading_data
purchase_data = pd.read_csv("https://insidesherpa.s3.amazonaws.com/vinternships/companyassets/32A6DqtsbF7LbKdcq/QVI_purchase_behaviour.csv")


# In[322]:


# set seed for reproducibility
np.random.seed(0) 
purchase_data.head()


# In[323]:




transaction_data = pd.read_csv(r"C:\Users\swetagupta110\Desktop\ML\QVI_transaction_data.csv")


# In[324]:


np.random.seed(0) 
transaction_data.head()


# In[325]:


P_NUMBER=transaction_data.PROD_NAME.str.split(' ').str[-1]
P_NAME=transaction_data.PROD_NAME.str.split(' ').str[:-2]


# In[326]:


P_NAME.head()
P_NUMBER.head()


# In[327]:



pnew = pd.DataFrame(P_NAME[:-2])
pnew.head()


# In[328]:


transaction_data['P_NAME'] = P_NAME


# In[329]:


transaction_data['P_NUMBER'] = P_NUMBER


# In[330]:


del transaction_data['PROD_NAME']


# In[331]:


transaction_data.head()


# In[332]:


# get the number of missing data points per column
missing_values_count_purchase = purchase_data.isnull().sum()
missing_values_count_purchase[:]


# In[333]:


# get the number of missing data points per column
missing_values_count_transaction = transaction_data.isnull().sum()
missing_values_count_transaction[:]


# In[334]:


# data types as result 
datatype_purchase = purchase_data.dtypes 
print(datatype_purchase)


# In[335]:


datatype_transaction = transaction_data.dtypes 
print(datatype_transaction)


# In[336]:


transaction_data['DATE'] = pd.to_datetime(transaction_data['DATE'])


# In[337]:


transaction_data.head()


# In[338]:


transaction_data['DATE'] = transaction_data['DATE'].apply(str)


# In[339]:


transaction_data['DATE'] = transaction_data.DATE.str.split(' ').str[0]


# In[340]:


transaction_data.head()


# In[341]:


transaction_data['DATE'] = pd.to_datetime(transaction_data['DATE'])


# In[342]:


transaction_data.head()


# In[343]:


transaction_data['P_NAME'] = transaction_data['P_NAME'].apply(str)
transaction_data['P_NAME'] = transaction_data['P_NAME'].str.replace(',', '')


# In[344]:


transaction_data.head()


# In[345]:


transaction_data['P_NAME'] = transaction_data['P_NAME'].str.replace("''", '')


# In[346]:


transaction_data.head()


# In[347]:


transaction_data['P_NAME'] = transaction_data['P_NAME'].str.replace("'", '')
transaction_data.head()


# In[348]:


transaction_data['P_NAME'] = transaction_data['P_NAME'].apply(str)
#transaction_data[P_NAME] =transaction_data[P_NAME].str[:]
transaction_data.head()


# In[349]:


transaction_data['P_NUMBER'] = transaction_data['P_NUMBER'].str.replace("g", '')


# In[350]:


transaction_data.head()


# In[351]:


purchase_data.head()


# In[352]:


merged_outer = pd.merge(left=purchase_data, right=transaction_data, how='outer', left_on='LYLTY_CARD_NBR', right_on='LYLTY_CARD_NBR')
merged_outer


# In[353]:


merged_outer.dtypes


# In[354]:


merged_outer.drop_duplicates()


# In[355]:


merged_outer['P_NUMBER'] = merged_outer['P_NUMBER'].str.extract('(\d+)', expand=False)



# In[356]:


merged_outer.dtypes


# In[357]:


merged_outer["P_NUMBER"] = merged_outer["P_NUMBER"].astype(str).astype(float)
#pd.to_numeric(merged_outer['P_NUMBER'])


# In[358]:


merged_outer.dtypes


# In[359]:


#merged_outer["P_NUMBER"] = merged_outer["P_NUMBER"].astype(float).astype(int)
merged_outer["P_NUMBER"].describe()


# In[387]:


264835-261579
merged_outer['Total_Value'] = merged_outer.PROD_QTY  * merged_outer.P_NUMBER   
merged_outer.head()


# In[361]:


#checking for null values
a=pd.isnull(merged_outer["P_NUMBER"])


# In[362]:


a.describe()


# In[363]:


#Checking for Duplicates
merged_outer.drop_duplicates()


# In[388]:


merged_outer.groupby(by=["LIFESTAGE"]).count()


# In[389]:


25110+6919+48596+54479+49763+43592+36377


# In[390]:


merged_outer.groupby(by=["PREMIUM_CUSTOMER"]).count()


# In[391]:


merged_outer.PREMIUM_CUSTOMER.describe()


# In[392]:


merged_outer.LIFESTAGE.describe()


# In[393]:


merged_outer.P_NAME.describe()


# In[394]:


merged_outer.STORE_NBR.describe()


# In[395]:


merged_outer["STORE_NBR"] = merged_outer["STORE_NBR"].astype(str)


# In[396]:


merged_outer.STORE_NBR.describe()


# In[397]:


#countries_reviewed = merged_outer.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])
#countries_reviewed.head()


# In[458]:


Merged_grouped_data=merged_outer.groupby(by=['LIFESTAGE', 'PREMIUM_CUSTOMER']).sum()


# In[399]:


A=merged_outer.groupby(by=['PREMIUM_CUSTOMER', 'LIFESTAGE']).sum()



# In[400]:


A.head()


# In[459]:


Merged_grouped_data


# In[410]:


np.transpose
B_transposed = B.T
B_transposed.head()


# In[417]:


#merged_outer.pivot("LIFESTAGE", "PREMIUM_CUSTOMER", "PROD_QTY")


# In[424]:


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# In[428]:


# Change the style of the figure to the "dark" theme
#sns.set_style("dark")

# Line chart 
#plt.figure(figsize=(12,6))
#sns.barplot(data=B[ "PROD_QTY"])


# In[429]:


merged_outer.to_csv('file1.csv') 


# In[431]:


final_data = pd.read_csv('file1.csv')


# In[432]:


final_data.head()


# In[439]:


B


# In[451]:


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(100,100))

labels = ['MSC','NF','OF','OSC','R','YF','YSC']
Budget = [9496,5571,45065,35220,28764,37111,16671]
Mainstream = [22699,4319,27756,34997,40518,25044,38632]
Premium=[15526,2957,22171,33986,24884,22406,11331]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x -width/2, Budget, width, label='Budget')
rects2 = ax.bar(x + width/2, Mainstream, width, label='Mainstream')
rects3 = ax.bar(x + 2*width/2, Premium, width, label='Premium')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Quantity')
ax.set_title('Lifestages')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom') 


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)


fig.tight_layout()

plt.show()


# In[460]:


Merged_grouped_data_count=merged_outer.groupby(by=['LIFESTAGE', 'PREMIUM_CUSTOMER']).count()


# In[461]:


Merged_grouped_data_count


# In[457]:


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(100,100))

labels = ['MSC','NF','OF','OSC','R','YF','YSC']
Budget_per_person = [9496/5020,5571/3005,45065/23160,35220/18407,28764/15201,37111/19122,16671/9242]
Mainstream_per_person = [22699/11874,4319/2325,27756/14244,34997/18318,40518/21466,25044/12907,38632/20854]
Premium_per_person=[15526/8216,2957/1589,22171/11192,33986/17754,24884/13096,22406/11563,11331/6281]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x -width/2, Budget, width, label='Budget')
rects2 = ax.bar(x + width/2, Mainstream, width, label='Mainstream')
rects3 = ax.bar(x + 2*width/2, Premium, width, label='Premium')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Quantity')
ax.set_title('Lifestages')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()





fig.tight_layout()

plt.show()


# In[ ]:




