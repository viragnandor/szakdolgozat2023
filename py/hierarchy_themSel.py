#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('themSel.tsv',sep='\t')


# In[ ]:


featureset = pd.DataFrame()
featureset["I"] = df["I"]
featureset["N"] = df["N"]


# In[ ]:


ndarray = featureset.to_numpy()
x = featureset.values


# In[ ]:


import scipy
leng = ndarray.shape[0]
D = np.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(ndarray[i], ndarray[j])


# In[ ]:


import scipy
leng = ndarray.shape[0]
D = np.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(x[i], x[j])


# In[ ]:


import pylab
import scipy.cluster.hierarchy as hierarchy
Z = hierarchy.linkage(D, 'average', optimal_ordering=True)


# In[ ]:


from scipy.cluster.hierarchy import fcluster
max_d = 380
clusters = fcluster(Z, max_d, criterion='distance')


# In[ ]:


fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s]' % (df['v'][id])
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right', color_threshold=380.0)


# In[ ]:


classed = clusters.tolist()


# In[ ]:


df_mod = pd.DataFrame()
df_mod["v"] = df["v"]
df_mod["I"] = df["I"]
df_mod["N"] = df["N"]
df_mod["KAT"] = classed


# In[ ]:




