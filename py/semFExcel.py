#!/usr/bin/env python
# coding: utf-8

# In[ ]:


xmlDir = 'xml3/'
cpDir = 'cplist/'
outputDir = 'testing/'
fileNames = ['']
targetVerb = ['']
sfFile = 'semFeatures.tsv'


# In[ ]:


import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


# In[ ]:


v = ['id', 'pvform', 'lemma', 'pvv', 'mood', 'cau', 'pot']
f = ['PV', 'CP_cnd', 'CP_imp', 'CP_ind', 'HKM', 'inf', 'nom', 'acc', 'dat', 
     'BAN', 'ON', 'RA', 'VAL', 'UL', 'BA', 'RÓL', 'HOZ', 'BÓL', 'TÓL', 
     'NÁL', 'VÁ', 'IG', 'ÉRT', 'KÉNT', 'KOR', 'SZOR', 'NKÉNT', 'ADP', 'ADV']
loc = ['FROM', 'IN', 'TO']


# In[ ]:


semFeatures = pd.read_csv(sfFile, sep='\t')
sfDict = semFeatures.set_index('lemma')[['H', 'I', 'N', 'A', 'L']].apply(tuple, axis=1).to_dict()


# In[ ]:


for i in range(len(fileNames)):
    txtout = ''
    fileName = fileNames[i]
    vLemma = targetVerb[i]
    print(fileName)
    root = ET.parse(cpDir + fileName + '.xml').getroot()
    sroot = ET.parse(xmlDir + fileName + '.xml').getroot()
    cps = root.findall('cp')
    for cp in cps:
        cpHead = cp.findall('head')
        cpId = cpHead[0].text.split('w')[0]
        headLemma = cpHead[0].attrib['lemma']
        headMood = cpHead[0].attrib['mood']
        headPV = cpHead[0].attrib['pv']
        if headLemma == vLemma and headMood == 'ind' and headPV == '0':
            xpList = cp.findall('xp/head[@f="nom"]')
            for xp in xpList:
                if xp.attrib['lemma'] in sfDict.keys():
                    stext = ''
                    xpLemma = xp.attrib['lemma']
                    wf = sfDict[xpLemma]
                    s = sroot.findall('s[@id="' + cpId + '"]')[0]
                    ws = s.findall('w/form')
                    for w in ws:
                        stext = stext + w.text + ' '
                    txtout = txtout + cpId + '\t' + headLemma + '\t' + xpLemma + '\t'
                    for i in range(5):
                        f = wf[i]
                        txtout = txtout + str(f) + '\t'
                    txtout = txtout + stext + '\n'
    f = open(outputDir+fileName+'.tsv', "w")
    f.write(txtout)
    f.close()


# In[ ]:




