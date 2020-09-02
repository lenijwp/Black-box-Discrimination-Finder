import sys
import os
#sys.path.append("C://Users//shen//Desktop//jwp//BLFT")
print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import keras as K
import tensorflow as tf
import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

from Utils.input_config import Dat


target_model_path = Dat.model_path
protected_attribute=Dat.protected_attri ##protected_attribute information
pro_at=Dat.pro_at
max_iter=10

input_bound = Dat.input_bounds
dim=Dat.params
min_I=[]
max_I=[]
for i in range(dim):
    min_I.append(input_bound[i][0])
    max_I.append(input_bound[i][1])

all_visit=[]  #all samples know in the discrimanation samples searchering
all_visit_label=[]
all_grads=[]
all_pro=[]
all_dis=[]   #all discrimination pairs (0 1) (2,3) ……
all_search=[]  #all samples known
alter_visit = 0 # alternative access numbers to target model
train_num=0 #access numbers cost during sub_model_train
T_1 = 0
pa_space_size = 1  
all_dis_without=[]
result_for_vis=[]

for i in pro_at:
    pa_space_size *= max_I[i]-min_I[i]+1;
pa_space_choices = [[] for i in range(pa_space_size)]  ##protected attribute space
for i in pro_at:
    cnt =int(pa_space_size/(max_I[i]-min_I[i]+1))
    pos = 0
    for j in range(min_I[i],max_I[i]+1):
        for k in range(cnt):
            pa_space_choices[pos].append(j)
            pos = pos+1
gradzero=0
all_success=0
global_success=0
global_success_first=0
alter_visit=0
target_model = K.models.load_model(target_model_path)
sess = K.backend.get_session()

G_target0 = K.backend.one_hot(0, 2)
G_target1 = K.backend.one_hot(1, 2)
G_loss0 = K.backend.categorical_crossentropy(G_target0, target_model.output)
G_loss1 = K.backend.categorical_crossentropy(G_target1, target_model.output)
G_grad0 = K.backend.gradients(G_loss0, target_model.input)
G_grad1 = K.backend.gradients(G_loss1, target_model.input)

def Querytargetmodel(x):
    
    if x in all_visit:
        i = all_visit.index(x)
        return all_visit_label[i],all_grads[i],all_pro[i]
    global alter_visit
    alter_visit-=1

    if alter_visit%200==0:
        result_for_vis.append(len(all_dis_without))


    nx = np.array(x,dtype = float)
    nx=nx.reshape(1,dim)
    res = target_model.predict(nx,steps = 1)[0]
    tmp_label = np.argmax(res)
    tmp_grad=0
   # global sess,target_model,G_grad0,G_grad1
    if tmp_label == 0:
        tg = sess.run(G_grad0[0],feed_dict={target_model.input:nx})
        tmp_grad = tg[0]
    if tmp_label == 1:
        tg = sess.run(G_grad1[0],feed_dict={target_model.input:nx})
        tmp_grad = tg[0]

    all_visit.append(x.copy())
    all_grads.append(tmp_grad.tolist())
    all_visit_label.append(tmp_label)
    all_pro.append(res.tolist())
    return tmp_label,tmp_grad.tolist(),res.tolist()
def Clip(x):
    for i in range(len(x)):
        x[i] =max(x[i],min_I[i])
        x[i] =min(x[i],max_I[i])
    return x
def Choose_argmax(x):
    res0 = Querytargetmodel(x)[2]
    y = x.copy()
    ans = 0.0
    for changes in pa_space_choices:
        nx = x.copy()
        for i in range(len(pro_at)):
            nx[pro_at[i]] = changes[i]
        if (nx==x):
            continue
        res1 = Querytargetmodel(nx)[2]
        tmp = 0.0
        for i in range(len(res1)):
            tmp = tmp + abs(res0[i]-res1[i])
        if tmp>ans:
            ans = tmp
            y = nx.copy()
    return y.copy() 

def Judge_discriminatory(x):
    res0 = Querytargetmodel(x)[0]

    for changes in pa_space_choices:
        nx = x.copy()
        for i in range(len(pro_at)):
            nx[pro_at[i]] = changes[i]
        if nx==x:
            continue
        res1 = Querytargetmodel(nx)[0]
        if res0!=res1:
            return True,nx.copy()

    return False,None

def Find_discriminatory(x):
    xpos = all_dis.index(x)
    if xpos %2==0:
        return all_dis[xpos+1]
    else:
        return all_dis[xpos-1]

def Normalization(Gx,Gy):
    Gs=np.zeros(dim)
    Record=[]
    for i in range(dim):
        if i in protected_attribute:
            Gs[i]=0
            continue
        saliency = abs(Gx[i])+abs(Gy[i])
        if saliency==0.0:
            Record.append(i)
            continue
        Gs[i]=1.0/saliency
    MaxGs=max(Gs)
    if len(Record)!=0:
        for i in range(len(Record)):
            Gs[i]=2.0*MaxGs
    Sum=0.0
    for i in range(dim):
        Sum+=Gs[i]
    if Sum==0:
        for i in range(dim):
            Gs[i] = 1.0/(dim-len(protected_attribute))
    else:
        for i in range(dim):
            Gs[i]/=Sum
    return Gs.tolist()


def Global_Generation(Seeds,C_num,G_snum,Max_iters,steplen):
    global gradzero,global_success,global_success_first,alter_visit,all_success
    clusters = KMeans(n_clusters=C_num).fit(Seeds)
    labels = clusters.labels_
    cforl_lists = [[] for i in range(C_num)]
    lens = len(Seeds)
    seed_flag = 0
    for i in range(lens):
        cforl_lists[labels[i]].append(i)

    for i in range(G_snum):
        
        pos = random.choice(cforl_lists[seed_flag])
        seed_flag+=1
        if seed_flag==C_num:
            seed_flag=0
        x = Seeds[pos].copy()
        if alter_visit<=0:
            return
        for iters in range(Max_iters):
            if alter_visit<=0:
                return
            if x not in all_dis:
                Judge = Judge_discriminatory(x)
                if Judge[0]==True:
                    all_dis.append(x.copy())
                    all_dis.append(Judge[1].copy())
                    if (x[0:pro_at[0]]+x[pro_at[0]+1:len(x)]) not in all_dis_without: 
                        all_dis_without.append((x[0:pro_at[0]]+x[pro_at[0]+1:len(x)]).copy())
                
                    global_success+=1
                    if iters ==0:
                        global_success_first+=1
                    Local_Generation([x.copy()],200,1)
                    break
            else:
                all_success+=1
                Local_Generation([x.copy()],50,1)

            y = Choose_argmax(x)
            Gx= Querytargetmodel(x)[1]
            Gy= Querytargetmodel(y)[1]
            sGx=np.sign(Gx)
            sGy=np.sign(Gy)
            if (sGx==0).all() and (sGy==0).all():
                gradzero+=1
                break
            if (sGx==0).all():
                for k in range(dim):
                    if k in protected_attribute:
                        continue
                    x[k]=x[k]+sGy[k]
                x = Clip(x)
                continue
            if (sGy==0).all():
                for k in range(dim):
                    if k in protected_attribute:
                        continue
                    x[k]=x[k]+steplen*sGx[k]
                x = Clip(x)
                continue
            for k in range(dim):
                if k in protected_attribute:
                    continue
                if sGx[k]==sGy[k]:
                    x[k]=x[k]+steplen*sGx[k]
            x = Clip(x)

    return 

def Local_Generation(Seeds,L_snum,steplen):

    for i in range(len(Seeds)):
        x = Seeds[i].copy()
        
        for lnum in range(L_snum):
            
            y = Find_discriminatory(x)
            Gx= Querytargetmodel(x)[1]
            Gy= Querytargetmodel(y)[1]
            Probability=Normalization(Gx.copy(),Gy.copy())
            rom = random.random()
            temp=0.0
            Attr=0
            for j in range(dim):
                if Probability[j]==0:
                    continue
                if temp<rom and temp+Probability[j]>=rom:
                    Attr=j
                    break
                temp+=Probability[j]
            rom=random.random()
            direc=1
            if rom<0.5:
                direc=-1
            nx = x.copy()
            nx[Attr]+=direc*steplen
            nx = Clip(nx)
            if nx in all_dis:
                x=nx.copy()
                continue
            Judge = Judge_discriminatory(nx)
            if Judge[0]==True:
                all_dis.append(nx.copy())
                all_dis.append(Judge[1].copy())
                if (nx[0:pro_at[0]]+nx[pro_at[0]+1:len(nx)]) not in all_dis_without: 
                    all_dis_without.append((nx[0:pro_at[0]]+nx[pro_at[0]+1:len(nx)]).copy())
                    
                x = nx.copy()
            if alter_visit<=0:
                return

    return

def load_data(file_path):

    file = open(file_path)
    data=[]
    label=[]
    count=0
    while True:
        temp=file.readline()
        if not temp:
            break
        temp = temp.replace(',',' ')
        temp = temp.split()
        count+=1
        if count==1:
            continue
    #print(temp)
        for i in range(len(temp)):
            temp[i]=float(temp[i])
        data.append(temp[0:len(temp)-1])
   
    return data


# In[35]:

T_1=time.time()
alter_visit=3000
all_visit=[]
all_visit_label=[]
all_grads=[]
all_dis=[]
all_pro=[]
gradzero=0
all_success=0
global_success=0
global_success_first=0

data = load_data(Dat.data_path)

while alter_visit>0:
    Global_Generation(data,4,200,10,1)

f = open('Demo/Results/ADF_result.txt','w+')
for i in range(len(all_dis_without)):
    st=''
    for j in range(len(all_dis_without[i])):
        st=st+str(int(all_dis_without[i][j]))+' '
    st=st+'\n'
    f.write(st)
f.close



