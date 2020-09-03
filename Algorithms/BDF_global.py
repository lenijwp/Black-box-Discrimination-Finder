import sys
import os
#sys.path.append("C://Users//shen//Desktop//jwp//BLFT")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))



import keras as K
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import time
import tensorflow as tf
from Utils.input_config import Dat


target_model_path = Dat.model_path
sub_model_path = 'Demo/Shadowmodels/1_global_model_local.h5'
protected_attribute = Dat.protected_attri ##protected_attribute information
pro_at = Dat.pro_at  ##protected_attribute id for calculate##protected_attribute id for calculate
max_iter = 10  

##min and max for each attribute·
input_bound = Dat.input_bounds
dim=Dat.params
min_I=[]
max_I=[]
for i in range(dim):
    min_I.append(input_bound[i][0])
    max_I.append(input_bound[i][1])
all_visit=[]  #all samples know in the discrimanation sample
# s searchering
label_all_visit=[]
all_dis=[]   #all discrimination pairs (0 1) (2,3) ……
all_dis_without=[]
all_search=[]  #all samples known
alter_visit = 0 # alternative access numbers to target model
train_num=0 #access numbers cost during sub_model_train

false_jus=0
false_jus_true=0

pa_space_size = 1  


for i in pro_at:
    pa_space_size *= max_I[i]-min_I[i]+1
pa_space_choices = [[] for i in range(pa_space_size)]  ##protected attribute space
for i in pro_at:
    cnt =int(pa_space_size/(max_I[i]-min_I[i]+1))
    pos = 0
    for j in range(min_I[i],max_I[i]+1):
        for k in range(cnt):
            pa_space_choices[pos].append(j)
            pos = pos+1
gradzero=0
global_success=0
global_success_first=0
old_success=0

#initialize the session vars
sub_model=K.models.load_model(target_model_path)# !only for initialization,before using it we will reflash it by shadow models.
#the 'target_model_path' can be replaced to any model.
target_model=K.models.load_model(target_model_path)
sess = K.backend.get_session()

G_target0 = K.backend.one_hot(0, 2)
G_target1 = K.backend.one_hot(1, 2)
G_loss0 = K.backend.categorical_crossentropy(G_target0, sub_model.output)
G_loss1 = K.backend.categorical_crossentropy(G_target1, sub_model.output)
G_grad0 = K.backend.gradients(G_loss0, sub_model.input)
G_grad1 = K.backend.gradients(G_loss1, sub_model.input) 

J_target0 = sub_model.output[0][0]
J_target1 = sub_model.output[0][1]
J_jacobi0 = K.backend.gradients(J_target0, sub_model.input)
J_jacobi1 = K.backend.gradients(J_target1, sub_model.input)


def Reflash_session(Target_model_path,Sub_model_path):
    global sub_model,target_model,sess
    global G_target0,G_target1,G_loss0,G_loss1,G_grad0,G_grad1,J_jacobi0,J_jacobi1,J_target0,J_target1
    #with K.backend.session():
#    K.backend.clear_session()
    sub_model=K.models.load_model(Sub_model_path)
    target_model=K.models.load_model(Target_model_path)
    sess = K.backend.get_session()
    
    G_target0 = K.backend.one_hot(0, 2)
    G_target1 = K.backend.one_hot(1, 2)
    G_loss0 = K.backend.categorical_crossentropy(G_target0, sub_model.output)
    G_loss1 = K.backend.categorical_crossentropy(G_target1, sub_model.output)
    G_grad0 = K.backend.gradients(G_loss0, sub_model.input)
    G_grad1 = K.backend.gradients(G_loss1, sub_model.input) 

    J_target0 = sub_model.output[0][0]
    J_target1 = sub_model.output[0][1]
    J_jacobi0 = K.backend.gradients(J_target0, sub_model.input)
    J_jacobi1 = K.backend.gradients(J_target1, sub_model.input)



def Jacobi(x):
    global sess,sub_model,target_model,J_jacobi0,J_jacobi1

    flag = Querytargetmodel(target_model,x)[0] 
    nx = np.array(x,dtype = float)
    nx = nx.reshape(1,dim)
    if flag==0:
        res =sess.run(J_jacobi0[0],feed_dict={sub_model.input:nx})
        return res[0]
    if flag==1:
        res =sess.run(J_jacobi1[0],feed_dict={sub_model.input:nx})
        return res[0]



def Grad(x):
    global sess,sub_model,G_grad0,G_grad1
    flag = Querytargetmodel(sub_model,x)[0]   #before get the Grad, the label has been know,so the result is in all_visit and label_all_visit, so we can directly use the Query targetmodel
    nx = np.array(x,dtype = float)
    nx = nx.reshape(1,dim)
    if flag==0:
        res =sess.run(G_grad0[0],feed_dict={sub_model.input:nx})
        return res[0]
    if flag==1:
        res =sess.run(G_grad1[0],feed_dict={sub_model.input:nx})
        return res[0]

result_for_vis=[]

def Querytargetmodel(model,x): # x is a list
    global alter_visit
    for i in range(len(all_visit)):
        #print(x)
        #print(all_visit[i])
        if x==all_visit[i]:
            return label_all_visit[i],0 #nocost
    nx = np.array(x, dtype = float)
    res0 = model.predict(nx.reshape(1,dim),steps=1)[0]
    all_visit.append(x.copy())
    label_all_visit.append(np.argmax(res0))
    alter_visit-=1
    if alter_visit%200==0:
        result_for_vis.append(len(all_dis_without))
    
    return np.argmax(res0),1 #have cost
def Clip(x):
    for i in range(len(x)):
        x[i] =max(x[i],min_I[i])
        x[i] =min(x[i],max_I[i])
    return x
def Substitute_DNN_Train(max_iters,Seeds,steplen,Target_model_path,Sub_model_path,beta,train_step):
    datas=Seeds
    labels=[]
    global train_num
    global alter_visit
    for itrs in range(max_iters+1):
        datas,labels=Label_data(datas,Target_model_path)
        sub_model = Train_model(datas,labels,itrs,train_step)
        sub_model.save(Sub_model_path)
        if itrs==max_iters:
            return 
        #K.backend.clear_session()
        Reflash_session(Target_model_path,Sub_model_path)
        lens = len(labels)
        for i in range(lens):
            if random.random()<beta:
                continue
            tmp = np.array(datas[i],dtype=float)+steplen*np.sign(Jacobi(datas[i]))
            tmp=Clip(tmp)
            tmp=tmp.tolist()
            if tmp not in datas:
                datas.append(tmp)
        K.backend.clear_session()
        
def Label_data(Data_set,target_model_path):
    target_model = K.models.load_model(target_model_path)
    lens = len(Data_set)
    Label_set=np.zeros(lens)
    global train_num
    for i in range(lens):
        tmp_res=Querytargetmodel(target_model,Data_set[i])
        Label_set[i]=tmp_res[0]
        train_num+=tmp_res[1]
    #K.backend.clear_session()
    return Data_set,Label_set.tolist()
def Define_model():
    model = Sequential()
    model.add(Dense(64,input_dim=dim,activation='relu'))
    model.add(Dense(32,activation='relu'))
    
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(4,activation='relu'))
    #model.add(Dense(64,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def Train_model(datas,labels,steps,train_steps):
    X=np.array(datas,dtype=float)
    Y=np.array(labels,dtype=int)
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)

    from keras.utils import to_categorical
    dummy_y=to_categorical(Y)
    
    model_tmp = Define_model()
    model_tmp.fit(X,dummy_y,batch_size=200,epochs=steps*50+int(train_steps),shuffle=True,verbose=0)
    return model_tmp
def Choose_argmax(x,model):  #sub_model
    x = np.array(x, dtype = float)
    res0 = model.predict(x.reshape(1,dim),steps=1)[0]
    ans = 0.0
    y = np.zeros(dim)
    for changes in pa_space_choices:
        nx=x.copy()
        for i in range(len(pro_at)):
            nx[pro_at[i]]=changes[i]
        if (nx==x).all():
            continue
        res1 = model.predict(nx.reshape(1,dim),steps=1)[0] #sub_model
        tmp = 0.0
        for i in range(len(res0)):
            tmp=tmp+abs(res0[i]-res1[i])
        if tmp>=ans:
            ans = tmp
            y=nx.copy()
    return y.tolist()
def Judge_discriminatory(x,model): #target_model
    tmpres=Querytargetmodel(model,x.copy())
    res0=tmpres[0]
 
    for changes in pa_space_choices:
        nx=x.copy()
        for i in range(len(pro_at)):
            nx[pro_at[i]]=changes[i]
        if nx==x:
            continue
        tmpres=Querytargetmodel(model,nx)
        res1 = tmpres[0]
        if res0!=res1:
            return True,[x.copy(),nx.copy()]
    return False,[x.copy(),nx.copy()]

def Querys(x,model):
    dim = len(x)
    nx = np.array(x,dtype = float)
    nx=nx.reshape(1,dim)
    res = model.predict(nx,steps = 1)[0]
    tmp_label = np.argmax(res)
    return tmp_label

def Query_dis(x,model):
    res0 = Querys(x,model)

    for changes in pa_space_choices:
        nx = x.copy()
        for i in range(len(pro_at)):
            nx[pro_at[i]] = changes[i]
        if nx==x:
            continue
        res1 = Querys(nx,model)
        if res0!=res1:
            return True

    return False

def Find_discriminatory(x): #find the another example for any example pairs in the all_dis list
    xpos = all_dis.index(x)
    if xpos %2==0:
        return all_dis[xpos+1]
    else:
        return all_dis[xpos-1]
def Access_filter(x,model,opt):
    
    return True


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

def Global_Generation(Target_model_path,Sub_model_path,Seeds,C_num,G_snum,Max_iters,steplen,optflag):
    #Seeds clustering
    global gradzero,global_success,global_success_first,alter_visit,old_success
    clusters = KMeans(n_clusters=C_num).fit(Seeds)
    labels = clusters.labels_
    cforl_lists = [[] for i in range(C_num)]
    lens = len(Seeds)
    seed_flag = 0
    for i in range(lens):
        cforl_lists[labels[i]].append(i)
    Reflash_session(Target_model_path,Sub_model_path) 
    for i in range(G_snum):
        if alter_visit<=0:
            return
        
        pos = random.choice(cforl_lists[seed_flag])
        seed_flag+=1
        if seed_flag==C_num:
            seed_flag=0
        x = Seeds[pos].copy()
        for iters in range(Max_iters):
            if x in all_dis:
                old_success+=1
                Local_Generation(Target_model_path,Sub_model_path,[x].copy(),50,50,1,optflag)
                break
            if Access_filter(x,sub_model,1)==True:
                flag =  Judge_discriminatory(x,target_model)
                if alter_visit<0:
                    return
                if flag[0]==True:
                    global_success +=1
                    if iters ==0:
                        global_success_first+=1
                    all_dis.append(flag[1][0].copy())
                    all_dis.append(flag[1][1].copy())
                    if (x[0:pro_at[0]]+x[pro_at[0]+1:len(x)]) not in all_dis_without: 
                        all_dis_without.append((x[0:pro_at[0]]+x[pro_at[0]+1:len(x)]).copy())
                      
                    Local_Generation(Target_model_path,Sub_model_path,[x].copy(),200,200,1,optflag)
                     
                    break
                if alter_visit<=0:
                    return
            y=Choose_argmax(x.copy(),sub_model)
            if alter_visit<0:
                return
            Gx=Grad(x.copy())
            Gy=Grad(y.copy())
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

def Local_Generation(Target_model_path,Sub_model_path,Seeds,L_snum,Failure_num,steplen,optflag):

    global alter_visit,false_jus,false_jus_true
    if alter_visit<0:
        return
    for i in range(len(Seeds)):
        x = Seeds[i].copy()

        for lnum in range(L_snum):
            if alter_visit<0:
                return
            y=Find_discriminatory(x.copy())
            Gx = Grad(x)
            Gy = Grad(y)
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
            if Access_filter(nx,sub_model,optflag)==True:
                flag =  Judge_discriminatory(nx,target_model)
                if alter_visit<0:
                    return
                if flag[0]==True:
                    all_dis.append(flag[1][0].copy())
                    all_dis.append(flag[1][1].copy())
                    if (nx[0:pro_at[0]]+nx[pro_at[0]+1:len(nx)]) not in all_dis_without:

                        all_dis_without.append((nx[0:pro_at[0]]+nx[pro_at[0]+1:len(nx)]).copy())
                        
                    x = nx.copy()
                    continue
                else:
                    Failure_num-=1
                    if Failure_num==-1:
                        return
           
                



def Get_ini_seeds(k):
    Seeds=[]
    for i in range(k):
        seed=[]
        for j in range(dim):
            seed.append(random.randint(min_I[j],max_I[j]))
        Seeds.append(seed)
    return Seeds


T_1=time.time()

alter_visit=5000
all_visit=[]
label_all_visit=[]
all_dis=[]
gradzero=0
old_success=0
global_success=0
global_success_first=0

false_jus=0
false_jus_true=0



inidata=Get_ini_seeds(3000)
Substitute_DNN_Train(0,inidata,1,target_model_path,sub_model_path,0.2,300)
Time_begin = time.time()


i=0
opt_flag=1
filter_tod=0.1
while True:
    base_line = len(all_visit)
    while True:
        Global_Generation(target_model_path,sub_model_path,Get_ini_seeds(1000),4,100,10,1,opt_flag)
        if alter_visit<=0:
            break
        if len(all_visit)-base_line>3000:
            break
    Time_now = time.time()
    if alter_visit<=0:
        break
    X = np.array(all_visit,dtype=float)

    Substitute_DNN_Train(0,all_visit.copy(),1,target_model_path,sub_model_path,0.2,250+int(i)*30)
    K.backend.clear_session()
    sub_model=K.models.load_model(sub_model_path) 
    target_model=K.models.load_model(target_model_path)
    i+=1





f = open('Demo/Results/BDF_global_result.txt','w+')
for i in range(len(all_dis_without)):
    st=''
    for j in range(len(all_dis_without[i])):
        st=st+str(int(all_dis_without[i][j]))+' '
    st=st+'\n'
    f.write(st)
f.close




