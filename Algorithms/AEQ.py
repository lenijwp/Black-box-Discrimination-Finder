
import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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




from Utils.new_config import census_age, census_gender, census_race, bank_age, credit_age, credit_sex
#Dat_list=[census_age(), census_gender(), census_race(), bank_age(), credit_age(), credit_sex()]
Dat_list=[credit_sex(),credit_age()]
for Dat in Dat_list:
    for Exeiter in range(1):

        target_model_path = Dat.model_path
        protected_attribute=Dat.protected_attri  ##protected_attribute information
        pro_at=Dat.pro_at ##protected_attribute id for calculate
        max_iter=10  ##maximum global iteration number
        ##min and max for each attribute

        input_bound = Dat.input_bounds
        dim=Dat.params
        min_I=[]
        max_I=[]
        for i in range(dim):
            min_I.append(input_bound[i][0])
            max_I.append(input_bound[i][1])


        #min_I = [1,0,0,0,0,0,0,0,0,0,0,0,0]   
        #max_I = [9,7,39,15,6,13,5,4,1,99,39,99,39]

        result_for_vis=[]

        pa_space_size=1

        pro_for_attri=np.zeros(dim)
        pro_for_direc=np.zeros(dim)

        alter_visit=0
        all_visit=[]
        all_dis=[]
        all_visit_label=[]
        step_attri=0
        step_direc=0

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

        target_model = K.models.load_model(target_model_path)
        def ini_pro():
            for i in range(dim):
                if i in protected_attribute:
                    continue
                pro_for_attri[i]=float(1)/float(dim-pa_space_size)
                pro_for_direc[i]=0.5


        def Querytargetmodel(x):
            global target_model
            if x in all_visit:
                i = all_visit.index(x)
                return all_visit_label[i]
            global alter_visit
            alter_visit-=1
            if alter_visit%200==0:
                result_for_vis.append(len(all_dis))
            
            nx = np.array(x,dtype = float)
            nx = nx.reshape(1,dim)
            res = target_model.predict(nx,steps = 1)[0]
            tmp_label = np.argmax(res)
            all_visit.append(x.copy())
            all_visit_label.append(tmp_label)
            #print(tmp_label)
            return tmp_label

        def Clip(x):
            for i in range(len(x)):
                x[i] =max(x[i],min_I[i])
                x[i] =min(x[i],max_I[i])
            return x

        def Judge_discriminatory(x):
            res0 = Querytargetmodel(x)

            for changes in pa_space_choices:
                nx = x.copy()
                for i in range(len(pro_at)):
                    nx[pro_at[i]] = changes[i]
                if nx==x:
                    continue
                res1 = Querytargetmodel(nx)
                
                if res0!=res1:
                    return True,nx.copy()

            return False,None

        def Global_search(G_num,opt):
            global alter_visit
            for i in range(G_num):
                x = []
                for j in range(dim):
                    x.append(random.randint(min_I[j],max_I[j]))
                Judge = Judge_discriminatory(x)
        
                if Judge[0]==True:
                    if (x[0:pro_at[0]]+x[pro_at[0]+1:len(x)]) not in all_dis:
                        all_dis.append((x[0:pro_at[0]]+x[pro_at[0]+1:len(x)]).copy())
                        Local_search([x.copy()],200,1,opt)
                    else:
                        Local_search([x.copy()],50,1,opt)
                    if alter_visit<=0:
                        return
        def Local_search(Seeds,L_num,steplen,opt):
            global alter_visit,pro_for_attri,pro_for_direc,step_attri,step_direc
            for x in Seeds:
                for iter in range(L_num):
                    if alter_visit<=0:
                        return
                    random_p_attri = random.random()
                    attri = 0
                    tmp=0.0
                    for j in range(dim):
                        if j in protected_attribute:
                            continue
                        if tmp<random_p_attri and tmp+pro_for_attri[j]>=random_p_attri:
                            attri = j
                            break
                    random_p_direc = random.random()
                    direc=1
                    if random.random()<pro_for_direc[attri]:
                        direc=-1
                    nx = x.copy()
                    nx[attri]+=direc*steplen
                    nx = Clip(nx)
                    Judge = Judge_discriminatory(nx)
                    flag=False
                    if Judge[0]==True:
                        flag=True
                        if (nx[0:pro_at[0]]+nx[pro_at[0]+1:len(nx)]) not in all_dis:
                            all_dis.append((nx[0:pro_at[0]]+nx[pro_at[0]+1:len(nx)]).copy())
                        x = nx
                    
                    Updata(step_attri,step_direc,direc,attri,flag,opt)

        def Updata(step_attri,step_direc,direc,attri,flag,opt):
            global pro_for_attri,pro_for_direc
            if opt==0:
                return
            if opt==1:
                if (flag==True and direc==-1) or (flag==False and direc==1):
                    pro_for_direc[attri]=min(1,pro_for_direc[attri]+step_direc)
                if (flag==True and direc==1) or (flag==False and direc==-1):
                    pro_for_direc[attri]=max(0,pro_for_direc[attri]-step_direc) 
                return
            if opt==2:
                if (flag==True and direc==-1) or (flag==False and direc==1):
                    pro_for_direc[attri]=min(1,pro_for_direc[attri]+step_direc)
                if (flag==True and direc==1) or (flag==False and direc==-1):
                    pro_for_direc[attri]=max(0,pro_for_direc[attri]-step_direc)
                if flag==True:
                    pro_for_attri[attri]+=step_attri
                    tmp =sum(pro_for_attri)
                    pro_for_attri/=tmp
                return

        start_time=time.time()

        alter_visit=50000
        all_visit=[]
        all_dis=[]
        all_visit_label=[]
        ini_pro()
        step_attri=0.001
        step_direc=0.001
        num=0
        while True:
            num+=1
            Global_search(10,2)
            if alter_visit<=0:
                break
            if num%1000==0:
                print(num,"   ",len(all_dis),"   alter_visit = ",alter_visit)

        end_time=time.time()

        outfile_path2='Demo/Results/Numbers/AEQ/'+Dat.__class__.__name__+'_'+str(Exeiter)+'.txt'
        print(outfile_path2)
    # f = open('Demo/Results/ADF_result.txt','w+')
        
        with open(outfile_path2,'w+') as f2:
            print(str(result_for_vis))
            f2.write(str(result_for_vis))
            f2.write('\n')
            print(str(end_time-start_time))
            f2.write(str(end_time-start_time))
            
        #print(result_for_vis)



        outfile_path='Demo/Results/AEQ/'+Dat.__class__.__name__+'_'+str(Exeiter)+'.txt'
        print(outfile_path)
    # f = open('Demo/Results/ADF_result.txt','w+')
        f = open(outfile_path,'w+')

        #f = open('Demo/Results/AEQ_result.txt','w+')
        for i in range(len(all_dis)):
            st=''
            for j in range(len(all_dis[i])):
                st=st+str(all_dis[i][j])+' '
            st=st+'\n'
            f.write(st)
        f.close

        

