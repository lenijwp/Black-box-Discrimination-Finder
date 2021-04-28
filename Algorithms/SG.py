
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
from adf_baseline.lime import lime_tabular
from sklearn.tree import DecisionTreeClassifier


from Utils.new_config import census_age, census_gender, census_race, bank_age, credit_age, credit_sex
Dat_list=[census_age(), census_gender(), census_race(), bank_age(), credit_age(), credit_sex()]
#Dat_list=[census_age()]
for Dat in Dat_list:
    for Exeiter in range(5):

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
       
        #from z3 import *
        import z3
        from queue import PriorityQueue
        import copy

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

        LIME_samples=20
        def Get_path(X,input):
            # use the original implementation of LIME
            explainer = lime_tabular.LimeTabularExplainer(X,
                                                  feature_names=Dat.feature_name, class_names=Dat.class_name, categorical_features=Dat.categorical_features,
                                                  discretize_continuous=True)
            g_data = explainer.generate_instance(input, num_samples=LIME_samples)
            g_labels = []
            if alter_visit<=len(g_data):
                return None
            for g in g_data:
                g_labels.append(Querytargetmodel(g.tolist()))
                
            if alter_visit<=0:
                    return None
            # build the interpretable tree
            tree = DecisionTreeClassifier(random_state=2021) #min_samples_split=0.05, min_samples_leaf =0.01
            tree.fit(g_data, g_labels)

            # get the path for decision
            path_index = tree.decision_path(np.array([input])).indices
            path = []
            for i in range(len(path_index)):
                node = path_index[i]
                i = i + 1
                f = tree.tree_.feature[node]
                if f != -2:
                    left_count = tree.tree_.n_node_samples[tree.tree_.children_left[node]]
                    right_count = tree.tree_.n_node_samples[tree.tree_.children_right[node]]
                    left_confidence = 1.0 * left_count / (left_count + right_count)
                    right_confidence = 1.0 - left_confidence
                    if tree.tree_.children_left[node] == path_index[i]:
                        path.append([f, "<=", tree.tree_.threshold[node], left_confidence])
                    else:
                        path.append([f, ">", tree.tree_.threshold[node], right_confidence])
            return path

        def Global_solve(path_constraint, arguments, t):
            """
            Solve the constraint for global generation
            :param path_constraint: the constraint of path
            :param arguments: the name of features in path_constraint
            :param t: test case
            :param conf: the configuration of dataset
            :return: new instance through global generation
            """
            s = z3.Solver()
            for c in path_constraint:
                s.add(arguments[c[0]] >= min_I[c[0]])
                s.add(arguments[c[0]] <= max_I[c[0]])
                if c[1] == "<=":
                    s.add(arguments[c[0]] <= c[2])
                else:
                    s.add(arguments[c[0]] > c[2])

            if s.check() == z3.sat:
                m = s.model()
            else:
                return None

            tnew = copy.deepcopy(t)
            for i in range(len(arguments)):
                if m[arguments[i]] == None:
                    continue
                else:
                    tnew[i] = int(m[arguments[i]].as_long())
            return tnew.astype('int').tolist()

        def Local_solve(path_constraint, arguments, t, index):
            """
            Solve the constraint for local generation
            :param path_constraint: the constraint of path
            :param arguments: the name of features in path_constraint
            :param t: test case
            :param index: the index of constraint for local generation
            :param conf: the configuration of dataset
            :return: new instance through global generation
            """
            c = path_constraint[index]
            s = z3.Solver()
            s.add(arguments[c[0]] >= min_I[c[0]])
            s.add(arguments[c[0]] <= max_I[c[0]])
            for i in range(len(path_constraint)):
                if path_constraint[i][0] == c[0]:
                    if path_constraint[i][1] == "<=":
                        s.add(arguments[path_constraint[i][0]] <= path_constraint[i][2])
                    else:
                        s.add(arguments[path_constraint[i][0]] > path_constraint[i][2])

            if s.check() == z3.sat:
                m = s.model()
            else:
                return None

            tnew = copy.deepcopy(t)
            tnew[c[0]] = int(m[arguments[c[0]]].as_long())
            return tnew.astype('int').tolist()

        def Average_confidence(path_constraint):
            """
            The average confidence (probability) of path
            :param path_constraint: the constraint of path
            :return: the average confidence
            """
            r = np.mean(np.array(path_constraint)[:,3].astype(float))
            return r

        def Gen_arguments(conf):
            """
            Generate the argument for all the features
            :param conf: the configuration of dataset
            :return: a sequence of arguments
            """
            arguments = []
            for i in range(conf.params):
                arguments.append(z3.Int(conf.feature_name[i]))
                #arguments.append(conf.categorical_features[i])
            return arguments


        def Get_ini_seeds(k):
            Seeds = []
            for i in range(k):
                seed = []
                for j in range(dim):
                    seed.append(random.randint(min_I[j], max_I[j]))
                Seeds.append(seed)
            return Seeds
        
        

        def Symbolic_generation():

            arguments=Gen_arguments(Dat)
            sensitive_param=pro_at[0] # ordered from 0

            # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
            rank1 = 5
            rank2 = 1
            rank3 = 10
            T1 = 0.3
            
            # select the seed inputs
            inputs=Get_ini_seeds(1000)
            
            q = PriorityQueue()
            for inp in inputs:
                q.put((rank1,inp.copy()))
             
            visited_path = []

            while alter_visit>0 and q.qsize()!=0:
                t = q.get()
                t_rank = t[0]
                #t = np.array(t[1])
                t=t[1].copy()
                found=Judge_discriminatory(t)
                p=Get_path(np.array(inputs),np.array(t))

                if alter_visit<=0:
                    break
                temp=t.copy()
                temp=temp[0:sensitive_param]+temp[sensitive_param+1:len(t)]

                if p== None:
                    continue

                if found[0]==True:
                    if temp not in all_dis:
                        all_dis.append(temp.copy())
                    
                    for i in range(len(p)):
                        path_constraint = copy.deepcopy(p)
                        c = path_constraint[i]
                        if c[0] == sensitive_param:
                            continue

                        if c[1] == "<=":
                            c[1] = ">"
                            c[3] = 1.0 - c[3]
                        else:
                            c[1] = "<="
                            c[3] = 1.0 - c[3]

                        if path_constraint not in visited_path:
                            visited_path.append(path_constraint)
                            input = Local_solve(path_constraint, arguments, np.array(t), i)
                            
                            if input != None:
                                r = Average_confidence(path_constraint)
                                q.put((rank2 + r, input))

                # global search
                prefix_pred = []
                #print(p)
                if p== None:
                    continue
                for c in p:
                    if c[0] == sensitive_param:
                            continue
                    if c[3] < T1:
                        break

                    n_c = copy.deepcopy(c)
                    if n_c[1] == "<=":
                        n_c[1] = ">"
                        n_c[3] = 1.0 - c[3]
                    else:
                        n_c[1] = "<="
                        n_c[3] = 1.0 - c[3]
                    path_constraint = prefix_pred + [n_c]

                    # filter out the path_constraint already solved before
                    if path_constraint not in visited_path:
                        visited_path.append(path_constraint)
                        input = Global_solve(path_constraint, arguments, np.array(t))
                    
                        if input != None:
                            r = Average_confidence(path_constraint)
                            q.put((rank3-r, input))

                    prefix_pred = prefix_pred + [c]






        

        start_time=time.time()
        alter_visit=50000
        LIME_samples=2*dim
        num=0
        while True:
            num+=1
            Symbolic_generation()
            print(num,"   ",len(all_visit))
            if alter_visit<=0:
                break
        
        end_time=time.time()

        outfile_path2='Demo/Results/Numbers/SG/'+Dat.__class__.__name__+'_'+str(LIME_samples)+'_'+str(Exeiter)+'.txt'
        print(outfile_path2)
        
        with open(outfile_path2,'w+') as f2:
            print(str(result_for_vis))
            f2.write(str(result_for_vis))
            f2.write('\n')
            f2.write(str(end_time-start_time))

        #print(result_for_vis)



        outfile_path='Demo/Results/SG/'+Dat.__class__.__name__+'_'+str(LIME_samples)+'_'+str(Exeiter)+'.txt'
        print(outfile_path)
        f = open(outfile_path,'w+')

        for i in range(len(all_dis)):
            st=''
            for j in range(len(all_dis[i])):
                st=st+str(all_dis[i][j])+' '
            st=st+'\n'
            f.write(st)
        f.close

        

