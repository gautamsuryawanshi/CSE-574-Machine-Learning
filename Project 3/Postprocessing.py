
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: #
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""

import utils as u
import numpy as np
import itertools 
import random

def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    
    
    # Must complete this function!
    #return demographic_parity_data, thresholds

    return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}
    
    
    a=categorical_results['African-American']
    b=categorical_results['Caucasian']
    c=categorical_results['Hispanic']
    d=categorical_results['Other']
    
    
    (c1,d1,b1)=u.get_ROC_data(a, 'African-American')
    (c2,d2,b2)=u.get_ROC_data(b, 'Caucasian')
    (c3,d3,b3)=u.get_ROC_data(c, 'Hispanic')
    (c4,d4,b4)=u.get_ROC_data(d, 'Other')
   
  
    arr=[]
   
    for i in range(100):
        
        n1=c1[i]-0.01
        n2=c1[i]+0.01
        
        for j in range(100):
            if(c2[j]>=n1 and c2[j]<=n2):
                
                for k in range(100):
                    if(c3[k]>=n1 and c3[k]<=n2):
                        
                        for l in range(100):
                            if(c4[l]>=n1 and c4[l]<=n2):
                                #print("i am coming here")
                                p=[i/100.,j/100.,k/100.,l/100.]
                                arr.append(p)
                                
   
    
    accValue=0
    thresholdList=None
    abc = arr.copy()
    for i in range(len(arr)):
        arr1=u.apply_threshold(a, abc[i][0])
        arr2=u.apply_threshold(b, abc[i][1])
        arr3=u.apply_threshold(c, abc[i][2])
        arr4=u.apply_threshold(d, abc[i][3])
        
        
        
            
        d9={'African-American':arr1,
           'Caucasian':arr2,
           'Hispanic':arr3,
           'Other':arr4}
    
        acc=u.get_total_accuracy(d9)
        
        
        if(accValue<acc):
            accValue=acc
            thresholdList=arr[i]
            equal_opportunity_data={}
            equal_opportunity_data =d9
            
        d9={}
        
    
    
    thresholds={'African-American':thresholdList[0],
                'Caucasian':thresholdList[1],
                'Hispanic':thresholdList[2],
                'Other':thresholdList[3]}
    
   
    
    
    # Must complete this function!
    return equal_opportunity_data, thresholds

    #return None, None

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}
    
    
    
    a=categorical_results['African-American']
    b=categorical_results['Caucasian']
    c=categorical_results['Hispanic']
    d=categorical_results['Other']
    
    best_accuracy=0
    best_threshold=None
    
    
    l1=l2=l3=l4=[]
    for i in range(10):
        l1.append(i/10)
        
    l2=l3=l4=l1
    
    for i in (itertools.product(l1,l2,l3,l4)):
        arr1=u.apply_threshold(a, i[0])
        arr2=u.apply_threshold(b, i[1])
        arr3=u.apply_threshold(c, i[2])
        arr4=u.apply_threshold(d, i[3])
        
        
        d9={'African-American':arr1,'Caucasian':arr2,'Hispanic':arr3,'Other':arr4}
        acc=u.get_total_accuracy(d9)
        
        if(best_accuracy<acc):
            best_accuracy=acc
            best_threshold=[i[0],i[1],i[2],i[3]]
            mp_data={}
            mp_data=d9
        d9={}
        
        '''
        if(acc>=0.63):
            break
        '''
   
    
    thresholds={'African-American':best_threshold[0],
                'Caucasian':best_threshold[1],
                'Hispanic':best_threshold[2],
                'Other':best_threshold[3]}    
    
    
    
    

    # Must complete this function!
    return mp_data, thresholds
    
    #return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def func(prediction_label_pairs):
    true_positives = []
    
    for i in range(1, 101):
        threshold = float(i) / 100.0
        eval_copy = list.copy(prediction_label_pairs)
        eval_copy = u.apply_threshold(eval_copy, threshold)
        TPR = u.get_positive_predictive_value(eval_copy)
        true_positives.append(TPR)
        

    return (true_positives)
    

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    
    
    a=categorical_results['African-American']
    b=categorical_results['Caucasian']
    c=categorical_results['Hispanic']
    d=categorical_results['Other']
    
    a1=func(a)
    b1=func(b)
    c1=func(c)
    d1=func(d)
    
    best_accuracy=0
    best_threshold=None
    
    arr=[]
    
    for i in range(100):
        n1=a1[i]-0.01
        n2=a1[i]+0.01
        for j in range(100):
            if(b1[j]>=n1 and b1[j]<=n2):
                for k in range(100):
                    if(c1[k]>=n1 and c1[k]<=n2):
                        for l in range(100):
                            if(d1[l]>=n1 and d1[l]<=n2):
                                p=[i/100,j/100,k/100,l/100]
                                arr.append(p)
                                
    
    
    
    
    best_accuracy=0
    best_threshold=None
    
   
    testdata = list(set(tuple(x) for x in arr))
    
    
    myarray = np.asarray(arr)
    
    unique_dict_a={}
    unique_dict_b={}
    unique_dict_c={}
    unique_dict_d={}
    
    uniqueValues=np.unique(myarray)
    
    for i in range(len(uniqueValues)):
        i1 = uniqueValues[i]
        unique_dict_a.update({ i1 : u.apply_threshold(a, i1) } )
        unique_dict_b.update({ i1 : u.apply_threshold(b, i1) } )
        unique_dict_c.update({ i1 : u.apply_threshold(c, i1) } )
        unique_dict_d.update({ i1 : u.apply_threshold(d, i1) } )
    
   
    
    
    for i in range(10000):
        
        if(arr[i][0] in unique_dict_a.keys()):
            arr1=unique_dict_a[arr[i][0]]
        else:
            arr1=u.apply_threshold(a, arr[i][0])
            
        if(arr[i][0] in unique_dict_a.keys()):
            arr2=unique_dict_b[arr[i][1]]
        else:
            arr2=u.apply_threshold(b, arr[i][1])
            
            
        if(arr[i][0] in unique_dict_a.keys()):
            arr3=unique_dict_c[arr[i][2]]
        else:
            arr3=u.apply_threshold(c, arr[i][2])
            
        if(arr[i][0] in unique_dict_a.keys()):
            arr4=unique_dict_d[arr[i][3]]
        else:
            arr4=u.apply_threshold(d, arr[i][3])
        
        
         
        
    
        d9={'African-American':arr1,
           'Caucasian':arr2,
           'Hispanic':arr3,
           'Other':arr4}
    
        acc=u.get_total_accuracy(d9)
        
        
        if(best_accuracy<acc):
            best_accuracy=acc
            best_threshold=arr[i]
            predictive_parity_data={}
            predictive_parity_data =d9
        d9={}
        
    
    
    thresholds={'African-American':best_threshold[0],
                'Caucasian':best_threshold[1],
                'Hispanic':best_threshold[2],
                'Other':best_threshold[3]}
    
    
    # Must complete this function!
    return predictive_parity_data, thresholds
    
    #return None, None

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}
    
    
    a=categorical_results['African-American']
    b=categorical_results['Caucasian']
    c=categorical_results['Hispanic']
    d=categorical_results['Other']
    
    best_accuracy=0
    best_threshold=None
    
    for i in range(100):
        theshold=i/100
        arr1=u.apply_threshold(a, theshold)
        arr2=u.apply_threshold(b, theshold)
        arr3=u.apply_threshold(c, theshold)  
        arr4=u.apply_threshold(d, theshold)
        
        d9={'African-American':arr1,
           'Caucasian':arr2,
           'Hispanic':arr3,
           'Other':arr4}
    
        acc=u.get_total_accuracy(d9)
        
        if(best_accuracy<acc):
            best_accuracy=acc
            best_threshold=theshold
            single_threshold_data={}
            single_threshold_data =d9
        d9={}
        
    
    
    
    
    thresholds={'African-American':best_threshold,
                'Caucasian':best_threshold,
                'Hispanic':best_threshold,
                'Other':best_threshold}
    
    
    
    #Must complete this function!
    return single_threshold_data, thresholds
    
    #return None, None