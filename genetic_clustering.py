# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:46:52 2015

@author: Chux
"""
#import the libraries to be used
from __future__ import division
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random
from scipy.spatial import distance
import math
from sklearn.metrics import mean_squared_error


#load the data here by putting the name of the file in quotes
data = pd.read_csv('wine_Blend1.csv', header = None).values

#set all the variables to be used
num_of_features = data.shape[1]
num_of_instances = data.shape[0]
population = 6
global k 
k = 3
nan = float('nan')



#stores the missing data indices
def get_missing_index(np_array):
    '''returns two arrays corresponding to row and column of missing values'''
    return (np.where(np.isnan(np_array)))
missing_index = get_missing_index(data)
missing_rows = [x for x in missing_index[0]]

#converts all nan values to zero
data = np.nan_to_num(data)

complete_data = []
predicted_data =[]

#load 'complete' dataset here
c_data = pd.read_csv('wine_Blend1_UD_complete.csv', header = None).values



#genotype generator
def generate_genotype():
    genotype =[random.randint(0,k-1) for i in range(num_of_instances)]
    return genotype
    
#genotype pool generator
def generate_pool():
    '''gets population of genotypes'''
    g_pool =[]
    for i in range(population):
        g= generate_genotype()
        g_pool.append(g)
    return g_pool

#gets centers of each cluster
def get_centers(genotype, data):
    '''takes in a genotype string and returns the centers of each cluster'''
    centers =[]
    
    for i in range(k):
        if i in genotype:
            centers.append (np.nan_to_num(np.mean([data[x] for x in range(len(genotype)) if genotype[x]==i \
                                            and x not in set(missing_rows)], axis=0)))#
        else:
            centers.append(np.zeros(num_of_features))
    #return centers
    return (np.asarray(centers))

#applying of kmeans and silhouette function    
def get_fit_score (g_pool, data):
    '''returns a tuple of partition labels and fitness score for each genotype in the population'''
    count = 0
    fitness_scores = []
    partition_labels = []
    for i in g_pool:

        center = get_centers(i,data)
        clusterer = KMeans(n_clusters =k, max_iter =1, n_init=1, init=center)
        cluster_labels = clusterer.fit_predict(data)
        if len(set(cluster_labels)) >> 1:
            fitness_scores.append( silhouette_score(data,cluster_labels))
        else:
            fitness_scores.append(0.0)
            #fitness_scores.append((1/np.mean(distance.cdist(data,data))))
        partition_labels.append(list(cluster_labels))
        #print (silhouette_score(iris_data.data,cluster_labels), "Genotype %i" %count)
        count = count + 1
    return (partition_labels,fitness_scores)
    
#works on the partition labels to make them a list
def get_partition_list (kmeans_labels):
    #temp_list =[]
    partition_list = []
    for i in kmeans_labels:
        temp_list =[]
        for j in i:
            temp_list.append(j)
        partition_list.append (temp_list)
    return partition_list

def roulette_selection(fit_scores):
    '''takes in a list of fitness scores and randomly chooses one biasedly though. Returns index of chosen one'''
    rnd = random.random() * sum(fit_scores)
    for i, w in enumerate(fit_scores):
        rnd -= w
        if rnd < 0:
            return i
#basically does roulette selection the number of times needed            
def proportional_selection (fit_scores):
    '''takes in fitness scores and uses roulette selection to choose the fittest
    genotypes. Does this the number of population times'''
    selected_genotypes=[]
    for i in range(population):
        selected_genotypes.append(roulette_selection(fit_scores[1]))
    
    return selected_genotypes

#start of mutation one
def get_most_distant(p_label, data, center, cluster):
    '''gets the most distant point in the cluster and returns the length and index'''
    #count = 0
    d_point = (0,0)
    
    new_label = p_label.copy()
    
    for i, j, k  in zip (new_label, data, range(num_of_instances)):

        dist = distance.euclidean(j,center)   
        if i == cluster:
            if d_point[0] <= dist:
                d_point = (dist,k)
            
            else:
                pass
 
    return d_point

def get_new_cluster (p_label, data, center, cluster):
    '''gets the most distant point and other points join it depending on distance'''
    index = get_most_distant(p_label, data, center, cluster)[1]
    new_cluster = [x for x in range (30) if x not in set(p_label)]
    count = 0
    new_label = p_label.copy()
    for i,j,k in zip(new_label, data, range(150)):

        if i == cluster:
            if distance.euclidean(j, center) >= distance.euclidean(j,data[index]):
                #print (i,j,k)
               # p_labels[count] = new_cluster[0] #wrong
                new_label[count]= [x for x in range (30) if x not in p_label][0] #new_cluster[0]
        count = count + 1
    if new_cluster[0] >> k:
        k=k+1
    else:
        pass
    
    return new_label
    
def mutation_one(real_data, fit_score, selected_genotypes):
    '''does mutation operation one on half of the data'''
    genotypes = selected_genotypes[0:int(population/2)] #take 50% for first mutation
    new_genotype_one = []
    
    for i in genotypes:
        random_cluster =random.choice(fit_score[0][i])#choose random gene (cluster)
        c_center = get_centers(fit_score[0][i], real_data)[random_cluster] #get center of cluster
        #new_genotype_one.append (get_most_distant(partition_labels[i], data, c_center, random_cluster))
        new_genotype_one.append ((get_new_cluster(fit_score[0][i], real_data,c_center,random_cluster )))#, random_cluster))
        
    return new_genotype_one
    
#start of mutation two
def get_next_centers(genotype,random_cluster,data):
    '''get centers for a for the other clusters apart from random_cluster'''
    next_centers = []
    other_clusters = ([x for x in set(genotype) if x!=random_cluster])
    for i in other_clusters:
        next_centers.append((get_centers(genotype,data)[i], i))
    return next_centers
    
def split_cluster(p_label, data, r_cluster):
    '''splits a cluster an the points go to the nearest center'''
    new_label = p_label.copy()
    new_centers = get_next_centers(new_label,r_cluster, data)
    
    for i,j,k in zip(new_label, data, range(num_of_instances)):
        if i == r_cluster:
           new_label[k] =  min([(distance.euclidean(data[k],new_centers[i[0]][0]),new_centers[i[0]][1])\
                                for i in enumerate(new_centers)])[1]
    return new_label
    
def mutation_two(data, fit_score, selected_genotypes):
    '''implements mutation operator 2 and returns new genotypes'''
    genotypes = selected_genotypes[int(population/2):int(population)] #select upper half of gene pool
    new_genotype_two = []
    for i in genotypes:
        random_cluster =random.choice(fit_score[0][i])#choose random gene (cluster)
        new_genotype_two.append(((split_cluster(fit_score[0][i], data, random_cluster)),random_cluster))
    return new_genotype_two
    
def get_fittest_genotype(scores):
    '''gets the fitness scores and returns the fittest genotype (the strings ooo)'''
    m_score = max(scores[1])
    fittest = []
    
    for i in range (len(scores[1])):
        if scores[1][i] == m_score:
            fittest = scores[0][i]
    return fittest
    
#start of weighted KNN

def get_weights(missing_row, full_row):
    
    num = 0
    denum = 0
    for i,j in zip(missing_row,full_row):
        if np.isnan(i) != True:
            num = num + float(((i*j)*((i-j)**2)))
            denum = denum + float((i*j))
            
        else:
                pass
                
    if num == 0 or denum == 0:
        return 1.0
    else:
        return (float(1/math.sqrt(num/denum)))
    
def get_sum_weights(full_array):
    '''gets the sum of all the weights for the non-missing data for each column'''   
    full_data = []
    nan_data = []
    sum_weights =[]
    
    
    for i in range(full_array.shape[0]):
        if i not in set(missing_rows):
            full_data.append(full_array[i])
        else:
            nan_data.append(full_array[i])
        
    for i in nan_data:
        weights = 0.0
        for j in full_data:
            weights = weights + (get_weights(i,j))
            
        sum_weights.append(weights)
    
    return sum_weights
    
def input_values (full_array, fittest_genotype):
    #full_array = np.copy(an_array)
    indices = missing_index
    
    sum_scores = dict(zip(set(missing_rows), get_sum_weights(full_array)))
   
    for x,y in zip(*indices):
        values = 0.0
        for j in range (len(fittest_genotype)):
            if j not in (set(missing_rows)) and (fittest_genotype[x] == fittest_genotype[j]): 
                values = values + (get_weights(full_array[x], full_array[j])*full_array[j][y])
            else:
                pass
        print ("missing values for %s,%s is %f" %(x,y,float(values/sum_scores[x])))


        full_array[x,y] = ((values/sum_scores[x]))
    return full_array

#execution starts here



for i in range(20):
    print ("generation %s" %(i))    
    gene_pool = generate_pool()
    fitness_score = get_fit_score(gene_pool, data)
    fittest_g = get_fittest_genotype(fitness_score)
    data = input_values(data, fittest_g)
    p_select = proportional_selection(fitness_score)
    gene_pool = mutation_one(data, fitness_score, p_select) + mutation_two(data, fitness_score, p_select)
    

for i,j in zip(*missing_index):
    complete_data.append(c_data[i][j])
    predicted_data.append(data[i][j])
    print ("complete data is %f and incomplete data is %f" %(c_data[i][j],data[i][j]))


    
print ("mean square error is: %f" %(mean_squared_error(complete_data, predicted_data)))
    
