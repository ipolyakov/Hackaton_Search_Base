import pickle, os, time, gdown
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg 
from .search import Base

import random

#import cProfile

import gc

#import memory_profiler

#def profile(func):
#    """Decorator for run function profile"""
#    def wrapper(*args, **kwargs):
#        profile_filename = func.__name__ + '.prof'
#        profiler = cProfile.Profile()
#        result = profiler.runcall(func, *args, **kwargs)
#        profiler.dump_stats(profile_filename)
#        return result
#    return wrapper

def cos_sim(vec, query):
    result = np.dot(vec, query)
    return result

class Leaf:
    def __init__(self, vec, num):
        self.vec = vec
        self.num = num

    def search(self, query):
        sim = cos_sim(self.vec, query)
        if sim > 0.6:
            return [self.num, sim]
        return []

    def optimize(self):
        pass

class Node:
    def __init__(self, reg_matrix, ids, depth, left_number):
        self.depth = depth
#        print("reg_matrix len: ", len(reg_matrix), " depth ", depth, " left number ", left_number)
        best_pc = None
        best_balance = 2.5

#        for i  in range(2):
#        for i  in range(3):
        for i  in range(5):
            pc = self.calc_pc(reg_matrix)
            left, left_ids, right, right_ids, balance = self.try_break(reg_matrix, ids, pc)
            if balance < best_balance:
                best_pc = pc
                best_balance = balance
                best_left = left
                best_right = right
                best_left_ids = left_ids
                best_right_ids = right_ids

#        print("Best balance: ", best_balance, " ", len(best_left), " ", len(best_right), " " , len(reg_matrix))
        self.pc = best_pc
        self.left = create_node(best_left, best_left_ids, depth+1, left_number)
        self.right = create_node(best_right, best_right_ids, depth+1, left_number + len(best_left))

    def try_break(self, reg_matrix, ids, pc):
        left = []
        left_ids = []
        right =[]
        right_ids = []
        balanced_left = 0
        balanced_right = 0
        for vec, idd in zip(reg_matrix, ids):
            sim = cos_sim(vec, pc)
#            if sim > -0.02
            if sim > -0.004:
                left.append(vec)
                left_ids.append(idd)
            if sim > 0.05:
                balanced_left = balanced_left + 1
#            if sim < 0.02:
            if sim < 0.004:
                right.append(vec)
                right_ids.append(idd)
            if sim < -0.05:
                balanced_right = balanced_right + 1
        balance = (len(reg_matrix) - (balanced_left + balanced_right)) / len(reg_matrix)
        return left, left_ids, right, right_ids, balance
        

    def calc_pc(self, reg_matrix):
        vec = random.choice(reg_matrix)
        x = np.random.randn(512)
        x -= x.dot(vec) * vec
        x /= np.linalg.norm(x)
        return x

    def optimize(self):
        self.left.optimize()
        self.right.optimize()
        if self.depth < 5:
            gc.collect()

    def search(self, query):
        sim = cos_sim(query, self.pc)
        result = []
#        if sim > -0.05:
        if sim > -0.0485:
            result = result + self.left.search(query)
#        if sim < 0.05:
        if sim < 0.0485:
            result = result + self.right.search(query)
        return result

class Empty:
    def search(self, query):
        return []

    def optimize(self):
        pass

class OurList:
    def __init__(self, reg_matrix, ids):
        self.reg_matrix = reg_matrix
        self.ids = ids

    def optimize(self):
#        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)
        self.reg_matrix = np.array(self.reg_matrix)
#        self.reg_matrix = np.ascontiguousarray(self.reg_matrix)

    def search(self, query):
        similarity = np.dot(self.reg_matrix, query)
        return [(self.ids[i], sim) for i, sim in enumerate(similarity)]

def create_node(reg_matrix, ids, depth, left_number):
#    if reg_matrix.shape[0] == 1:
    if len(reg_matrix) == 1:
        return Leaf(reg_matrix[0], ids[0])
#    if reg_matrix.shape[0] < 1:
    if len(reg_matrix) < 1:
        return Empty()
    if depth > 13:
        return OurList(reg_matrix, ids)
    return Node(reg_matrix, ids, depth, left_number)

class SearchSolution(Base):
    ''' SearchBase class implements 
    search through the database to find matching
    vector for query vector. It measures
    search speed and assign a score based
    on the search time and correctness of the
    search 
    '''
    # @profile
    def __init__(self, data_file='./data/train_data.pickle', 
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        '''
        Creates regestration matrix and passes 
        dictionary. Measures baseline speed on
        a given machine
        '''
        print('reading from pickle')
        self.data_file = data_file
        self.data_url = data_url

#    @memory_profiler.profile
    def set_base_from_pickle(self):
        '''
        Downloads the data, if it does not exist.
        Sets reg_matrix and pass_dict

        reg_matrix : np.array(N, 512)
        pass_dict : dict -> dict[idx] = [np.array[1, 512]]
        '''
        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data') 
            gdown.download(self.data_url, self.data_file, quiet=False) 

        with open(self.data_file, 'rb') as f:
            data = pickle.load(f) 
    
        self.reg_matrix = [None] * len(data['reg'])
        self.ids = {} 
        for i, key in enumerate(data['reg']):
            self.reg_matrix[i] = data['reg'][key][0][None]
            self.ids[i] = key

        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)
#        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)[:200000] # NB!!!
        self.pass_dict = data['pass']

        del data
        gc.collect()

#        print("n vec", len(self.reg_matrix))
        self.root = create_node(self.reg_matrix, self.ids, 0, 0)
        del self.reg_matrix
        self.reg_matrix = None
        gc.collect()

        self.root.optimize()

    
    # @profile
    def cal_base_speed(self, base_speed_path='./base_speed.pickle') -> float:
        '''
        Validates baseline and improved searh
        Return:
                metric : float - score for search
        ''' 

        samples = cfg.samples 
        N, C, C_time, T_base = 0, 0, 0, 0
        for i, tup in enumerate(tqdm(self.pass_dict.items(), total=samples)):

            idx, passes = tup
            for q  in passes:
                t0 = time.time()
                c_output = self.search(query=q) 
                t1 = time.time()
                T_base += (t1 - t0)

                C_set = [True for tup in c_output if tup[0] == idx]
                if len(C_set):
                    C += 1
                    C_time += (t1 - t0) 
                N += 1

            if i > samples:
                break

        base_speed = T_base / N
        print(f"Base Line Speed: {base_speed}")
        print(f"Base Line Accuracy: {C / N * 100}")
        with open(base_speed_path, 'wb') as f:
            pickle.dump(base_speed, f)

#    @profile
    def search(self, query: np.array) -> List[Tuple]:
        '''
        Baseline search algorithm. 
        Uses simple matrix multiplication
        on normalized feature of face images

        Arguments:
            query : np.array - 1x512
        Return:
            List[Tuple] - indicies of search, similarity
        '''
        return self.root.search(query)
    

    def insert_base(self, feature: np.array) -> None:

        ## there no inplace concationation in numpy so far. For inplace
        ## concationation operation both array should be contingious in 
        ## memory. For now, let us suffice the naive implementation of insertion
        self.reg_matrix = np.concatenate(self.reg_matrix, feature, axis=0) 

    def cos_sim(self, query: np.array) -> np.array:
        return np.dot(self.reg_matrix, query)
