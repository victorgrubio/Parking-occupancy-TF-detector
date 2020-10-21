#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:51:36 2018

@author: visiona
"""
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('load_ext', 'Cython')
    
# %load_ext Cython
#%%cython
def primes(kmax):
    p = [None] * 1000 # Initialize the list to the max number of elements
    if kmax > 1000:
        kmax = 1000
    result = []
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result

print(primes(10))   
if ipy is not None:
    ipy.run_line_magic('timeit', 'primes(1000)')
    
