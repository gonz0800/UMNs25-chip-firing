```python
%%time

# written by Angel Chavez, UMN, chave389@umn.edu
# tweaked by Annika Gonzalez-Zugasti
import numpy as np
from scipy.special import binom
from itertools import combinations

# k = number of branches
# m = number of chips per branch, total chips = k*m

def findStables(k,m):
    # here we pre-allocate the space needed to record all Rank 1 configurations
    rank_1 = int(binom(m * k, k))
    listOfConfigurations = np.zeros((rank_1, 2, m * k))
    Ci = 0  # length of listOfConfigurations, rather index of next to be added
    chips = list(range(m * k))  # list of chip indexes, not the names
    for combs_i in combinations(chips, k):
        for branch, chip_index in enumerate(combs_i):  # combs_i is list of chip indexes
            # enumerate(combs_i) returns index , value pair
            # branch is the index relative to combs_i,
            # the first chip of combs_i should be the smallest and go to branch 1
            # the last chip of combs_i should be the biggest and go to branch k
            # name of branch =/= index of branch
            # name of chip =/= index of chip
            listOfConfigurations[Ci, 0, chip_index] = branch + 1
            listOfConfigurations[Ci, 1, chip_index] = 1  # all go to level 1
        Ci += 1
    
    
    # start loop with level 2 until finished
    maxRank = int(m * (m + 1) / 2 + k * (m - 1) * m * (m + 1) / 6)  # height of decision tree
    # print(f"{maxRank=}")
    #for Rank in range(2, 3):
    for Rank in range(2, maxRank + 1):
        # print(f'{Rank=}')
    
        # first pre allocate space
        pre_alo = 0
        # for each configuration
        # figure out which vertices can be fired and how many ways
        # keeps running total
        for configuration in listOfConfigurations:
            columns, counts = np.unique(configuration, return_counts=True, axis=1)
            for index, count in np.ndenumerate(counts):
                if np.all(columns[:, index] == [0, 0]):
                    # [0, 0] is the center node
                    pre_alo += binom(count, k)
                else:
                    pre_alo += binom(count, 2)
    
        newListOfConfigurations = np.zeros((int(pre_alo), 2, m * k))  # define now, with correct size
        L = 0  # length of newListOfConfigurations, rather index of next to be added
    
        # for each configuration
        # figure out what can be fired and fire it,
        # add results to the list of configurations
        for configuration in listOfConfigurations:
            columns = np.unique(configuration, axis=1)
            for i in range(len(columns[0])):
                column = columns[:, i]
                if np.all(column == [0, 0]):
                    # we have the center node
                    indices = np.where(np.all(configuration == [[0], [0]], axis=0))[0]
                    # fire center node
                    for ids in combinations(indices, k):
                        # ids is a list (size k) of indexes of chips
                        # ids should be sorted already
    
                        # add the fired chips to the current configuration
                        newListOfConfigurations[L] = configuration
                        for [i, chip_id] in enumerate(ids):
                            # the first chip of combs_i should be the smallest and go to branch 1
                            # the last chip of combs_i should be the biggest and go to branch k
                            # name of branch =/= index of branch
                            branch = i + 1  # name of branch
                            newListOfConfigurations[L, 0, chip_id] = branch
                            newListOfConfigurations[L, 1, chip_id] = 1
                        L += 1  # length of newListOfConfigurations
                else:
                    # fire the non-center nodes
                    indices = np.where(np.all(configuration == column.reshape(2,1), axis=0))[0]
                    for i_1, i_2 in combinations(indices, 2):
                        newListOfConfigurations[L] = configuration
                        newListOfConfigurations[L, 1, i_1] -= 1
                        newListOfConfigurations[L, 1, i_2] += 1
                        # if entered level 0, then set branch to 0
                        if newListOfConfigurations[L, 1, i_1] == 0:
                            newListOfConfigurations[L, 0, i_1] = 0
                        L += 1  # length of newListOfConfigurations
    
        # print('deleting duplicates')
        listOfConfigurations = np.unique(newListOfConfigurations, axis=0)
    
    
    print(f'total stable configurations: {listOfConfigurations.shape[0]}')
    print(" ")
    # print(listOfConfigurations)
    # print as tableau
    N_syt = 0
    for configuration in listOfConfigurations:
        # convert configuration into a tableau
        tableau = np.zeros((k, m))
        for c in range(m * k):  # this is index, recall index of chip =/= name of chip
            # name of branch =/= index of branch
            # name of level =/= index of level
            branch = int(configuration[0, c]) - 1  # index
            level = int(configuration[1, c]) - 1  # index
            tableau[branch, level] = c + 1  # name of chip
        # print(tableau, '\n')
        # check if tableau is standard young tableau
        sorted_cols = np.all(tableau[:-1, :] <= tableau[1:, :])
        # sorted_rows = np.all(tableau[:, :-1] <= tableau[:, 1:])
        is_syt = sorted_cols  # and sorted_rows
        if not is_syt:
            N_syt += 1
            print(tableau, '\n')
    print(f'number of non-standard young tableaux: {N_syt}')
    print(" ")
```

    CPU times: user 154 ms, sys: 154 ms, total: 308 ms
    Wall time: -1.91e+09 ns



```python
%%time

findStables(1,5)
```

    total stable configurations: 1
     
    number of non-standard young tableaux: 0
     
    CPU times: user 175 ms, sys: 9.32 ms, total: 185 ms
    Wall time: 221 ms



```python
%%time

findStables(2,5)
```

    total stable configurations: 53
     
    [[ 1.  2.  6.  7.  8.]
     [ 3.  4.  5.  9. 10.]] 
    
    [[ 1.  2.  6.  7.  9.]
     [ 3.  4.  5.  8. 10.]] 
    
    [[ 1.  3.  6.  7.  8.]
     [ 2.  4.  5.  9. 10.]] 
    
    [[ 1.  3.  6.  7.  9.]
     [ 2.  4.  5.  8. 10.]] 
    
    [[ 1.  4.  5.  6.  7.]
     [ 2.  3.  8.  9. 10.]] 
    
    [[ 1.  4.  5.  6.  8.]
     [ 2.  3.  7.  9. 10.]] 
    
    [[ 1.  4.  5.  6.  9.]
     [ 2.  3.  7.  8. 10.]] 
    
    [[ 1.  4.  5.  7.  8.]
     [ 2.  3.  6.  9. 10.]] 
    
    [[ 1.  4.  5.  7.  9.]
     [ 2.  3.  6.  8. 10.]] 
    
    [[ 1.  4.  6.  7.  8.]
     [ 2.  3.  5.  9. 10.]] 
    
    [[ 1.  4.  6.  7.  9.]
     [ 2.  3.  5.  8. 10.]] 
    
    number of non-standard young tableaux: 11
     
    CPU times: user 12min 4s, sys: 4.7 s, total: 12min 8s
    Wall time: 12min 30s



```python
%%time

findStables(5,1)
```

    total stable configurations: 1
     
    number of non-standard young tableaux: 0
     
    CPU times: user 1.08 ms, sys: 12 μs, total: 1.09 ms
    Wall time: 1.17 ms



```python
%%time

findStables(5,2)
```

    total stable configurations: 42
     
    number of non-standard young tableaux: 0
     
    CPU times: user 619 ms, sys: 9.37 ms, total: 628 ms
    Wall time: 673 ms



```python
%%time

findStables(6,1)
```

    total stable configurations: 1
     
    number of non-standard young tableaux: 0
     
    CPU times: user 4.33 ms, sys: 0 ns, total: 4.33 ms
    Wall time: 6.97 ms



```python
%%time

findStables(6,2)
```

    total stable configurations: 132
     
    number of non-standard young tableaux: 0
     
    CPU times: user 3.87 s, sys: 26.4 ms, total: 3.9 s
    Wall time: 4.03 s



```python
%%time

findStables(7,1)
```

    total stable configurations: 1
     
    number of non-standard young tableaux: 0
     
    CPU times: user 301 μs, sys: 0 ns, total: 301 μs
    Wall time: 315 μs



```python
%%time

findStables(7,2)
```

    total stable configurations: 429
     
    number of non-standard young tableaux: 0
     
    CPU times: user 26.8 s, sys: 187 ms, total: 27 s
    Wall time: 26.8 s



```python

```
