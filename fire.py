import pandas as pd

import scipy
from scipy.sparse import csc_matrix
import copy
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

def r_squared(yTest,pred,yTrain):
    top = np.sum((yTest - pred)**2)
    bottom = np.sum((yTest - np.mean(yTrain))**2)
    return 1 - top/bottom

def mcpthresh(w_j,lambda1,gamma):
    if np.abs(w_j) <= lambda1*gamma:
        return soft_threshold(w_j,lambda1)/(1-(1/gamma))
    else:
        return w_j
    
def soft_threshold(w, lambda_):
    return (w / abs(w)) * max(abs(w) - lambda_, 0) if abs(w) > lambda_ else 0

def mcploss(w,lambda1,gamma):
    cond1 = (w <= lambda1*gamma)
    cond2 = (w >= lambda1*gamma)
    mcp = np.zeros(len(w))
    mcp[cond1] = lambda1*w[cond1]- (w[cond1]*w[cond1])/(2*gamma)
    mcp[cond2] = 0.5*gamma*lambda1*lambda1
    return sum(mcp)

def mcpsingle(w_j,lambda1,gamma):
    if np.abs(w_j) <= lambda1*gamma:
        return lambda1*w_j - (w_j*w_j)/(2*gamma)
    else:
        return 0.5*gamma*lambda1*lambda1

def MCP_CD(y,M,lambda1,gamma,threshold= 10**-3,ws = []):
    n = M.shape[0]
    p = M.shape[1]

    loss_sequence = []
    cycle_loss = []
    converged = False
    
    if len(ws) == 0:
        w = np.zeros(p)
    else:
        w = ws
    
    r = y - M@w
    mcp_penalty = mcploss(np.abs(w),lambda1,gamma)
    ind = 0
    while converged == False:
        j = ind%p
        M_j = M[:,j]
        w_j = w[j]

        r = r + M_j*w_j
        mcp_penalty = mcp_penalty - mcpsingle(np.abs(w_j),lambda1,gamma)


        w_sol = M_j@r/(M_j@M_j)
        w[j] = mcpthresh(w_sol,lambda1,gamma)

        r = r - M_j*w[j]
        mcp_penalty = mcp_penalty + mcpsingle(np.abs(w[j]),lambda1,gamma)

        loss = (0.5/n)*r@r + mcp_penalty
        loss_sequence.append(loss)

        ind = ind + 1

        if (j == 0):
            cycle_loss.append(loss)
            if len(cycle_loss) >= 2:
                converged = np.abs(cycle_loss[-1]-cycle_loss[-2])<= threshold
    return w, loss_sequence

def analyze_nodes(nodes,w):
    nodes1 = copy.deepcopy(nodes)
    counter = 0
    for i in range(len(nodes1)):
        for j in range(len(nodes1[i])):
            if w[counter] == 0:
                nodes1[i][j] = int(0)
            else:
                nodes1[i][j] = int(1)
            counter = counter+1
    return nodes1


def get_tree_matrix_sparse(X,tree1):
    leaf_all = np.where(tree1.tree_.feature < 0)[0]
    leaves_index = tree1.apply(X.values)
    leaves = np.unique(leaves_index)
    values = np.ndarray.flatten(tree1.tree_.value)
    leaves_values = [values[i] for i in leaves_index]
    df = pd.DataFrame(np.column_stack((range(0,len(leaves_index)),leaves_index,leaves_values))
             ,columns = ['instance','node','value'])
    setdiff = list(set(leaf_all) - set(np.unique(leaves_index)))
    toadd = pd.DataFrame(np.column_stack((np.zeros(len(setdiff)),setdiff,np.zeros(len(setdiff)))),
                        columns = ['instance','node','value'])
    #df = df.append(toadd)
    df = pd.concat([df, toadd, ignore_index=True)
    matrix_temp = pd.pivot_table(df, index = 'instance',columns = 'node',values = 'value').fillna(0)
    return csc_matrix(matrix_temp.values), matrix_temp.columns.values

def get_rule_matrix_sparse(X,tree_list):
    matrix_full, nodes = get_tree_matrix_sparse(X,tree_list[0])
    node_list = [nodes]
    for tree1 in tree_list[1:]:
        matrix_temp, nodes = get_tree_matrix_sparse(X,tree1)
        node_list.append(nodes)
        matrix_full = scipy.sparse.hstack([matrix_full,matrix_temp])
    return matrix_full.tocsc(), node_list # node list gives the indicies of the nodes in the tree structure
