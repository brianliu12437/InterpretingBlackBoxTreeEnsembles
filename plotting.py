### Plotting functions
import numpy as np
from itertools import count
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout



class EnsemblePlot:
    """Attributes:"""

    def __init__(self, tree_list, feature_names, node_array, method ):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.node_array = node_array
        self.method = method

    #Helper Functions
    def prune_tree_leaves(self, tree1, inds):
        inds = np.array(inds)
        children_left = tree1.tree_.children_left.copy()
        children_right = tree1.tree_.children_right.copy()

        node_count = tree1.tree_.node_count
        nodes = np.ones(node_count) # main array of trees to prune: 1 on, 0 off

        nodes_off = set(np.where([children_left<0])[1][~inds.astype(bool)])
        nodes[list(nodes_off)] = 0
        while True:
            left_off = [i for i, e in enumerate(children_left) if e in nodes_off]
            right_off = [i for i, e in enumerate(children_right) if e in nodes_off]
            children_left[left_off] = -99
            children_right[right_off] = -99
            index_off = np.where((children_left == -99) & (children_right == -99))[0]
            nodes[index_off] = 0
            t1 = len(nodes_off)
            nodes_off.update(list(index_off))
            t2 = len(nodes_off)
            if t1 == t2:
                break
        return nodes

    def get_node_depths(self, tree1):
        """
        Get the node depths of the decision tree

        >>> d = DecisionTreeClassifier()
        >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
        >>> get_node_depths(d.tree_)
        array([0, 1, 1, 2, 2])
        """
        def get_node_depths_(current_node, current_depth, l, r, depths):
            depths += [current_depth]
            if l[current_node] != -1 and r[current_node] != -1:
                get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
                get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

        depths = []
        get_node_depths_(0, 0, tree1.tree_.children_left, tree1.tree_.children_right, depths)
        return np.array(depths)

    def prune_tree_layers(self,tree1,inds):
        inds = np.array(inds)
        node_depths = self.get_node_depths(tree1)
        depth_threshold = sum(inds)
        nodes = np.ones(len(node_depths))
        nodes[node_depths > depth_threshold] = 0
        return nodes


    def prune_tree_graph(self, tree1, ind1,feature_names, tree_id, method = 'leaves',
                                show_pruned = False ):
        nnodes = tree1.tree_.node_count
        feats = tree1.tree_.feature
        children_left = tree1.tree_.children_left
        children_right = tree1.tree_.children_right

        G = nx.DiGraph()
        nodes = set()

        if np.any(ind1!=1) & (method == 'leaves'):
            pruned = self.prune_tree_leaves(tree1,ind1)
        elif np.any(ind1!=1) & (method == 'layers'):
            pruned = self.prune_tree_layers(tree1,ind1)
        else:
            pruned = np.ones(nnodes)

        for i in range(0,nnodes):

            name_start = str(i) + "_" + str(tree_id)

            if (i not in nodes):
                if (pruned[i]!= 0):
                    G.add_nodes_from([(name_start, {'tree_id': tree_id,
                                        'feature': feature_names[feats[i]],
                                        'pruned': False })])
                    nodes.add(i)
                elif (show_pruned == True) & (pruned[i]!= 0):
                    G.add_nodes_from([(name_start, {'tree_id': tree_id,
                                        'feature': feature_names[feats[i]],
                                        'pruned': True })])
                    nodes.add(i)

            if (children_left[i] > 0):
                name = str(children_left[i]) + "_" + str(tree_id)
                if (pruned[children_left[i]]!= 0):
                    if children_left[i] not in nodes:
                        G.add_nodes_from([(name,{'tree_id': tree_id,
                                                'feature': feature_names[feats[children_left[i]]],
                                                'pruned': False})])
                        nodes.add(children_left[i])
                    G.add_edge(name_start,name,  color = 'black', style = 'solid')

                elif (show_pruned == True) & (pruned[children_left[i]] == 0):
                    if children_left[i] not in nodes:
                        G.add_nodes_from([(name,{'tree_id': tree_id,
                                                'feature': feature_names[feats[children_left[i]]],
                                                'pruned': True})])
                        nodes.add(children_left[i])
                    G.add_edge(name_start,name,  color = 'black', style = 'dashed')

            if (children_right[i] > 0 ):
                name = str(children_right[i]) + "_" + str(tree_id)
                if (pruned[children_right[i]]!= 0):
                    if children_right[i] not in nodes:
                        name = str(children_right[i]) + "_" + str(tree_id)
                        G.add_nodes_from([(name,{'tree_id': tree_id,
                                                'feature': feature_names[feats[children_right[i]]],
                                                'pruned': False})])
                        nodes.add(children_right[i])
                    G.add_edge(name_start,name, color = 'black', style = 'solid')

                elif (show_pruned == True) & (pruned[children_right[i]] == 0):
                    if children_right[i] not in nodes:
                        name = str(children_right[i]) + "_" + str(tree_id)
                        G.add_nodes_from([(name,{'tree_id': tree_id,
                                                'feature': feature_names[feats[children_right[i]]],
                                                'pruned': True})])
                        nodes.add(children_right[i])
                    G.add_edge(name_start,name, color = 'black', style = 'dashed')

        #set nodes with no children as leaf nodes after pruning
        childless_nodes = set([i for i in G.nodes()]) - set([i[0] for i in G.edges()])
        for i in childless_nodes:
            G.nodes()[i]['feature'] = 'leaf'

        return G

    def prune_ensemble_graph(self, tree_list, ind_list, feature_names, method = 'leaves',
                                show_pruned = False ):

        nz = list(np.where([sum(i)!=0 for i in ind_list]))[0]
        graphs = []
        for i in range(len(nz)):
            tree1 = tree_list[nz[i]]
            ind1 = np.array(ind_list)[nz[i]]
            G = self.prune_tree_graph(tree1, ind1,feature_names, tree_id = i,
                                method = method, show_pruned = show_pruned )
            graphs.append(G)

        G_all = nx.compose_all(graphs)
        return G_all, graphs


    ### Plotting functions
    def get_colors(self,G, plot_legend = False , cmap = []):
        groups = set(nx.get_node_attributes(G,'feature').values())
        mapping = dict(zip(sorted(groups),count()))
        mapping['leaf'] = -1
        nodes = G.nodes()
        node_colors = [mapping[G.nodes[n]['feature']] for n in nodes]

        if len(cmap) == 0:
            cmap = plt.cm.tab20

        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (0, 0, 0, 0.1)
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)

        cNorm  = mpl.colors.Normalize(vmin=-1, vmax=max(node_colors))
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        if plot_legend == True:
            f = plt.figure(1)
            ax = f.add_subplot(1,1,1)
            for label in mapping:
                ax.plot([0],[0],color=scalarMap.to_rgba(mapping[label]),
                                            label=label,linewidth = 10)
            plt.legend(loc = 8)
            f.tight_layout()
            plt.axis('off')
            f.set_facecolor('w')
            plt.show()

        return node_colors, cmap , [mapping, scalarMap]

#Helper Functions for Plotting
    def get_pruned_edges_nodes(self, G): # add to plotting helpers
        E_keep = []
        E_pruned = []
        for g in G.edges():
            temp = G.edges()[g]
            if temp['style'] == 'solid':
                E_keep.append(g)
            else:
                E_pruned.append(g)

        N_keep = []
        N_pruned = []
        ind_keep = []
        i = 0
        for n in G.nodes:
            temp = G.nodes[n]
            if temp['pruned'] == False:
                N_keep.append(n)
                ind_keep.append(i)
            else:
                N_pruned.append(n)
            i = i+1

        return E_keep, E_pruned, N_keep, N_pruned, ind_keep

    def pos_grid_layout(self, G, nrow, offset = 40):
        """
        TODO: for depth pruning, offset each row by the corresponding column
        G: networkx graph object
        nrow: number of rows in the grid
        offset: pixel distance between rows.
        """
        pos=graphviz_layout(G, prog='dot')
        keys = np.array(list(pos.keys()))
        trees = np.unique([k.split('_')[1] for k in keys])
        per_row = int(np.ceil(len(trees)/nrow))


        row = 0
        row_starts = trees[::per_row][1:]
        y_maxes = []
        r_maxes = []

        x_mins = []
        r_mins = []

        global_xmin = np.min([pos[k][0] for k in keys])

        for t in trees:
            tree_keys = keys[[k.split('_')[1] == t for k in keys]]

            if t in row_starts:
                r_maxes.append(np.max(y_maxes))
                row = row + 1
                x_min = np.min([pos[k][0] for k in tree_keys])
                xshift = x_min - global_xmin

            if row > 0:
                for key in tree_keys:
                    temp = pos[key]
                    pos.update({key:(temp[0] - xshift,temp[1] + r_maxes[row-1] + offset)})


            y_max = np.max([pos[k][1] for k in tree_keys])
            y_maxes.append(y_max)

        return pos
