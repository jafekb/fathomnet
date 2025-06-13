import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from ete3 import Tree
from fathomnet.api import worms

with open('/home/bjafek/Nuro/benj_prac/fathomnet/data/dataset_train.json', 'r') as f:
    data = json.load(f)
classes = [x['name'] for x in data['categories']]
classes

def recursive_child_snatcher(anc):
    # Recursively gets a list of children and ranks from a fathomnet ancestor.
    children = [x.name for x in anc.children]
    childrens_ranks = [x.rank for x in anc.children]

    assert len(children) == 1 # bifurcating trees not implemented
    if len(anc.children[0].children) > 0:
        childrens_children, childrens_childrens_ranks = recursive_child_snatcher(anc.children[0])
        return children + childrens_children, childrens_ranks + childrens_childrens_ranks
    else:
        return children, childrens_ranks

# convert to an ete3 Tree (This is personal preference as I have worked with them before)
tree = Tree()
already_added = ['']
for label in tqdm(classes):
    if label in already_added:
        continue

    anc = worms.get_ancestors(label)
    children, ranks = recursive_child_snatcher(anc)
    children = [''] + children
    ranks = [''] + ranks
    for i in range(len(children)-1):
        parent_name, child_name = children[i:i+2]
        parent_rank, child_rank = ranks[i:i+2]
        if child_name in already_added:
            continue

        parent_node = [node for node in tree.traverse() if node.name == parent_name][0]
        parent_node.rank = parent_rank
        child = Tree(name=child_name)
        child.rank = child_rank
        parent_node.add_child(child)
        already_added += [child_name]
print(tree)

# set distances to 0 for ranks not included in loss calculation
for node in tree.traverse():
    if node.name in classes:
        continue
    accepted_ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    if node.rank not in accepted_ranks:
        node.dist = 0

# make distance matrix
def tree_to_distance_matrix(tree, labels):
    n = len(labels)
    labels = sorted(labels)

    # Create a blank distance matrix
    dist_matrix = np.zeros((n, n))

    # Fill the matrix with pairwise distances
    for i, name1 in enumerate(labels):
        node1 = [node for node in tree.traverse() if node.name == name1][0]
        for j, name2 in enumerate(labels):
            if i <= j:

                d = node1.get_distance(str(name2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d  # symmetric

    df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    return df
df = tree_to_distance_matrix(tree, classes)

arr = df.values
import pdb; pdb.set_trace()
