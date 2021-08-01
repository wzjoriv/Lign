import lign as lg
import torch as th

## Create a new graph with 5 empty nodes
n = 5
g = lg.Graph()
g.add(n)

## Create/Overwrite a data property and set its values 
# Size of data on dimension 0 must match number of nodes in the graph
# Use set_data() with parameters nodes=[list] to change specific nodes
g['x'] = th.rand(n, 3)  # Or, g.set_data('x', th.rand(n, 3))

## Below presents various ways to see your data
# To view the whole graph
print(g)

# To view a data property
print(g['x'])  # Or, g.get_data('x')

# To view node(s) properties
print(g[0])  # Or, g.get_nodes(0)
print(g[[1, 2]])  # Or, g.get_nodes([1, 2])
print(g[3:4])  # Or, g.get_nodes(slice(3, 4))

# To view the edges for node(s)
print(g[(4,)])  # Or, g.get_edges(4)
