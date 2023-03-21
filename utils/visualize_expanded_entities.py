import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('../entities/queries_entities_expanded1_title.tsv', sep='\t', header=0)
df['bfs'] = df['bfs'].apply(eval)
df['dfs'] = df['dfs'].apply(eval)
df['ppr'] = df['ppr'].apply(eval)

for index, row in df.iterrows():
    print(row)

query_id = 2
data_id = 0

bfs_Graph = nx.Graph()
dfs_Graph = nx.Graph()
ppr_Graph = nx.Graph()
print(df['qid'][query_id])

bfs_Graph.add_nodes_from([df['qid'][query_id]])
bfs_Graph.add_nodes_from(df['bfs'][data_id][:5])
bfs_Graph.add_edges_from([(df['qid'][query_id], node) for node in df['bfs'][data_id][:5]])

dfs_Graph.add_nodes_from([df['qid'][query_id]])
dfs_Graph.add_nodes_from(df['dfs'][data_id][:5])
dfs_Graph.add_edges_from([(df['qid'][query_id], node) for node in df['dfs'][data_id][:5]])

ppr_Graph.add_nodes_from([df['qid'][query_id]])
ppr_Graph.add_nodes_from(df['ppr'][data_id])
ppr_Graph.add_edges_from([(df['qid'][query_id], node) for node in df['ppr'][data_id]])

# draw graphs in one figure with titles
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
ax1.set_title('BFS')
ax2.set_title('DFS')
ax3.set_title('PPR')
nx.draw(bfs_Graph, with_labels=True, ax=ax1, node_size=1000)
nx.draw(dfs_Graph, with_labels=True, ax=ax2, node_size=1000)
nx.draw(ppr_Graph, with_labels=True, ax=ax3, node_size=1000)
plt.savefig('graphs.png')
plt.show()
