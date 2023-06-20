"""
Read extracted entities file and add relevant page_ids to the file
"""

from tqdm import tqdm
import pickle
import argparse
from scipy.sparse import csr_matrix
import numpy as np
from fast_pagerank import pagerank_power

index_file = '../resources/linked_pages.pkl'
csr_matrix_file = '../resources/graph_csr_format.pkl'
page_links_file = '../resources/Wikinformetrics/page_link.tsv'

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='entities input file', default='../entities/queries_entities.tsv')
parser.add_argument('-o', '--output', help='entities output file', default='../entities/queries_entities_expanded.tsv')
parser.add_argument('-k', '--number', help='number of nodes to be expanded from each entity', default=5, type=int)
parser.add_argument('-p', '--pagerank', help='whether to use pagerank or not', default=False, type=bool)
args = parser.parse_args()


def create_index_file():
    with open(page_links_file, 'r') as f:
        lines = f.readlines()

    linked_pages = {}
    for line in tqdm(lines[1:]):
        pair = line.split('\t')
        if int(pair[1]) not in linked_pages:
            linked_pages[int(pair[1])] = []
            linked_pages[int(pair[1])].append(int(pair[0].rstrip()))
        else:
            linked_pages[int(pair[1])].append(int(pair[0].rstrip()))

    with open(index_file, 'wb') as f:
        pickle.dump(linked_pages, f)

    return linked_pages


try:
    with open(index_file, 'rb') as f:
        print('Loading index file...')
        linked_pages = pickle.load(f)
except FileNotFoundError:
    print('Index file not found. Creating index file...')
    linked_pages = create_index_file()


def create_csr_matrix():
    with open(page_links_file, 'r') as f:
        lines = f.readlines()

    edges = []
    for line in tqdm(lines[1:]):
        pair = line.split('\t')
        edges.append((int(pair[1].rstrip()), int(pair[0])))
    edges = np.array(edges)
    max_index = max(np.max(edges[:, 0]), np.max(edges[:, 1]))
    G = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(max_index + 1, max_index + 1))

    with open(csr_matrix_file, 'wb') as f:
        pickle.dump(G, f)

    del edges
    return G


try:
    with open(csr_matrix_file, 'rb') as f:
        print('Loading Graph...')
        G = pickle.load(f)
except FileNotFoundError:
    print('Graph not found. Creating Graph...')
    G = create_csr_matrix()


def dfs(graph, start, k):
    visited, stack = [], [start]
    while stack and len(visited) < k:
        vertex = stack.pop()
        if vertex not in visited:
            if vertex != start:
                visited.append(vertex)
            connected_nodes = graph.get(vertex, [])
            if connected_nodes:
                stack.extend(set(graph[vertex]) - set(visited))
    return visited


def bfs(graph, start, k):
    visited, queue = [], [start]
    while queue and len(visited) < k:
        vertex = queue.pop(0)
        if vertex not in visited:
            if vertex != start:
                visited.append(vertex)
            connected_nodes = graph.get(vertex, [])
            if connected_nodes:
                queue.extend(set(graph[vertex]) - set(visited))
    return visited


with open(args.input, 'r', encoding='utf-8') as in_file, open(args.output, 'w', encoding='utf-8') as out_file:
    lines = in_file.readlines()
    for line in tqdm(lines):
        qid, entities = line.split('\t')
        entities = eval(entities)

        bfs_relevant_nodes = []
        dfs_relevant_nodes = []
        ppr_relevant_nodes = []

        seed = np.zeros(G.shape[0])
        for entity in entities:
            bfs_relevant_nodes.extend(bfs(linked_pages, entity['id'], args.number))
            dfs_relevant_nodes.extend(dfs(linked_pages, entity['id'], args.number))
            seed[entity['id']] = 1

        if args.pagerank:
            try:
                pr = pagerank_power(G, tol=1e-09, personalize=seed)
                ppr_relevant_nodes = list(np.argpartition(pr, -(args.number + 1))[-(args.number + 1):][:-1])
            except ValueError:
                pass

        out_file.write(f'{qid}\t{bfs_relevant_nodes}\t{dfs_relevant_nodes}\t{ppr_relevant_nodes}\n')
