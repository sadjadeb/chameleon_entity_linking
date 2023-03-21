from tqdm import tqdm

data_path = '/home/sajadeb/msmarco/'
run_file = 'bert_sbert.trec'

docs_entities = set()
with open(data_path + 'entities/docs_entities_notnull.tsv', 'r') as f:
    print('Loading docs_entities_notnull.tsv...')
    for line in tqdm(f):
        docs_entities.add(line.split("\t")[0])

queries_entities = set()
with open(data_path + 'entities/queries_entities_notnull.tsv', 'r') as f:
    print('Loading queries_entities_notnull.tsv...')
    for line in tqdm(f):
        queries_entities.add(line.split("\t")[0])


with open(run_file, 'r') as in_file, open(run_file + '.notnull', 'w', encoding='utf-8') as out_file:
    print('Writing entity containing docs and queries...')
    lines = in_file.readlines()

    for line in tqdm(lines):
        qid, _, pid, _, _, _ = line.strip().split()
        if (qid in queries_entities) and (pid in docs_entities):
            out_file.write(line)