from tqdm import tqdm

results = {}
pruned_run_file = '/home/sajadeb/chameleon_entity_linking/SBERT/output/cross-encoder_luke-base_with-entities-entities-latest/Run.txt'
entire_run_file = '/home/sajadeb/chameleon_entity_linking/SBERT/output/cross-encoder_luke-base_with-entities-entities-latest/Run_full.txt'
with open(pruned_run_file, 'r') as pruned, open(entire_run_file, 'r') as entire:
    for line in tqdm(pruned):
        data = line.strip().split()
        if data[0] in results:
            results[data[0]].append({'pid': data[2], 'score': data[4]})
        else:
            results[data[0]] = [{'pid': data[2], 'score': data[4]}]

    for line in tqdm(entire):
        data = line.strip().split()
        if data[0] in results:
            if data[2] not in [hit['pid'] for hit in results[data[0]]]:
                results[data[0]].append({'pid': data[2], 'score': data[4]})
        else:
            results[data[0]] = [{'pid': data[2], 'score': data[4]}]

with open('Run_merged.txt', 'w', encoding='utf-8') as out:
    for qid, passages in results.items():
        passages = sorted(results[qid], key=lambda x: x['score'], reverse=True)
        rank = 1
        for hit in passages:
            out.write(str(qid) + ' Q0 ' + hit['pid'] + ' ' + str(rank) + ' ' + str(hit['score']) + ' ' + 'merged_with_all' + '\n')
            rank = rank + 1
