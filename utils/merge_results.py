from tqdm import tqdm

results = {}
with open('Run.txt', 'r') as f, open('Run_null.txt', 'r') as n:
    for line in tqdm(f):
        data = line.strip().split()
        if data[0] in results:
            results[data[0]].append({'pid': data[2], 'score': data[4]})
        else:
            results[data[0]] = [{'pid': data[2], 'score': data[4]}]

    for line in tqdm(n):
        data = line.strip().split()
        if data[0] in results:
            results[data[0]].append({'pid': data[2], 'score': data[4]})
        else:
            results[data[0]] = [{'pid': data[2], 'score': data[4]}]

with open('Run_merged.txt', 'w', encoding='utf-8') as out:
    for qid, passages in results.items():
        passages = sorted(results[qid], key=lambda x: x['score'], reverse=True)
        rank = 1
        for hit in passages:
            out.write(str(qid) + ' Q0 ' + hit['pid'] + ' ' + str(rank) + ' ' + str(hit['score']) + ' ' + 'OA_on_all' + '\n')
            rank = rank + 1
