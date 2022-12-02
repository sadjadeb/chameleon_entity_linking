from tqdm import trange
import argparse
import asyncio
import aiohttp
import sys
import ir_datasets

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=125, help='batch size')
parser.add_argument('-t', '--tagme_treshold', type=float, default=0.26, help='tagme treshold')
parser.add_argument('-m', '--mode', type=str, default="tagme", help='mode: tagme or graft-net')
parser.add_argument('-d', '--dataset', type=str, default="msmarco-passage/dev/small", help='dataset name')
args = parser.parse_args()

MY_GCUBE_TOKEN = "90c6802e-a3ba-41ef-8eb9-870571f53692-843339462"
TAGME_URL = "https://tagme.d4science.org/tagme/tag"
TAGME_TRESHOLD = args.tagme_treshold
BATCH_SIZE = args.batch_size
dataset = ir_datasets.load(args.dataset)


async def get_tagme_entities(text_id, text: str, input_type):
    payload = {
        "text": text,
        "lang": "en",
        "gcube-token": MY_GCUBE_TOKEN,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(TAGME_URL, data=payload) as response:
            response = await response.json()
            annotations = [annon for annon in response['annotations'] if annon['rho'] > TAGME_TRESHOLD]
            with open(f'entities/{input_type}_entities.tsv', 'a', encoding='utf-8') as f:
                f.write(f'{text_id}\t{annotations}\n')
            return annotations


async def tagme_queries_runner():
    for i in trange(0, len(dataset.queries_iter()), BATCH_SIZE):
        tasks = []
        for query in dataset.queries_iter()[i:i + BATCH_SIZE]:
            tasks.append(get_tagme_entities(query.query_id, query.text, 'queries'))
        await asyncio.gather(*tasks)


async def tagme_docs_runner():
    for i in trange(0, len(dataset.docs_iter()), BATCH_SIZE):
        tasks = []
        for doc in dataset.docs_iter()[i:i + BATCH_SIZE]:
            tasks.append(get_tagme_entities(doc.doc_id, doc.text, 'docs'))
        await asyncio.gather(*tasks)


if args.mode == "tagme":
    print(f'Getting entities from TAGME with "score > {TAGME_TRESHOLD}"')

    if sys.platform == 'win32':
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.new_event_loop().run_until_complete(tagme_queries_runner())
    asyncio.new_event_loop().run_until_complete(tagme_docs_runner())
elif args.mode == "graft-net":
    print('Getting entities from Graft-Net')
    # TODO
