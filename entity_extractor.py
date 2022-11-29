from tqdm import tqdm
import requests
import argparse
import asyncio
import aiohttp
import time
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default="Tevatron/msmarco-passage-corpus", help='dataset name')
parser.add_argument('-o', '--output', type=str, default="entities.txt", help='output file name and path')
parser.add_argument('-m', '--mode', type=str, default="async", help='mode: sync or async')
parser.add_argument('--min_rho', type=float, default=0.26, help='minimum rho value for annotations')
args = parser.parse_args()

MY_GCUBE_TOKEN = "90c6802e-a3ba-41ef-8eb9-870571f53692-843339462"
TAGME_URL = "https://tagme.d4science.org/tagme/tag"
TAGME_MIN_RHO = args.min_rho
dataset = load_dataset(args.dataset, split="dev")


def get_entities(text: str):
    payload = {
        "text": text,
        "lang": "en",
        "gcube-token": MY_GCUBE_TOKEN,
    }
    response = requests.post(TAGME_URL, data=payload).json()
    annotations = [annon for annon in response['annotations'] if annon['rho'] > TAGME_MIN_RHO]
    return annotations


async def get_entities_async(text: str):
    payload = {
        "text": text,
        "lang": "en",
        "gcube-token": MY_GCUBE_TOKEN,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(TAGME_URL, data=payload) as response:
            response = await response.json()
            annotations = [annon for annon in response['annotations'] if annon['rho'] > TAGME_MIN_RHO]
            return annotations


async def async_runner():
    tasks = []
    for doc in dataset:
        tasks.append(asyncio.create_task(get_entities_async(doc)))
    entities = await asyncio.gather(*tasks)
    return entities


if args.mode == "async":
    start = time.time()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    entities = asyncio.new_event_loop().run_until_complete(async_runner())
    print(f"Time taken: {time.time() - start}")
elif args.mode == "sync":
    entities = []
    for doc in tqdm(dataset):
        annotations = get_entities(doc)
        entities.append(annotations)
