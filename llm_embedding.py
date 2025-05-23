import os
import time
import subprocess
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import boto3
import config
import torch
from transformers import AutoTokenizer, AutoModel
import argparse

timestr = time.strftime("%Y%m%d-%H%M%S")
chunksize = 10000
batch_size = 2048
max_len = 512
startline = 0

model_name = "malteos/scincl"
absloc = (
    "data/s2ag/processed/abstracts.parquet.gz"
)
saveloc = f"embeddings"

def run(batch_size=batch_size, chunksize=chunksize, model_name=model_name, absloc=absloc, saveloc=saveloc,
    startline=startline, endline=None):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.to("cuda")

    savename = f"{model_name.replace('/','-')}_{timestr}_{startline}-{endline}.jsonl"
    saveloc = os.path.join(saveloc, savename)
    if os.path.exists(saveloc):
        os.remove(saveloc)

    parquet_file = pq.ParquetFile(absloc)

    # count number of rows
    dataset = pq.ParquetDataset(absloc)
    nrows = sum(p.count_rows() for p in dataset.fragments)
    print(f"Number of rows in {absloc}: {nrows}")

    n = 0
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunksize), desc='file batches', 
                      total=(nrows // chunksize) if endline is None else ((endline - startline) // chunksize)):
        
        if n < startline:
            n += chunksize
            if endline is not None and n >= endline:
                break
            print(f'skipping {n} lines')
            continue
        
        df = batch.to_pandas().drop(columns=["title", "year"])

        embds = None
        for bix in tqdm(range(0, df.shape[0], batch_size), desc='embedding batches'):
            batch = df.iloc[bix : bix + batch_size]
            input_ids = tokenizer(
                batch["abstract"].tolist(),
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=max_len,
            )
            if torch.cuda.is_available():
                input_ids.to("cuda")
            with torch.no_grad():
                output = model(**input_ids)
                cls_embeddings = output.last_hidden_state[:, 0, :].half().cpu().numpy()
                if embds is None:
                    embds = cls_embeddings
                else:
                    embds = np.append(embds, cls_embeddings, axis=0)
        df = df.drop(columns=["abstract"])
        df = df.reset_index(drop=True).rename(columns={"corpusid": "doc_id"})
        df['embedding'] = embds.tolist()

        df.to_json(
            saveloc,
            lines=True,
            orient="records",
            index=False,
            mode="a",  # append data to csv file
        )

        n += chunksize
        if endline is not None and n >= endline:
            break
        print('processed', n, 'rows')
    
    
    # now upload to msi s3
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key'],
        endpoint_url="https://s3.msi.umn.edu",
    )

    try:
        s3.Bucket("scirepeval").upload_file(
            saveloc,
            saveloc
        )
        
        print(f"{saveloc} successfully uploaded to s3://scirepeval")
        os.remove(saveloc)
    except Exception as e:
        print(f"Error uploading to s3: {e}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embedding generation.")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size for processing")
    parser.add_argument("--chunksize", type=int, default=chunksize, help="Chunk size for reading CSV")
    parser.add_argument("--model_name", type=str, default=model_name, help="Model name for embeddings")
    parser.add_argument("--startline", type=int, default=startline, help="First line number of abstract csv to process")
    parser.add_argument("--endline", type=int, default=None, help="Final line number of abstract csv to process")
    parser.add_argument("--absloc", type=str, default=absloc, help="Location of the abstracts CSV file")
    parser.add_argument("--saveloc", type=str, default=saveloc, help="Location to save the embeddings CSV file")

    args = parser.parse_args()

    run(batch_size=args.batch_size, chunksize=args.chunksize, model_name=args.model_name, absloc=args.absloc, saveloc=args.saveloc,
        startline=args.startline, endline=args.endline)
