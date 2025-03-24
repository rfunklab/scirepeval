import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from expansion.defaults import default_config
from expansion.utils import train_test_split_by_index

paper_loc = 'data/s2ag/processed/papers.parquet.gz'
saveloc = 'data/s2ag/expansion/scidocs_cite'
abs_loc = 'data/s2ag/processed/abstracts.parquet.gz'
cites_loc = 'data/s2ag/processed/citations.parquet.gz'
chunk_size = 100000

def main(paper_loc=paper_loc, saveloc=saveloc, 
         abs_loc=abs_loc, cites_loc=cites_loc,
         start_year=default_config['start_year'], 
         end_year=default_config['end_year'],
         cite_lim=default_config['cite_lim'], negative_cite_lim=default_config['negative_cite_lim'],
         train_split_pct=default_config['train_split_pct']):
    
    # load the data
    df = pd.read_parquet(paper_loc, columns=['corpusid','year'])
    df = df[df['year'].between(start_year, end_year)]
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # should probably join on abstract non-null values here to make sure we're only
    # taking papers which could feasibly support embeddings
    ids = pd.read_parquet(abs_loc, columns=['corpusid'])
    df = df.merge(ids, on='corpusid', how='inner')

    citations = pd.read_parquet(cites_loc, filters=[('citingyear', '>=', start_year),
                                                    ('citingyear', '<=', end_year),
                                                    ('citedyear', '>=', start_year),
                                                    ('citedyear', '<=', end_year)],
                                                    columns=['citingcorpusid','citedcorpusid','citingyear','citedyear'])
    print()
    citations = citations.dropna(how='any')
    citations = citations.astype(int)

    # shuffle all rows so we are randomly choosing true citations
    print('aggregating true citations...')
    citations = citations.sample(frac=1, random_state=default_config['seed'])
    true_cites = citations.groupby('citingcorpusid').head(cite_lim)[['citingcorpusid','citedcorpusid']]
    true_cites['score'] = 1

    print('permuting...')
    np.random.seed(default_config['seed'])
    citations['citedcorpusid'] = citations.groupby(["citedyear", "citingyear"])["citedcorpusid"].transform(np.random.permutation)
    print('aggregating false citations...')
    citations = citations.sample(frac=1, random_state=default_config['seed'])
    false_cites = citations.groupby('citingcorpusid').head(negative_cite_lim)[['citingcorpusid','citedcorpusid']]
    false_cites['score'] = 0

    print('concatenating true and false citations...')
    df = pd.concat([true_cites, false_cites], axis=0).dropna(how='any')
    df['score'] = df['score'].astype(int)

    # rename for scirepeval
    df = df.rename(columns={'citingcorpusid':'query_id','citedcorpusid':'cand_id'})
    df = df.set_index('query_id')
    
    # get dataset splits
    print('splitting...')
    train, test = train_test_split_by_index(df, train_split_pct)
    train = train.reset_index()
    test = test.reset_index()

    # if saveloc doesn't exist, create it
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    # check if test_qrel.jsonl and train_qrel.jsonl already exist, if so, delete them
    test_file = os.path.join(saveloc, 'test_qrel.jsonl')
    train_file = os.path.join(saveloc, 'train_qrel.jsonl')
    
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(train_file):
        os.remove(train_file)
    
    print('saving out test...')
    # save out test to json in chunks
    for i in tqdm(range(0, len(test), chunk_size)):
        chunk = test.iloc[i:i + chunk_size]
        chunk.to_json(test_file, orient='records', lines=True, mode='a')

    print('saving out train...')
    # save out train to json in chunks
    for i in tqdm(range(0, len(train), chunk_size)):
        chunk = train.iloc[i:i + chunk_size]
        chunk.to_json(train_file, orient='records', lines=True, mode='a')

    # train.to_json(os.path.join(saveloc, 'train_qrel.jsonl'), orient='records', lines=True)
    # test.to_json(os.path.join(saveloc, 'test_qrel.jsonl'), orient='records', lines=True)


if __name__ == '__main__':
    main()