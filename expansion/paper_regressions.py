import os
import pandas as pd
import numpy as np
from expansion.defaults import default_config
from expansion.utils import train_test_split

paper_loc = 'data/s2ag/processed/papers.parquet.gz'
cite_loc = 'data/s2ag/s2ag_disruption.parquet.gz'
saveloc = 'data/s2ag/expansion/paper_regressions'
abs_loc = 'data/s2ag/processed/abstracts.parquet.gz'

def main(paper_loc=paper_loc, cite_loc=cite_loc, 
         saveloc=saveloc, abs_loc=abs_loc, 
         horizon=default_config['horizon'],
         start_year=default_config['start_year'], 
         end_year=default_config['end_year'],
         train_split_pct=default_config['train_split_pct']):
    
    # load the data
    print('loading year data...')
    df = pd.read_parquet(paper_loc, columns=['corpusid','year'])
    df = df[df['year'].between(start_year, end_year)]
    df = df.dropna(subset=['year'])
    # scale year column to 0-1
    df['year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())

    # load citation data
    print('loading citation data...')
    citations = pd.read_parquet(cite_loc, columns=[f'i_{horizon}'])
    citations[f'i_{horizon}'] = np.log(1 + citations[f'i_{horizon}'])
    df = df.merge(citations, left_on='corpusid', right_index=True, how='inner')
    
    # should probably join on abstract non-null values here to make sure we're only
    # taking papers which could feasibly support embeddings
    print('loading abstract data...')
    abstracts = pd.read_parquet(abs_loc, columns=['corpusid','abstract'])
    df = df.merge(abstracts, on='corpusid', how='inner')

    # get dataset splits
    train, test = train_test_split(df, train_split_pct)

    # if saveloc doesn't exist, create it
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)

    train.to_json(os.path.join(saveloc, 'train.jsonl'), orient='records', index=False, lines=True)
    test.to_json(os.path.join(saveloc, 'test.jsonl'), orient='records', index=False, lines=True)



if __name__ == '__main__':
    main()