import os
import numpy as np
import pandas as pd
from expansion.defaults import default_config
from expansion.utils import train_test_split

paper_loc = 'data/s2ag/processed/papers.parquet.gz'
cite_loc = 'data/s2ag/s2ag_disruption.parquet.gz'
saveloc = 'data/s2ag/expansion/citation_count'
abs_loc = 'data/s2ag/processed/abstracts.parquet.gz'

def main(paper_loc=paper_loc, cite_loc=cite_loc, 
         saveloc=saveloc, abs_loc=abs_loc, 
         horizon=default_config['horizon'],
         start_year=default_config['start_year'], 
         end_year=default_config['end_year'],
         train_split_pct=default_config['train_split_pct']):
    
    # load the data
    df = pd.read_parquet(paper_loc, columns=['corpusid','year'])
    df = df[df['year'].between(start_year, end_year-horizon)]
    df = df.dropna(subset=['year']).drop(columns=['year'])

    # should probably join on abstract non-null values here to make sure we're only
    # taking papers which could feasibly support embeddings
    ids = pd.read_parquet(abs_loc, columns=['corpusid'])
    df = df.merge(ids, on='corpusid', how='inner')
    
    # load citation data
    citations = pd.read_parquet(cite_loc, columns=[f'i_{horizon}'])
    df = df.merge(citations, left_on='corpusid', right_index=True, how='inner')

    # rename for scirepeval
    df = df.rename(columns={'corpusid':'paper_id', f'i_{horizon}': 'label'})

    # drop 0-citations and convert to log scale
    df = df[df['label']  > 0]
    df['label'] = np.log(df['label'])
    
    # get dataset splits
    train, test = train_test_split(df, train_split_pct)

    # if saveloc doesn't exist, create it
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)

    train.to_csv(os.path.join(saveloc, 'train.csv'), index=False)
    test.to_csv(os.path.join(saveloc, 'test.csv'), index=False)



if __name__ == '__main__':
    main()