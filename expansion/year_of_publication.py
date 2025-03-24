import os
import pandas as pd
from expansion.defaults import default_config
from expansion.utils import train_test_split

paper_loc = 'data/s2ag/processed/papers.parquet.gz'
saveloc = 'data/s2ag/expansion/year_of_publication'
abs_loc = 'data/s2ag/processed/abstracts.parquet.gz'

def main(paper_loc=paper_loc, saveloc=saveloc, abs_loc=abs_loc, 
         start_year=default_config['start_year'], 
         end_year=default_config['end_year'],
         train_split_pct=default_config['train_split_pct']):
    
    # load the data
    df = pd.read_parquet(paper_loc, columns=['corpusid','year'])
    df = df[df['year'].between(start_year, end_year)]
    df = df.dropna(subset=['year'])
    # scale year column to 0-1
    df['year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    # should probably join on abstract non-null values here to make sure we're only
    # taking papers which could feasibly support embeddings
    ids = pd.read_parquet(abs_loc, columns=['corpusid'])
    df = df.merge(ids, on='corpusid', how='inner')

    # rename for scirepeval
    df = df.rename(columns={'corpusid':'paper_id', f'year': 'label'})

    # get dataset splits
    train, test = train_test_split(df, train_split_pct)

    # if saveloc doesn't exist, create it
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)

    train.to_csv(os.path.join(saveloc, 'train.csv'), index=False)
    test.to_csv(os.path.join(saveloc, 'test.csv'), index=False)



if __name__ == '__main__':
    main()