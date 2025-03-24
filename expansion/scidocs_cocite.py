import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import triu
import multiprocessing as mp
from functools import partial
import json
from tqdm import tqdm
from expansion.defaults import default_config
from expansion.utils import train_test_split_by_index

paper_loc = 'data/s2ag/processed/papers.parquet.gz'
saveloc = 'data/s2ag/expansion/scidocs_cocite'
abs_loc = 'data/s2ag/processed/abstracts.parquet.gz'
cites_loc = 'data/s2ag/processed/citations.parquet.gz'
chunk_size = 100000
NCORES = max(mp.cpu_count() - 1, 1)

def create_sparse_pairs_numpy(n, x):
    h, w = np.meshgrid(x, x)
    return coo_matrix(
        (np.ones(h.size), (h.flatten(), w.flatten())),
        shape=(n, n),
    )

def compute_counts(n, nshuffles, cfo):
    cfo = np.stack(
        cfo[["obs"] + list(range(nshuffles))]
        .apply(lambda row: np.stack(row), axis=1)
        .to_list()
    )
    f = partial(create_sparse_pairs_numpy, n)
    omatrix = (np.apply_along_axis(f, 1, cfo[:, 0, :]).sum(axis=0))
    smatrices = []
    for j in range(nshuffles):
        smatrices.append((np.apply_along_axis(f, 1, cfo[:, j + 1, :]).sum(axis=0)))
    return omatrix, smatrices


def compute_configuration_zscores(cite_info, n, ncores=NCORES):
    cite_info = cite_info.set_index("count", append=True)

    omatrix = coo_matrix((n, n))

    cfos = [
        cite_info.loc[(slice(None), [c]), :]
        for c in sorted(cite_info.index.get_level_values("count").unique())
        if c > 1
    ]
    f = partial(compute_counts, n, 0)
    if ncores == 1:
        for o, _ in tqdm(map(f, cfos), total=len(cfos)):
            omatrix += o
    else:
        with mp.Pool(ncores) as pool:
            for o, _ in tqdm(pool.imap_unordered(f, cfos, chunksize=10), total=len(cfos)):
                omatrix += o

    omatrix.sum_duplicates()

    return omatrix

def main(paper_loc=paper_loc, saveloc=saveloc, 
         abs_loc=abs_loc, cites_loc=cites_loc,
         start_year=default_config['start_year'], 
         end_year=default_config['end_year'],
         cite_lim=default_config['cite_lim'], negative_cite_lim=default_config['negative_cite_lim'],
         train_split_pct=default_config['train_split_pct'],
         ncores=NCORES):

    start_year = 1950
    end_year = 2000
    ncores = 1

    # load the data
    print('loading papers...')
    df = pd.read_parquet(paper_loc, columns=['corpusid','year'])
    df = df[df['year'].between(start_year, end_year)]
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # join on abstract non-null values here to make sure we're only
    # taking papers which could feasibly support embeddings
    print('loading abstracts...')
    ids = pd.read_parquet(abs_loc, columns=['corpusid'])
    df = df.merge(ids, on='corpusid', how='inner')

    print('loading citations...')
    full_citations = pd.read_parquet(cites_loc, filters=[('citingyear', '>=', start_year),
                                                    ('citingyear', '<=', end_year),
                                                    ('citedyear', '>=', start_year),
                                                    ('citedyear', '<=', end_year)],
                                                    columns=['citingcorpusid','citedcorpusid','citingyear','citedyear'])
    full_citations = full_citations.dropna(how='any')
    full_citations = full_citations.astype(int)
    
    # cap citations at 25 cites
    full_citations = full_citations.groupby('citingcorpusid').head(25)

    yearly_dataframes = []
    cymin = full_citations['citingyear'].min()
    cymax = full_citations['citingyear'].max()
    for year in range(cymin, cymax + 1):
        print(f'running year {year} (up to {cymax})')

        # filter citations to only those in the current year
        citations = full_citations[full_citations['citedyear'] == year]

        # filter citedcorpusids which appear at least `cite_lim` times
        citation_counts = citations['citedcorpusid'].value_counts()
        valid_citedcorpusids = citation_counts[citation_counts >= cite_lim].index
        citations = citations[citations['citedcorpusid'].isin(valid_citedcorpusids)]

        # create consecutive indices for cited papers which we'll call attributes
        cited_indices = (pd.DataFrame(citations['citedcorpusid'].sort_values().unique(), 
                                    columns=['citedcorpusid'])
                                    .reset_index()
                                    .set_index('citedcorpusid')
                                    .to_dict()['index'])
        nattributes = len(cited_indices)
        citations['cited_attribute'] = citations['citedcorpusid'].map(cited_indices)

        print('counting number of references...')
        cite_info = pd.DataFrame()
        # count number of references for each citing paper
        cite_info = pd.concat(
            [
                cite_info,
                (
                    citations.groupby("citingcorpusid")["cited_attribute"]
                    .count()
                    .to_frame("count")
                ),
            ],
            axis=1,
        )

        print('arraying observations...')
        # sort observations by index attribute
        citations = citations.sort_values(by='cited_attribute')
        # convert each reference list into an array
        cite_info = pd.concat(
            [
                cite_info,
                (
                    citations.groupby("citingcorpusid")["cited_attribute"]
                    .apply(np.array)
                    .to_frame("obs")
                ),
            ],
            axis=1,
        )

        print('computing z-scores...')
        zscores = compute_configuration_zscores(cite_info, nattributes, ncores)
        zscores.setdiag(0)

        print('unpacking co-citations into scirepeval format...')
        full = np.arange(nattributes)
        res = []
        # unpack co-citations (true) and create negative samples (false)
        for i in tqdm(range(zscores.shape[0])):
            nnz = zscores[i].nonzero()[1]
            if len(nnz) > 1:
                nzvals = np.squeeze(np.array(zscores[i,nnz].todense()[0]))
                # draw true citations according to categorical distribution
                tru = np.random.choice(nnz, size=min(cite_lim, len(nnz)), p=nzvals/nzvals.sum())
                # draw false citations from the rest of the corpus
                fls = np.random.choice(np.setdiff1d(full, nnz), size=min(negative_cite_lim, len(full) - len(nnz)), replace=False)

                # true scores are 1, false scores are 0
                tru_scores = np.ones(len(tru))
                fls_scores = np.zeros(len(fls))
                scores = np.concatenate([tru_scores, fls_scores])

                # concatenate into single numpy array
                res.append(np.stack([np.full(len(scores), i), np.concatenate([tru, fls]), scores], axis=1).astype(int))

        df = pd.DataFrame(np.concatenate(res), columns=['query_id','cand_id','score'])
        # map back to original corpus ids
        rev_cited_indices = {v: k for k, v in cited_indices.items()}
        df['cand_id'] = df['cand_id'].map(rev_cited_indices)
        df['query_id'] = df['query_id'].map(rev_cited_indices)
        
        df = df.set_index('query_id')
        yearly_dataframes.append(df)
    
    # concatenate dataframes for each year
    df =  pd.concat(yearly_dataframes)
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

if __name__ == '__main__':
    main()