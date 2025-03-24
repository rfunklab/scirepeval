import os
import glob
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm

def reformat_embeddings(model_name):
    embeddings_loc = 'data/s2ag/processed/d2v/d2v_64d_20e.pkl'
    ids_loc = 'data/s2ag/processed/d2v/ids_en_core_web_lg_1950-2024.csv'
    output_file = f'embeddings/Doc2Vec/{model_name}_embeddings.jsonl'
    
    # load gensim model
    model = Doc2Vec.load(embeddings_loc)
    # read ids from matched file
    ids = pd.read_csv(ids_loc, header=None, names=['doc_id'])
    ids['doc_id'] = ids['doc_id'].astype(str)
    
    ids['embedding'] = model.dv.vectors.tolist()
    ids.to_json(output_file, orient='records', lines=True, index=False)


# Example usage
model_name = 'Doc2Vec'
reformat_embeddings(model_name)