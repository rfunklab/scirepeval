from typing import Dict, List, Union

import os
import math
import json
from collections import defaultdict, Counter
import pathlib
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
)

from evaluation.encoders import Model

def computePerplexity(sents, model_path, batch_size=50, num_digits=None,
                      compute_sent_perp=False):
    '''
    Compute perplexity for a list of tokenized sentences
    Varies batch size by the length of the sentences

    @param sents (list[list[int]]) - list of BERT tokenizer encoded sentences
    @param model_path (str) - Path to finetuned BERT model
    @param batch_size (int) - Batch size for forward pass through the model
        Defaults to 65 - Appropriate for 2080ti with max sequence length of 100 tokens
    @param num_digits (int) - Number of digits to round by
        If None, no rounding
        Note: inefficient implementation, better to vectorize
    @param compute_sent_perp (bool) - Whether to return sentence-level perplexity
        If False, return word-level perplexity

    @return perps (list[list[float]]) - List of the word-level perplexities from MLM task
        Note: Returned at the word-level, rather than the sentence level, to enable
              word-level analyses and adjustments 

    Dependencies:
        computePerplexityHelper
    '''

    # Load model on gpu in evalulation model if gpu is available
    # Note: Not sure eval is needed for BERT, but won't hurt anything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(model_path).to(device).eval()

    # Load loss function assuming BERT tokenizer pad token idx
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    # Compute and return perplexity
    perps = computePerplexityHelper(sents, batch_size, model, loss_fct)

    if compute_sent_perp:  # Reduce to sentence perplexity
        return [sentencePerplexity(sent) for sent in perps]
    elif num_digits is not None: # Round items in list while retaining list structure
        raise ValueError
        #return perps = [[round(perp, num_digits) for perp in sent] for sent in perps]
    else:
        return perps

def computePerplexityHelper(encoded_sents, batch_size, model, loss_func):
    '''
    Compute perplexity for a batch of sentences by minibatching
    @param encoded_sents (list[list[int]]) - list of BERT tokenizer encoded sentences
    @param batch_size (int) - size to batch by
    @param model - BERT model for forward pass
    @param loss_func - torch loss function

    @return perp (list[list[float]]) - List of the word-level perplexities
    Called in computePerplexity
    '''

    # Load tensors on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perps = []  # Array to hold sequence MLM perplexities
    encoded_sents = list(encoded_sents)  # Cast as list to remove potential indexing

    tqdm_desc = "Batch Size: {}".format(batch_size)

    n_minibatches = math.ceil(len(encoded_sents) / batch_size)
    for batch in tqdm(np.array_split(encoded_sents, n_minibatches), desc=tqdm_desc):

        # Get sequence lengths
        lens = [len(x) for x in batch]

        # Batch the tokens and forward pass through the langauge model
        batch_tokens = [torch.tensor(s, requires_grad=False, device=device) for s in batch]
        batch_tokens = pad_sequence(batch_tokens, batch_first=True, padding_value=0)
        perp = perplexity(model, batch_tokens, loss_func)

        for i, l in enumerate(lens):
            # Append perplexity, excluding pad tokens 
            perps.append(perp[i, :l])

    torch.cuda.empty_cache() 
    return perps

def perplexity(model, batch_tokens, loss_func):
    pred = model(batch_tokens)[0].detach()

    # Collapse batch compute loss
    pred = pred.view(batch_tokens.shape[0] * batch_tokens.shape[1], -1)
    batch_reshaped = batch_tokens.view(-1)
    loss = loss_func(pred, batch_reshaped).view(batch_tokens.shape[0], -1)

    # Exponentiate to obtain perplexity
    perp = torch.exp(loss).cpu().numpy()

    return perp

def sentencePerplexity(perp, trim_special_tokens=True):
    '''
    Compute sentence-level perplexity given a list of word-level perplexities
    @param perp (list[float]) - list of word-level perplexities
    @param trim_special_tokens (bool) - whether to crop [CLS] and [SEP] tokens from start and end of list

    @return float - sentence-level-perplexity
    '''

    if trim_special_tokens:
        vals = perp[1:-1]

    # Take word-level product and normalize by # of words
    return np.prod(np.float64(vals)) ** (1.0 / len(vals))

def computeTokenLikelihoods(token_lists, token_perp_list, as_dataframe=True):
    '''
    Compute relative token likelihoods for a list of sequences 
    @param token_lists (list[list[int]]) - list of list of token ids
    @param token_perp_list (list[list[float]]) - list of list of token perplexities from computePerplexity
    @param as_dataframe (bool) - whether to return the average token perplexities as dataframe (otherwise defaultdict)

    @return tokenDict - Dictionary of token likelihoods indexed by token_id with value mean token perplexity
    '''

    token_dict = defaultdict(list)

    # Append token perplexities to token_dict
    for (seq_tokens, seq_perps) in tqdm(zip(token_lists, token_perp_list), total=len(token_lists), desc="Processing Sents:"):
        for token, perp in zip(seq_tokens, seq_perps):
            token_dict[token].append(perp)  # Exponentiate to obtain perplexity

    # Consolidate by taking mean perplexity
    for token_id, token_perps in tqdm(token_dict.items(), total=len(token_dict), desc="Computing Avg Perplexity:"):
        token_dict[token_id] = np.mean(np.log(token_perps))

    if as_dataframe:
        return pd.DataFrame.from_dict(token_dict, orient='index').rename(columns={0 : 'avg_perplexity'})

    return token_dict

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    def __init__(self, datasets, models: Union[Model, List[Model]]):
        self.datasets = datasets
        self.models = models

    def generate_embeddings(self, save_path: str = None) -> Dict[str, np.ndarray]:
        results = dict()
        try:
            for dataset, model in zip(self.datasets, self.models):
                for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // dataset.batch_size):
                    emb = model(batch, batch_ids)
                    for paper_id, embedding in zip(batch_ids, emb.unbind()):
                        if type(paper_id) == tuple:
                            paper_id = paper_id[0]
                        if paper_id not in results:
                            results[paper_id] = embedding.detach().cpu().numpy()
                        else:
                            results[paper_id] += embedding.detach().cpu().numpy()
                    del batch
                    del emb
            results = {k: v/len(self.models) for k, v in results.items()}
        except Exception as e:
            print(e)
        finally:
            if save_path:
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as fout:
                    for k, v in results.items():
                        fout.write(json.dumps({"doc_id": k, "embedding": v.tolist()}) + '\n')
        logger.info(f"Generated {len(results)} embeddings")
        return results

    @staticmethod
    def load_embeddings_from_jsonl(embeddings_path: str) -> Dict[str, np.ndarray]:
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in tqdm(f, desc=f'reading embeddings from {embeddings_path}'):
                line_json = json.loads(line)
                embeddings[line_json['doc_id']] = np.array(line_json['embedding'], dtype=np.float16)
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings

    def compute_perplexity(self, save_path: str = None) -> Dict[str, np.ndarray]:
        results = dict()
        try:
            for dataset, model in zip(self.datasets, self.models):
                for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // dataset.batch_size):
                    perp = model.perplexity(batch, batch_ids)
                    for paper_id, perplexity in zip(batch_ids, perp.unbind()):
                        if type(paper_id) == tuple:
                            paper_id = paper_id[0]
                        if paper_id not in results:
                            results[paper_id] = perplexity.detach().cpu().numpy()
                        else:
                            results[paper_id] += perplexity.detach().cpu().numpy()
                    del batch
                    del emb
            results = {k: v/len(self.models) for k, v in results.items()}
        except Exception as e:
            print(e)
        finally:
            if save_path:
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as fout:
                    for k, v in results.items():
                        fout.write(json.dumps({"doc_id": k, "perplexity": v.tolist()}) + '\n')
        logger.info(f"Generated {len(results)} perplexity scores")
        return results