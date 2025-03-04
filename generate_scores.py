from scirepeval import SciRepEval
from evaluation.encoders import Model, Doc2VecModel

mode = "load_expansion"
model_names = ["SciBERT", "SPECTER", "SciNCL"]
model_checkpoints = [
    "allenai/scibert_scivocab_uncased",
    "allenai/specter2_base",
    "malteos/scincl",
]
batch_size = 8

# for i in range(len(model_names)):

#     model_name = model_names[i]
#     model_checkpoint = model_checkpoints[i]

#     print(f"Running {model_name}")

#     tasks_config = f"config/{model_name}/{mode}.jsonl"
#     results_loc = f"results/{model_name}/results.jsonl"

#     model = Model(variant="default", base_checkpoint=model_checkpoint)
#     evaluator = SciRepEval(
#         tasks_config=tasks_config, batch_size=batch_size, compute_perplexity=True
#     )
#     preds = evaluator.evaluate(model, results_loc)


# now do Doc2Vec independently
model_name = "Doc2Vec"
tasks_config = f"config/{model_name}/{mode}.jsonl"
results_loc = f"results/{model_name}/results.jsonl"
model_loc = f"20240626/{model_name}/d2v_64d_20e.pkl"
batch_size = 16384
model = Doc2VecModel(model_loc)
evaluator = SciRepEval(tasks_config=tasks_config, batch_size=batch_size)
preds = evaluator.evaluate(model, results_loc)
