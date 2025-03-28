import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class JsonlDataset(Dataset):
    def __init__(self, file_path, label_key):
        self.data = []
        self.label_key = label_key
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        abstract = item['abstract']
        label = item[self.label_key]
        return abstract, label
    
class Regressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(Regressor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # standard scale inputs
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True)
        x -= m
        x /= s
        return self.linear(x)
    

def collate_fn(batch):
    abstracts, labels = zip(*batch)
    return list(abstracts), torch.tensor(labels, dtype=torch.float32)

def embed_abstracts(abstracts, model, tokenizer, device):
    inputs = tokenizer(abstracts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

def train_and_evaluate(batch_size, train_file, test_file, label_key, save_loc, train=True):
    result_save_file = os.path.join(save_loc, f"{label_key}_outputs.jsonl")
    model_save_file = os.path.join(save_loc, f"{label_key}_model.pth")
    loss_save_file = os.path.join(save_loc, f"{label_key}_losses.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    if os.path.exists(model_save_file):
        print(f"Model already exists at {model_save_file}. Loading...")
        regressor = torch.load(model_save_file).to(device)
    else:
        regressor = Regressor(model.config.hidden_size).to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(regressor.parameters(), lr=0.01, weight_decay=0.0001)
    
    if train:
        # Training
        train_dataset = JsonlDataset(train_file, label_key)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        
        losses = np.array([])
        if os.path.exists(loss_save_file):
            losses = np.loadtxt(loss_save_file, delimiter=',')

        regressor.train()
        pbar = tqdm(train_loader)
        for abstracts, labels in pbar:
            embeddings = embed_abstracts(abstracts, model, tokenizer, device).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = regressor(embeddings).squeeze()
            loss = criterion(outputs, labels)
            losses = np.append(losses, loss.item())
            pbar.set_description(f"Training (loss: {loss.item():.4f})")
            loss.backward()
            optimizer.step()
        
        # Save the trained model
        torch.save(regressor.state_dict(), model_save_file)
        np.savetxt(loss_save_file, losses, delimiter=',')
        print(f"Model saved to {model_save_file}")
        print(f"Losses saved to {loss_save_file}")

    # Evaluation
    test_dataset = JsonlDataset(test_file, label_key)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    regressor.eval()
    all_preds = []
    with torch.no_grad():
        for abstracts, labels in tqdm(test_loader, desc="Test"):
            embeddings = embed_abstracts(abstracts, model, tokenizer, device).to(device)
            
            outputs = regressor(embeddings).squeeze()
            all_preds.extend(outputs.cpu().numpy())

    with open(result_save_file, 'w') as f:
        for i in tqdm(range(len(all_preds)), desc="Saving predictions"):
            pred = all_preds[i]
            f.write(json.dumps({'index':i, 'prediction': pred}) + '\n')

if __name__ == "__main__":
    batch_size = 1024
    train_file = 'data/s2ag/expansion/paper_regressions/train.jsonl'
    test_file = 'data/s2ag/expansion/paper_regressions/test.jsonl'
    label_key = 'year'  # or 'i_5'
    save_loc = f'data/s2ag/expansion/paper_regressions'
    train = True
    train_and_evaluate(batch_size, train_file, test_file, label_key, save_loc, train=train)