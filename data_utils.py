import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class CodeDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for item in self.dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break
            inputs = self.tokenizer(item['source_code'], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
            yield inputs.input_ids.squeeze(), inputs.attention_mask.squeeze(), item['source_code']
            count += 1

def create_dataloader(dataset_name, split, tokenizer, max_length, batch_size, max_samples=None):
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    code_dataset = CodeDataset(dataset, tokenizer, max_length, max_samples)
    return DataLoader(code_dataset, batch_size=batch_size)

def process_batch(collector, batch, device):
    input_ids, attention_mask, _ = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
   
    with torch.no_grad():
        _ = collector(input_ids, attention_mask=attention_mask)
        batch_activations = collector.get_normalized_activations()
   
    activations = []
    for i in range(batch_activations.size(0)):
        valid_activations = batch_activations[i, :attention_mask[i].sum()]
        activations.append(valid_activations.float())  # Already on CPU, just ensure float32
   
    return activations