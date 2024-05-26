import csv

import torch
from torch.utils.data import Dataset, BatchSampler, RandomSampler, SequentialSampler
import random
from tokenizer import BertTokenizer
from typing import Any, List, NamedTuple
from datasets import SentenceClassificationDataset

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


# concat two sentences with [SEP] in the middle for sentence pair inputs
class SentencePairConcatDataset(Dataset):
    def __init__(self, dataset, args, isRegression =False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

        # Concatenate sentence IDs with [SEP] (102) token in between
        max_len = max(token_ids.size(1), token_ids2.size(1))
        token_ids_combined = torch.cat((token_ids, torch.tensor([[102]]).expand(token_ids.size(0), 1), token_ids2), dim=1)
        attention_mask_combined = torch.cat((attention_mask, torch.ones_like(token_ids_combined[:, 1:2]), attention_mask2), dim=1)
        token_type_ids_combined = torch.cat((token_type_ids, torch.zeros_like(token_ids_combined[:, 1:2]), token_type_ids2), dim=1)

        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids_combined, token_type_ids_combined, attention_mask_combined,
                labels, sent_ids)

    def collate_fn(self, all_data):
        (token_ids_combined, token_type_ids_combined, attention_mask_combined,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids_combined,
            'token_type_ids': token_type_ids_combined,
            'attention_mask': attention_mask_combined,
            'labels': labels,
            'sent_ids': sent_ids
        }

        return batched_data


class SentencePairTestConcatDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        # Tokenize both sentences separately
        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = encoding1['input_ids']
        attention_mask = encoding1['attention_mask']
        token_type_ids = encoding1['token_type_ids']

        token_ids2 = encoding2['input_ids']
        attention_mask2 = encoding2['attention_mask']
        token_type_ids2 = encoding2['token_type_ids']

        # Concatenate sentence IDs with [SEP] token in between
        sep_token_id = self.tokenizer.sep_token_id
        max_len = max(token_ids.size(1), token_ids2.size(1))
        token_ids_combined = torch.cat((token_ids, torch.tensor([[sep_token_id]]).expand(token_ids.size(0), 1), token_ids2), dim=1)
        attention_mask_combined = torch.cat((attention_mask, torch.ones_like(token_ids_combined[:, 1:2]), attention_mask2), dim=1)
        token_type_ids_combined = torch.cat((token_type_ids, torch.zeros_like(token_ids_combined[:, 1:2]), token_type_ids2), dim=1)

        return (token_ids_combined, token_type_ids_combined, attention_mask_combined,
                sent_ids)

    def collate_fn(self, all_data):
        (token_ids_combined, token_type_ids_combined, attention_mask_combined,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids_combined,
            'token_type_ids': token_type_ids_combined,
            'attention_mask': attention_mask_combined,
            'sent_ids': sent_ids
        }

        return batched_data
    
class DatasetInfo(NamedTuple):
    datasetName: str
    isPairedDataSet: bool
    isRegression: bool

DATASET_INFOS = {
    'similarity': DatasetInfo('similarity', True, True),
    'paraphrase': DatasetInfo('paraphrase', True, False),
    'sentiment': DatasetInfo('sentiment', False, False),
}

class SentenceAllConcatDataset(Dataset):
    def __init__(self, datasets, args):
        self.dataset_names = []
        self.datasets = []
        for dataset in datasets:
            dataset_name = dataset[-1][-1]
            self.dataset_names.append(dataset_name)
            dataset_info = DATASET_INFOS[dataset_name]
            if dataset_info.isPairedDataSet:
                self.datasets.append(SentencePairConcatDataset(dataset, args, dataset_info.isRegression))
            else:
                self.datasets.append(SentenceClassificationDataset(dataset, args))
        self.dataset_lens = [len(dataset) for dataset in self.datasets]
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        for i, dataset_len in enumerate(self.dataset_lens):
            if idx < dataset_len:
                return self.datasets[i][idx]
            idx -= dataset_len
        raise IndexError
        # datasetId, sampleId = idx
        # return self.datasets[datasetId][sampleId]

    def collate_fn(self, all_data):
        dataset_name = all_data[-1][-1]
        batched_data = self.datasets[self.dataset_names.index(dataset_name)].collate_fn(all_data)
        batched_data['dataset_name'] = dataset_name
        return batched_data


class BatchSamplerAllConcatDataset(BatchSampler):
    def __init__(
        self,
        datasets,
        batch_size,
        shuffle,
    ):
        sampler_type = RandomSampler if shuffle else SequentialSampler
        self.batch_samplers = [
            BatchSampler(sampler_type(dataset), batch_size, drop_last=False) for dataset in datasets
        ]
        self.ordering_list = []
        for i, batch_sampler in enumerate(self.batch_samplers):
            self.ordering_list.extend([i] * len(batch_sampler))
        random.shuffle(self.ordering_list)
        self.ordering = iter(self.ordering_list)
        self.len_prefixes = [
            sum(len(dataset) for dataset in datasets[:i])
            for i in range(len(self.batch_samplers))
        ]
        self.batch_size = batch_size
    
    def __iter__(self):
        for sample_from in self.ordering:
            result = next(iter(self.batch_samplers[sample_from]))
            yield [x + self.len_prefixes[sample_from] for x in result]
        random.shuffle(self.ordering_list)
        self.ordering = iter(self.ordering_list)

    def __len__(self):
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)