import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, SentenceAllDataset, BatchSamplerAllDataset,\
    load_multitask_data, load_multitask_test_data
from cross_dataset import SentenceAllConcatDataset, BatchSamplerAllConcatDataset, SentencePairConcatDataset, SentencePairTestConcatDataset

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask, model_eval_multitask_cross,test_model_multitask_cross


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        if config.load_model_state_dict_from_model_path is not None:
            self.bert = BertModel.from_pretrained(config.load_model_state_dict_from_model_path)
            print('loading addtional pretained model')
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
            elif config.option == 'finetune_after_additional_pretraining':
                param.requires_grad = True
        ### TODO
        # self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.linear = torch.nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        # sentiment layers
        self.sentiment_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        # from febie: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/FebieJaneLinJackPLe.pdf
        self.paraphrase_linear = nn.Linear(config.hidden_size, 1)
        self.paraphrase_dropout = nn.Dropout(config.task_dropout_prob)
        self.similarity_linear = nn.Linear(config.hidden_size, 1)
        self.similarity_dropout = nn.Dropout(config.task_dropout_prob)



    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        return pooled_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        output = self.forward(input_ids, attention_mask)
        # output = self.sentiment_dropout(output) # dropout added
        sentiment_logits = self.sentiment_linear(output)
        return sentiment_logits


    def predict_paraphrase(self,
                           b_ids_combined, b_attention_mask_combined):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        emb = self.forward(b_ids_combined, b_attention_mask_combined)
        emb = self.paraphrase_dropout(emb)
        output = self.paraphrase_linear(emb)
        logits = output.view(-1)
        return logits


    def predict_similarity(self,
                          b_ids_combined, b_attention_mask_combined):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        emb = self.forward(b_ids_combined, b_attention_mask_combined)
        emb = self.similarity_dropout(emb)
        output = self.similarity_linear(emb)
        logits = output.view(-1)
        logits = torch.sigmoid(logits) * 5.0
        return logits




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def single_batch_train_sts(batch, model: MultitaskBERT, optimizer, device, debug=False):
    b_ids_combined, b_attention_mask_combined, b_labels = (
        batch['token_ids'],
        batch['attention_mask'],
        batch['labels'],
    )
    if debug:
        print(b_labels[:5])
    b_ids_combined = b_ids_combined.to(device)
    b_attention_mask_combined = b_attention_mask_combined.to(device)
    b_labels = b_labels.to(device)
    optimizer.zero_grad()
    predictions = model.predict_similarity(b_ids_combined, b_attention_mask_combined)

    # C = sum(np.exp(np.arange(1, 5))) # TODO fix bug here
    loss = F.mse_loss(predictions, b_labels.view(-1).float(), reduction='sum') / args.batch_size

    if debug:
        print("sts", predictions[:5], b_labels[:5], loss)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    return train_loss


def single_epoch_train_sts(sts_train_dataloader, epoch, model, optimizer, device, debug=False):
    train_loss = 0
    num_batches = 0
    for batch in tqdm(sts_train_dataloader, desc=f'train-sts-{epoch}', disable=TQDM_DISABLE):
        train_loss += single_batch_train_sts(batch, model, optimizer, device, debug)
        num_batches += 1
        if debug and num_batches >= 5:
            break
    train_loss = train_loss / num_batches
    return train_loss


def single_epoch_train_para(para_train_dataloader, epoch, model, optimizer, device, grad_scaling_factor_for_para, debug=False):
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-para-{epoch}', disable=TQDM_DISABLE):
        train_loss += single_batch_train_para(batch, model, optimizer, device, grad_scaling_factor_for_para, debug)
        num_batches += 1
        if debug and num_batches >= 5:
            break

    train_loss = train_loss / num_batches
    return train_loss

def single_batch_train_para(batch, model: MultitaskBERT, optimizer, device, grad_scaling_factor_for_para, debug=False):
    b_ids_combined, b_mask_combined, b_labels = (
        batch['token_ids'],
        batch['attention_mask'],
        batch['labels']
    )
    if debug:
        print(b_labels[:5])
    b_ids_combined = b_ids_combined.to(device)
    b_mask_combined = b_mask_combined.to(device)
    b_labels = b_labels.to(device)
    optimizer.zero_grad()
    logits = model.predict_paraphrase(b_ids_combined, b_mask_combined)

    # multi_negatives_ranking_loss = torch.sum((b_labels == 1) * multi_negatives_ranking_loss) / torch.sum(b_labels == 1)
    loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='sum') / args.batch_size
    if debug:
        print("para", logits[:5], b_labels[:5])
        print("para loss", loss)
    loss *= grad_scaling_factor_for_para
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    return train_loss

def single_batch_train_sst(batch, model: MultitaskBERT, optimizer, device, debug=False):
        b_ids, b_mask, b_labels = (batch['token_ids'],
                                    batch['attention_mask'], batch['labels'])
        if debug:
            print(b_ids[:5], b_labels[:5])
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        logits = model.predict_sentiment(b_ids, b_mask)
        if debug:
            print("sst", logits[:5, :], b_labels[:5])
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        return train_loss

def single_epoch_train_sst(sst_train_dataloader, epoch, model: MultitaskBERT, optimizer, device, debug = False):
    train_loss = 0
    num_batches = 0
    for batch in tqdm(sst_train_dataloader, desc=f'train-sst-{epoch}', disable=TQDM_DISABLE):
        train_loss += single_batch_train_sst(batch, model, optimizer, device, debug)
        num_batches += 1
        if debug and num_batches >= 5:
            break
    train_loss = train_loss / num_batches
    return train_loss


def single_epoch_train_all(
        train_dataloader,
        epoch,
        model,
        optimizer,
        device,
        grad_scaling_factor_for_para,
        debug=False,
    ):
    sst_train_loss, num_sst_batches = 0, 0
    para_train_loss, num_para_batches = 0, 0
    sts_train_loss, num_sts_batches = 0, 0
    
    for batch in tqdm(train_dataloader, desc=f'train-all-{epoch}', disable=TQDM_DISABLE):
        if batch['dataset_name'] == 'sentiment':
            sst_train_loss += single_batch_train_sst(batch, model, optimizer, device, debug)
            num_sst_batches += 1
        elif batch['dataset_name'] == 'paraphrase':
            para_train_loss += single_batch_train_para(batch, model, optimizer, device, grad_scaling_factor_for_para, debug)
            num_para_batches += 1
        elif batch['dataset_name'] == 'similarity':
            sts_train_loss += single_batch_train_sts(batch, model, optimizer, device, debug)
            num_sts_batches += 1

        if debug and num_sst_batches + num_para_batches + num_sts_batches >= 5:
            break

    def get_loss(loss, num_batches, desc):
        print(f"Loss of {loss} for {desc} after {num_batches} batches")
        if num_batches == 0:
            return -1
        return loss / num_batches
    
    sst_train_loss = get_loss(sst_train_loss, num_sst_batches, 'sst')
    para_train_loss = get_loss(para_train_loss, num_para_batches, 'para')
    sts_train_loss = get_loss(sts_train_loss, num_sts_batches, 'sts')
    return sst_train_loss, para_train_loss, sts_train_loss

## Currently only trains on sst dataset
def train_multitask(args):
    # device = torch.device(f'cuda:{args.cuda}') if args.use_gpu else torch.device('cpu')
    device = torch.device(f'cuda') if args.use_gpu else torch.device('cpu')

    # create a dataframe for training outcomes for each epoch (requires the pandas and openpyxl packages)
    df = pd.DataFrame(columns = ['epoch', 'train_acc_sst', 'train_acc_para', 'train_acc_sts', 'train_acc', \
                             'dev_acc_sst', 'dev_acc_para', 'dev_acc_sts', 'dev_acc'])
    
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    
    # even batching data
    if args.even_batching:
        all_train_datasets = [sst_train_data, para_train_data, sts_train_data]
        if args.cross_encode:
            train_data = SentenceAllConcatDataset(all_train_datasets, args)
            train_batch_sampler = BatchSamplerAllConcatDataset(train_data.datasets, args.batch_size, shuffle = True)
        else: 
            print('use --cross_encode!!')
            exit(0)

        train_dataloader = DataLoader(train_data, collate_fn = train_data.collate_fn, batch_sampler = train_batch_sampler)

        # test the dataloader
        '''
        for epoch in range(args.epochs):
            print("Epoch", epoch)
            batches_by_dataset = {}
            for batch in tqdm(train_dataloader, desc=f'testing-new-dataloader', disable=TQDM_DISABLE):
                dataset_name = batch['dataset_name']
                batches_by_dataset[dataset_name] = batches_by_dataset.get(dataset_name, 0) + 1
            print("Batches by dataset is", batches_by_dataset)
        '''
    else: # separate data
        # sst data
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sst_train_data.collate_fn)
        
        # para data
        para_train_data = SentencePairConcatDataset(para_train_data, args)
        para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)

        # sts data
        sts_train_data = SentencePairConcatDataset(sts_train_data, args)
        sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    


    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                            collate_fn=sst_dev_data.collate_fn)
    para_dev_data = SentencePairConcatDataset(para_dev_data, args)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                            collate_fn=para_dev_data.collate_fn)
    sts_dev_data = SentencePairConcatDataset(sts_dev_data, args, isRegression = True)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'task_embedding_size': 128,
              'task_dropout_prob': 0.3,
              'load_model_state_dict_from_model_path': args.load_model_state_dict_from_model_path if args.option == 'finetune_after_additional_pretraining' else None,
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    # if args.option == 'finetune_after_additional_pretraining' and args.load_model_state_dict_from_model_path is not None:
    #     print('Using model that pretrained on addtional sentences in training data')
    #     print(f"Loading model from {args.load_model_state_dict_from_model_path}")
    #     saved = torch.load(args.load_model_state_dict_from_model_path)
    #     model.load_state_dict(saved['model'])
    #     print("Loaded model state")
    #     print("Old config was", saved['model_config'])
    #     print("New config is", model.config)


    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    debug = args.debug
    print('start training...')
    # if debug and args.load_model_state_dict_from_model_path is not None:
    #     print('trying to save pretrained model')
    #     save_model(model, optimizer, args, config, args.filepath)
    #     print('successfully')

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        if args.even_batching: # even batching training
            losses = single_epoch_train_all(
                train_dataloader,
                epoch,
                model,
                optimizer,
                device,
                args.grad_scaling_factor_for_para,
                debug = debug
            )
            sst_train_loss, para_train_loss, sts_train_loss= losses

        else: # sequential full training
            sst_train_loss = single_epoch_train_sst(sst_train_dataloader, epoch, model, optimizer, device, debug)

            para_train_loss = single_epoch_train_para(para_train_dataloader, epoch, model, optimizer, device, args.grad_scaling_factor_for_para, debug)

            sts_train_loss = single_epoch_train_sts(sts_train_dataloader, epoch, model, optimizer, device, debug)

        print(f"Epoch {epoch}: train loss :: sst:: {sst_train_loss :.3f}, para:: {para_train_loss :.3f}, sts:: {sts_train_loss :.3f}")

        print(f"Epoch {epoch}: dev data stats")
        # train_acc_para, _, _, train_acc_sst, _, _, train_acc_sts, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        # dev_acc_para, _, _, dev_acc_sst, _, _, dev_acc_sts, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        dev_acc_para, _, _, dev_acc_sst, _, _, dev_acc_sts, *_ = model_eval_multitask_cross(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        # train_acc = np.average([train_acc_sst, train_acc_para, train_acc_sts])
        dev_acc = np.average([dev_acc_sst, dev_acc_para, dev_acc_sts])

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath) # edited: changed optimizer to optim

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        data = {'epoch': epoch,
        'train_acc_sst': 'skip to save time ',
        'train_acc_para': 'skip to save time ',
        'train_acc_sts': 'skip to save time ',
        'train_acc': 'skip to save time ',
        'dev_acc_sst': dev_acc_sst,
        'dev_acc_para': dev_acc_para,
        'dev_acc_sts': dev_acc_sts,
        'dev_acc': dev_acc}

        new_df = pd.DataFrame(data, index=[0])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(f'./results/pretrain-even_batch-cross-alldrop-batch32-multitask_results.csv', index = False)

    df.to_csv(f'./results/pretrain-even_batch-cross-alldrop-batch32-multitask_results.csv', index = False)

       



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        # device = torch.device(f'cuda:{args.cuda}') if args.cuda else torch.device('cpu')

        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask_cross(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated, finetune_after_additional_pretraining: first pretrain on training sentence, then fintune',
                        choices=('pretrain', 'finetune', 'finetune_after_additional_pretraining'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    # additional parameters
    parser.add_argument("--cuda", type=int, default=1,required=False)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--even_batching", action='store_true')
    parser.add_argument("--grad_scaling_factor_for_para", type=float, default=1.0)
    parser.add_argument(
        "--load_model_state_dict_from_model_path",
        type=str,
        help='Only loads model state dict; does NOT load optimizer state, config, etc.; only for finetune after pretrain'
    )
    parser.add_argument("--cross_encode", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.option == 'finetune_after_additional_pretraining':
        assert(args.load_model_state_dict_from_model_path is not None)
    args.filepath = f'./models/ram-pretrain_even_batch-cross-alldrop-batch32-{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)

# 1. change arg.filepath to differentiate different trained models
# 2. change the last line in train_multitask(), save the training result to a .csv