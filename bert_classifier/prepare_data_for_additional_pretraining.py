from datasets import load_multitask_data
import random
import csv
import argparse

def seed_everything(seed=11711):
    random.seed(seed)

def prepare_data(args, split):
    if split == 'train':
        sst_train_data, _, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
        output_path = args.output_train
    else:
        sst_train_data, _, para_train_data, sts_train_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='dev') 
        output_path = args.output_dev
    sst_train_data = [x[0] for x in sst_train_data]
    para_train_data = [x[0] for x in para_train_data] + [x[1] for x in para_train_data]
    sts_train_data = [x[0] for x in sts_train_data] + [x[1] for x in sts_train_data]
    train_data = sst_train_data + para_train_data + sts_train_data
    random.shuffle(train_data)
    with open(output_path, 'w') as file:
        for datapoint in train_data:
            # Write each sentence followed by a newline
            file.write(datapoint + "\n")


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

    parser.add_argument("--output_train", type=str, default="additional_pretraining_data/train.txt")
    parser.add_argument("--output_dev", type=str, default="additional_pretraining_data/dev.txt")

    parser.add_argument("--seed", type=int, default=11711)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    prepare_data(args, 'train')
    prepare_data(args, 'dev')