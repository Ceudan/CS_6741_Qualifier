import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        filepath = data_folder + "/" + split + ".nl"
        with open(filepath, 'r', encoding='utf-8') as file:
            # Read all lines from the file and store them in a list
            self.input_lines = file.readlines()
            self.input_lines = [line.strip() for line in self.input_lines]
        if(split!="test"):
            filepath = data_folder + "/" + split + ".sql"
            with open(filepath, 'r', encoding='utf-8') as file:
                # Read all lines from the file and store them in a list
                self.target_lines = file.readlines()
                self.target_lines = [line.strip() for line in self.target_lines]
        else:
            self.target_lines = None

        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.process_data(data_folder,split,tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        max_input_length = 256
        max_target_length = 256

        encoding = tokenizer(self.input_lines, padding = "longest", max_length = max_input_length, truncation=True,return_tensors="pt")
        self.input_ids, self.input_attention_mask = encoding.input_ids, encoding.attention_mask
        
        if(self.target_lines!=None):
            target_encoding = tokenizer(self.target_lines, padding = "longest", max_length = max_target_length, truncation=True,return_tensors="pt")
            self.target_ids = target_encoding.input_ids
            # replace padding token id's of the labels by -100 so it's ignored by the loss
            self.target_ids[self.target_ids == tokenizer.pad_token_id] = -100
        else:
            self.target_ids = None
        
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    pass
    # TODO
    return [], [], [], [], []

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    return [], [], []

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = 0
    train_y = 0
    dev_x = 0
    dev_y = 0
    test_x = 0 

    return train_x, train_y, dev_x, dev_y, test_x