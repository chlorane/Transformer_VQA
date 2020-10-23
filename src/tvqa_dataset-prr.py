import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import load_pickle, save_pickle, load_json, files_exist
import os
import csv
import io
import string
import pickle
import numpy.linalg as LA
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertForMultipleChoice, BertTokenizer, RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer, XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer, AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer, get_linear_schedule_with_warmup, BertModel)
from config import BaseOptions as opt

from config import BaseOptions as opt

class data(object):
    def __init__(self,vcpt,context_sentence,a0,a1,a2,a3,a4,subtitle):
        self.vcpt=vcpt
        self.context_sentence=context_sentence
        self.endings=[a0,a1,a2,a3,a4]
        self.subtitle=subtitle
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l=["vcpt: {}".format(self.vcpt),"context_sentence: {}".format(self.context_sentence),"ending_0: {}".format(self.endings[0]),"ending_1: {}".format(self.endings[1]),"ending_2: {}".format(self.endings[2]),"ending_3: {}".format(self.endings[3]),"ending_4: {}".format(self.endings[4]),"subtitle: {}".format(self.subtitle)]
        return ", ".join(l)

def read_examples(R_vcpt,R_q,R_sub,R_a0,R_a1,R_a2,R_a3,R_a4):
    lines=[]
    line=["0"]*len(R_q)
    for i in range(0,len(R_q)):
        line[i]=["0"]*8
        line[i][0]=R_vcpt[i]
        line[i][1]=R_q[i]
        line[i][2]=R_sub[i]
        line[i][3]=R_a0[i]
        line[i][4]=R_a1[i]
        line[i][5]=R_a2[i]
        line[i][6]=R_a3[i]
        line[i][7]=R_a4[i]
        lines.append(line[i])
        examples=[data(vcpt=line[i][0],context_sentence=line[i][1],subtitle=line[i][2],a0=line[i][3],a1=line[i][4],a2=line[i][5],a3=line[i][6],a4=line[i][7]) for line[i] in lines[0:]]
    return examples


class InputFeatures(object):
    def __init__(self, choices_features):
        self.choices_features = [{"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids} for input_ids, input_mask, segment_ids in choices_features]

def sampletofeature_sub(sample,tmoname,transformer_tokenizer,max_seq_length,sep_char):
    mask_padding_with_zero=True
    if tmoname.split("-")[0]=="xlnet":
        pad_on_left=True
        pad_token_segment_id=4
    else:
        pad_on_left=False
        pad_token_segment_id=0
    pad_token=0
    features_s=[]
    for index,example in enumerate(sample):
        subtitle=example.subtitle #subtitle
        context_tokens=example.context_sentence #question
        subtitle_tokens=subtitle
        choices_features=[]
        for eindex,ending in enumerate(example.endings):
            ending_tokens=ending#answer
            context_tokens_choice=subtitle_tokens+sep_char+context_tokens
            text_a=context_tokens_choice
            text_b=ending_tokens
            inputs=transformer_tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_seq_length)
            if tmoname.split("-")[0]=="roberta":
                inputs["token_type_ids"]=[0]*len(inputs["input_ids"])
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                segment_ids = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                input_mask = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                segment_ids = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                input_mask = token_type_ids + ([pad_token_segment_id] * padding_length)            
            assert len(input_ids) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            choices_features.append((input_ids,input_mask,segment_ids))
        features_s.append(InputFeatures(choices_features=choices_features))
    return features_s

def sampletofeature_vcpt(sample,tmoname,transformer_tokenizer,max_seq_length,sep_char):
    mask_padding_with_zero=True
    if tmoname.split("-")[0]=="xlnet":
        pad_on_left=True
        pad_token_segment_id=4
    else:
        pad_on_left=False
        pad_token_segment_id=0
    pad_token=0
    features_v=[]
    for index,example in enumerate(sample):
        context_tokens=example.context_sentence
        vcpt=example.vcpt
        vcpt_l=len(vcpt)
        vcpt_tokens=str(example.vcpt)
        vcpt_tokens=vcpt_tokens.replace("[", "")
        vcpt_tokens=vcpt_tokens.replace("]", "")
        vcpt_tokens=vcpt_tokens.replace("'", "")
        vcpt_tokens=vcpt_tokens.replace(",", "")
        vcpt_tokens=vcpt_tokens
        choices_features=[]
        for eindex,ending in enumerate(example.endings):
            ending_tokens=ending
            context_tokens_choice=vcpt_tokens+sep_char+context_tokens
            text_a=context_tokens_choice
            text_b=ending_tokens
            inputs=transformer_tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_seq_length)
            if tmoname.split("-")[0]=="roberta":
                inputs["token_type_ids"]=[0]*len(inputs["input_ids"])
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                segment_ids = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                input_mask = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                segment_ids = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                input_mask = token_type_ids + ([pad_token_segment_id] * padding_length)
            assert len(input_ids) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            choices_features.append((input_ids,input_mask,segment_ids))
        features_v.append(InputFeatures(choices_features=choices_features))
    return features_v

def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

def preprocess_transformer_sub(tmoname,transformer_tokenizer,max_seq_length,sample,sep_char):
    train_features_s=sampletofeature_sub(sample,tmoname,transformer_tokenizer,max_seq_length,sep_char)
    all_input_ids_s=torch.tensor(select_field(train_features_s,"input_ids"),dtype=torch.long)
    all_input_mask_s=torch.tensor(select_field(train_features_s,"input_mask"),dtype=torch.long)
    all_segment_ids_s=torch.tensor(select_field(train_features_s,"segment_ids"),dtype=torch.long)
    return all_input_ids_s, all_input_mask_s, all_segment_ids_s

def preprocess_transformer_vcpt(tmoname,transformer_tokenizer,max_seq_length,sample,sep_char):
    train_features_v=sampletofeature_vcpt(sample,tmoname,transformer_tokenizer,max_seq_length,sep_char)
    all_input_ids_v=torch.tensor(select_field(train_features_v,"input_ids"),dtype=torch.long)
    all_input_mask_v=torch.tensor(select_field(train_features_v,"input_mask"),dtype=torch.long)
    all_segment_ids_v=torch.tensor(select_field(train_features_v,"segment_ids"),dtype=torch.long)
    return all_input_ids_v, all_input_mask_v, all_segment_ids_v

class TVQADataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.raw_train = load_json(opt.train_path)
        self.raw_test = load_json(opt.test_path)
        self.raw_valid = load_json(opt.valid_path)
        self.vcpt_dict = load_pickle(opt.vcpt_path)
        self.normalize_v = opt.normalize_v
        self.with_ts = opt.with_ts
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

        # set word embedding / vocabulary
        self.word2idx_path = opt.word2idx_path
        self.idx2word_path = opt.idx2word_path
        self.vocab_embedding_path = opt.vocab_embedding_path
        self.embedding_dim = opt.embedding_size
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.offset = len(self.word2idx)

        # set entry keys
        if self.with_ts:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "located_sub_text"]
        else:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub_text"]
        self.vcpt_key = "vcpt"
        self.label_key = "answer_idx"
        self.qid_key = "qid"
        self.vid_name_key = "vid_name"
        self.located_frm_key = "located_frame"
        for k in self.text_keys + [self.vcpt_key, self.qid_key, self.vid_name_key]:
            if k == "vcpt":
                continue
            assert k in self.raw_valid[0].keys()

    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            return self.raw_test

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        items = []
        cur_vid_name = self.cur_data_dict[index][self.vid_name_key]
        
        R_q=self.cur_data_dict[index]["q"]
        R_a0=self.cur_data_dict[index]["a0"]
        R_a1=self.cur_data_dict[index]["a1"]
        R_a2=self.cur_data_dict[index]["a2"]
        R_a3=self.cur_data_dict[index]["a3"]
        R_a4=self.cur_data_dict[index]["a4"]
        if self.with_ts:
            R_sub=self.cur_data_dict[index]["sub_text"]
        else:
            R_sub=self.cur_data_dict[index]["sub_text"]

        # add text keys
        for k in self.text_keys:
            raw_words=self.cur_data_dict[index][k]

        # add vcpt
        if self.with_ts:
            cur_vis_sen = self.vcpt_dict[cur_vid_name]
        else:
            cur_vis_sen = self.vcpt_dict[cur_vid_name]
        cur_vis_sen = " , ".join(cur_vis_sen)
        R_vcpt=self.numericalize_vcpt(cur_vis_sen)

        #correct
        # add other keys
        if self.mode == 'test':
            items.append(666)  # this value will not be used
        else:
            items.append(int(self.cur_data_dict[index][self.label_key]))
        for k in [self.qid_key]:
            items.append(self.cur_data_dict[index][k])
        items.append(cur_vid_name)

        items.append(R_vcpt)
        items.append(R_q)
        items.append(R_a0)
        items.append(R_a1)
        items.append(R_a2)
        items.append(R_a3)
        items.append(R_a4)
        items.append(R_sub)

        return items

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        # !!!! remove comma here, since they are too many of them
        words = [w for w in words if w != ","]
        words = words + [eos_word] if eos else words
        return words

    def numericalize_vcpt(self, vcpt_sentence):
        """convert words to indices, additionally removes duplicated attr-object pairs"""
        attr_obj_pairs = vcpt_sentence.lower().split(", ")  # comma is also removed
        unique_pairs = []
        for pair in attr_obj_pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        Words=words
#        words.append("<eos>")
        return Words


class Batch(object):
    def __init__(self):
        self.__doc__ = "empty initialization"

    @classmethod
    def get_batch(cls, keys=None, values=None):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        assert keys is not None and values is not None
        for k, v in zip(keys, values):
            setattr(batch, k, v)
        return batch


def pad_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    def pad_sequences(sequences):
        sequences = [torch.LongTensor(s) for s in sequences]
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq[:end]
        return padded_seqs, lengths

    def pad_video_sequences(sequences):
        """sequences is a list of torch float tensors (created from numpy)"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        v_dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), v_dim).float()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
        return padded_seqs, lengths

    # separate source and target sequences
    column_data = list(zip(*data))
    label_key = "answer_idx"
    qid_key = "qid"
    vid_name_key = "vid_name"
    R_key=["R_vcpt","R_q", "R_a0", "R_a1", "R_a2", "R_a3", "R_a4", "R_sub"]
    all_keys = [label_key, qid_key, vid_name_key]+R_key
    all_values = []
    for i, k in enumerate(all_keys):
        if k == label_key:
            all_values.append(torch.LongTensor(column_data[i]))
        else:
            all_values.append(column_data[i])
    batched_data = Batch.get_batch(keys=all_keys, values=all_values)
    return batched_data


def preprocess_inputs(batched_data, max_sub_l, max_vcpt_l, tmoname, transformer_tokenizer, max_seq_length, sep_char, device):
    """clip and move to target device"""
    max_len_dict = {"sub": max_sub_l, "vcpt": max_vcpt_l}
    text_keys = ["R_vcpt", "R_q", "R_a0", "R_a1", "R_a2", "R_a3", "R_a4", "R_sub"]
    label_key = "answer_idx"
    qid_key = "qid"
    model_in_list = []
    T = []
    for k in text_keys :
        v = getattr(batched_data, k)
        if k in max_len_dict:
            ctx, ctx_l = v
            max_l = min(ctx.size(1), max_len_dict[k])
            if ctx.size(1) > max_l:
                ctx_l = ctx_l.clamp(min=1, max=max_l)
                ctx = ctx[:, :max_l]
            model_in_list.extend([ctx.to(device), ctx_l.to(device)])
        else:
            if(k not in text_keys):
                model_in_list.extend([v[0].to(device), v[1].to(device)])
            if(k in text_keys):
                T.extend([v])
            else:
                model_in_list.extend([v])

    sample=read_examples(T[0],T[1],T[7],T[2],T[3],T[4],T[5],T[6])
    all_input_ids_v, all_input_mask_v, all_segment_ids_v=preprocess_transformer_vcpt(tmoname,transformer_tokenizer,max_seq_length,sample,sep_char)
    all_input_ids_s, all_input_mask_s, all_segment_ids_s=preprocess_transformer_sub(tmoname,transformer_tokenizer,max_seq_length,sample,sep_char)

    model_in_list.extend([all_input_ids_s.to(device)])
    model_in_list.extend([all_input_mask_s.to(device)])
    model_in_list.extend([all_segment_ids_s.to(device)])

    model_in_list.extend([all_input_ids_v.to(device)])
    model_in_list.extend([all_input_mask_v.to(device)])
    model_in_list.extend([all_segment_ids_v.to(device)])

    target_data = getattr(batched_data, label_key)
    target_data = target_data.to(device)
    qid_data = getattr(batched_data, qid_key)
    return model_in_list, target_data, qid_data

if __name__ == "__main__":
    # python tvqa_dataset.py --input_streams sub
    import sys
    from config import BaseOptions
    sys.argv[1:] = ["--input_streams", "sub"]
    opt = BaseOptions().parse()

    dset = TVQADataset(opt, mode="valid")
    data_loader = DataLoader(dset, batch_size=16, shuffle=False, collate_fn=pad_collate)
    transformer_tokenizer = tokenizer_class.from_pretrained(tmoname, do_lower_case=True, cache_dir=None)

    for batch_idx, batch in enumerate(data_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, tmoname, transformer_tokenizer, opt.max_seq_length, opt.sep_char, opt.device)
        break
