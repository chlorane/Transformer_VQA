import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from tvqa_abc import ABC
from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs
from config import TestOptions
from utils import merge_two_dicts, save_json

from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertForMultipleChoice, BertTokenizer, RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer, XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer, AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer, get_linear_schedule_with_warmup)

import logging
logging.basicConfig(level=logging.ERROR)

def test(opt, dset, model):
    dset.set_mode(opt.mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    qid2preds = {}
    qid2targets = {}
    MODEL_CLASSES = {"bert": (BertConfig, BertForMultipleChoice, BertTokenizer),"xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),"roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),"albert":(AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES[tmodel]
    transformer_tokenizer = tokenizer_class.from_pretrained(tmoname, do_lower_case=True, cache_dir=None)
    corr=0
    for valid_idx, batch in tqdm(enumerate(valid_loader)):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, tmoname, transformer_tokenizer, opt.max_seq_length, opt.sep_char, device=opt.device)
        outputs = model(*model_inputs)
        pred_ids = outputs.data.max(1)[1].cpu().numpy().tolist()
        cur_qid2preds = {qid: pred for qid, pred in zip(qids, pred_ids)}
        qid2preds = merge_two_dicts(qid2preds, cur_qid2preds)
        cur_qid2targets = {qid:  target for qid, target in zip(qids, targets)}
        corr=corr+sum(pred_ids == targets.cpu().numpy())
        qid2targets = merge_two_dicts(qid2targets, cur_qid2targets)
    return qid2preds, qid2targets


def get_acc_from_qid_dicts(qid2preds, qid2targets):
    qids = qid2preds.keys()
    preds = np.asarray([int(qid2preds[ele]) for ele in qids])
    targets = np.asarray([int(qid2targets[ele]) for ele in qids])
    acc = sum(preds == targets) / float(len(preds))
    return acc


if __name__ == "__main__":
    opt = TestOptions().parse()
    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)
    tmodel=opt.tmodel
    tmoname=opt.tmoname
    model = ABC(opt)

    model.to(opt.device)
    cudnn.benchmark = True
    model_path = os.path.join("models", opt.model_dir, "best_valid.pth")
    model.load_state_dict(torch.load(model_path))
 
    all_qid2preds, all_qid2targets = test(opt, dset, model)

    if opt.mode == "valid":
        accuracy = get_acc_from_qid_dicts(all_qid2preds, all_qid2targets)
        print("In valid mode, accuracy is %.4f" % accuracy)

    save_path = os.path.join("models", opt.model_dir, "qid2preds_%s.json" % opt.mode)
    save_json(all_qid2preds, save_path)
