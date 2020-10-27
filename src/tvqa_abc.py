import torch
import sys
from torch import nn
sys.path.append('D:\TVQAvisualnew\model')
import numpy as np
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertForMultipleChoice, BertTokenizer, RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer, XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer, AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer, get_linear_schedule_with_warmup)
import pysrt
import os

class ABC(nn.Module):
    def __init__(self, opt):
        super(ABC, self).__init__()
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        self.bsz=opt.bsz
        self.Tmodel=opt.tmodel
        hidden_size_1 = opt.hsz1
        hidden_size_2 = opt.hsz2
        n_layers_cls = opt.n_layers_cls
        embedding_size = opt.embedding_size
        vocab_size = opt.vocab_size

        tmoname=opt.tmoname
        MODEL_CLASSES = {"bert": (BertConfig, BertForMultipleChoice, BertTokenizer),"xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),"roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),"albert":(AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer)}
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.Tmodel]
        config = config_class.from_pretrained(tmoname, num_labels=5, finetuning_task="TVQA", cache_dir=None)
        tokenizer=tokenizer_class.from_pretrained(tmoname, do_lower_case=True, cache_dir=None)

        if self.sub_flag:
            print("activate sub stream")
            self.sub_model_s=model_class.from_pretrained(tmoname,from_tf=bool(".ckpt" in tmoname),config=config,cache_dir=None)

        if self.vcpt_flag:
            print("activate vcpt stream")
            self.sub_model_v=model_class.from_pretrained(tmoname,from_tf=bool(".ckpt" in tmoname),config=config,cache_dir=None)

    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, all_input_ids_s, all_segment_ids_s, all_input_mask_s, all_input_ids_v, all_segment_ids_v, all_input_mask_v):
        if self.sub_flag:
            if self.Tmodel=="roberta":
                all_segment_ids_s=None
            input_s={"input_ids":all_input_ids_s, "attention_mask":all_input_mask_s, "token_type_ids":all_segment_ids_s, "output_attentions":True}
            sub_out=self.sub_model_s(**input_s)
            sub_out_c=sub_out[0]
        else:
            sub_out=0
            sub_out_c=0

        if self.vcpt_flag:
            if self.Tmodel=="roberta":
                all_segment_ids_v=None
            input_v={"input_ids":all_input_ids_v, "attention_mask":all_input_mask_v, "token_type_ids":all_segment_ids_v, "output_attentions":True}
            vcpt_out=self.sub_model_v(**input_v)
            vcpt_out_c=vcpt_out[0]
        else:
            vcpt_out=0
            vcpt_out_c=0

        out = sub_out_c + vcpt_out_c  # adding zeros has no effect on backward
        sftout=out.squeeze()
        return sftout

if __name__ == '__main__':
    from config import BaseOptions
    import sys
    sys.argv[1:] = ["--input_streams" "sub"]
    opt = BaseOptions().parse()

    model = ABC(opt)
    model.to(opt.device)
    test_in = model.get_fake_inputs(device=opt.device)
    test_out = model(*test_in)
