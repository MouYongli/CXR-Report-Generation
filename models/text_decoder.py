import warnings
warnings.filterwarnings("ignore")
import sys
import os

import logging
from models.vit_blip import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from functools import partial
from medical_knowledge.knowledge import create_knowledge
from medical_knowledge.SKG_knowledge import *
from models.tagencoder import TagEncoder, update_skg
import lightning as L

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
base_path = os.getcwd()
        
class TextDecoder(nn.module):
    def __init__(self,                 
                 config = os.path.join(base_path, './configs/med_config.json'),  
                 vision_width = 768,
                 args = None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        self.args = args

        # create the decoder
        decoder_config = config
        decoder_config.encoder_width = 768 # 1024 for large vit
        if args.bert == 'base':
            self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        elif args.bert == 'sci':
            self.text_decoder = BertLMHeadModel.from_pretrained('allenai/scibert_scivocab_uncased',config=decoder_config, ignore_mismatched_sizes=True)
        elif args.bert == 'cli':
            self.text_decoder = BertLMHeadModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT',config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        
    def forward(self, GA, caption):
        image_atts = torch.ones(GA.size()[:-1],dtype=torch.long)
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=90, return_tensors="pt")
        text.input_ids[:,0] = self.tokenizer.bos_token_id # add DEC token
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:,:self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = GA,
                                           encoder_attention_mask = image_atts,
                                           labels = decoder_targets,
                                           return_dict = True,
                                          )
        # loss_lm = decoder_output.loss
        return decoder_output
     
    