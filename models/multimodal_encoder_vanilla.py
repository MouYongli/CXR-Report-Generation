import os

import torch
from torch import nn
import torch.nn.functional as F
from models.med import *
base_path = os.getcwd()

class MultimodalEncoder(nn.Module):
    def __init__(self,                 
                 tokenizer = None,
                 args = None
                 ):
        """

        """            
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        med_config = os.path.join(base_path, 'configs/med_config_sci.json')
        med_config = BertConfig.from_json_file(med_config)
        med_config.add_cross_attention = True
        med_config.encoder_width = 768
        self.multimodal_encoder = BertModel(config=med_config, add_pooling_layer=False)
        text_width = self.multimodal_encoder.config.hidden_size
        self.itm_head = nn.Linear(text_width, 2)


        
    def forward(self, GA, caption):
        image_atts = torch.ones(GA.size()[:-1],dtype=torch.long)
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=90, return_tensors="pt")
        text.input_ids[:,0] = self.tokenizer.enc_token_id # add ENC token

        output = self.multimodal_encoder(text.input_ids, # used for Query
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = GA, # used for Key and Value. TBC
                                       encoder_attention_mask = image_atts, # which is full of ones. Every tokens can sensor each others.    
                                       return_dict = True,
                                      )
        vl_output = self.itm_head(output)

        # itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
        #                        dim=0)
        # loss_irm = F.cross_entropy(vl_output, itm_labels) # cross entropy, but bewteen [p(from 0 to 1)] and [1..(224 times)..1, 0..(448times)..0]?
        return vl_output