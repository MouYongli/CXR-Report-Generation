import warnings
warnings.filterwarnings("ignore")
import os

import transformers
transformers.logging.set_verbosity_error()
from models.med import BertConfig, BertModel
import torch.nn.functional as F
from torch import nn

# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()
base_path = os.getcwd()
        
class ReportEncoder(nn.Module):
    def __init__(self,                 
                 med_config = os.path.join(base_path, './configs/med_config.json'),  
                 tokenizer = None,
                 args = None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
        """            
        super().__init__()

        self.tokenizer = tokenizer
        self.args = args
        if args.bert == 'base':
            med_config = os.path.join(base_path, 'configs/med_config_blip.json')
        elif args.bert == 'sci':
            med_config = os.path.join(base_path, 'configs/med_config_sci.json')
        elif args.bert == 'cli':
            med_config = os.path.join(base_path, 'configs/med_config_cli.json')
        med_config = BertConfig.from_json_file(med_config)
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_proj = nn.Linear(768, 256)

        
    def forward(self, caption):
        # caption = '[CLS]' + caption # CLS token is automatically added
        tokens= self.tokenizer(caption, padding='max_length', truncation=True, max_length=90, return_tensors="pt")

        text_output = self.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask,
                                        return_dict=True, mode='text')
        
        # Only the CLS token embedding wanted
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return text_feat
