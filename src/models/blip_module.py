import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L
from utils.blip_utils import *

class LitBlip(L.LightningModule):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 tokenizer = None,
                 embed_dim = 256,     
                 args = None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        #self.tokenizer = tokenizer
        self.args = args
        self.prompt = prompt
        
        if args.bert == 'base':
            med_config = 'configs/med_config_blip.json'
        elif args.bert == 'sci':
            med_config = 'configs/med_config_sci.json'
        elif args.bert == 'cli':
            med_config = 'configs/med_config_cli.json'
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        # med_config.encoder_width = vision_width
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        text_width = self.text_encoder.config.hidden_size

        self.text_decoder = BertLMHeadModel(config=med_config)
        
        self.vision_proj = nn.Linear(vision_width, 256)
        self.text_proj = nn.Linear(768, 256)

        self.itm_head = nn.Linear(text_width, 2) 

        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)      
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        if args.dataset_name == 'iu_xray':
            self.iu_proj = nn.Linear(768*2, 768)
            self.iu_proj_m = nn.Linear(768*2, 768)
            queue_size = 1380
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.cross_attn, self.cross_attn_m],
                            [self.iu_proj,self.iu_proj_m]]
        else:
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.cross_attn, self.cross_attn_m]
                            ]
        

        self.copy_params() 
        
        decoder_config = med_config
        decoder_config.encoder_width = vision_width
        if args.bert == 'base':
            self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        elif args.bert == 'sci':
            self.text_decoder = BertLMHeadModel.from_pretrained('allenai/scibert_scivocab_uncased',config=decoder_config, ignore_mismatched_sizes=True)
        elif args.bert == 'cli':
            self.text_decoder = BertLMHeadModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT',config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
       
        return

    def trainning_epoch_end(self, outputs);
         pass

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
       
        return

    def _common_step(self, batch, batch_idx):
        # training_step defines the train loop.
       
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
