import os
from glob import glob
from typing import Any, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torchvision import transforms
import json
import pydicom
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import tqdm
import re
from lightning import LightningDataModule
import sys
sys.path.append('../')
import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def check_empty(ann, image_root, split):
    ann_check_exist = []
    ann = ann[split]
    for a in tqdm.tqdm(ann): 
        img_path = a['image_path']
        if isinstance(img_path, list):
            img_path = img_path[0]

        img_path = os.path.join(
            image_root, 
            img_path.replace('.jpg', '.dcm')
        )
        if os.path.exists(img_path):
            ann_check_exist.append(a)
    return ann_check_exist

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers

class MIMICDataset(Dataset):
    def __init__(self, transforms, data_dir, ann_dir, max_words=90, prompt='', split: str = "train"):
        self.annotation = json.load(open(os.path.join(ann_dir),'r'))
        self.ann = self.annotation
        self.transform = transforms
        self.image_root = data_dir
        self.max_words = max_words      
        self.prompt = prompt
        self.split = split
        self.ann_check_exist = check_empty(self.ann, self.image_root, self.split)

    def __len__(self):
        return len(self.ann_check_exist)
    
    def __getitem__(self, index):
        if self.split == "train":    
            ann = self.ann_check_exist[index]
            image_path = ann['image_path']
            path = os.path.join(self.image_root, image_path[0]).replace('.jpg', '.dcm')
            ds = pydicom.dcmread(path)
            ds_image = ds.pixel_array.astype(float)
            image = Image.fromarray(ds_image).convert('RGB')
            image = self.transform(image)
            caption_train = self.prompt + pre_caption(ann['report'], self.max_words)
            return image, caption_train
        elif self.split == "test":
            ann = self.ann_check_exist[index]
            image_path = ann['image_path']
            path = os.path.join(self.image_root, image_path[0]).replace('.jpg', '.dcm')
            ds = pydicom.dcmread(path)
            ds_image = ds.pixel_array.astype(float)
            image = Image.fromarray(ds_image).convert('RGB')
            image = self.transform(image)
            img_id = ann['id']
            caption = ann['report']
            return image, caption, img_id
        else:
            raise NameError("split must be in 'train' or 'test'")

    
class MIMICDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/DATA1/llm-research/MIMIC-CXR/files",
        ann_dir: str = "../../configs/data/mimic_annotation.json",
        batch_size: int = 2,
        prompt: str = "a picture of ",
        seed: int = 42,
        num_workers: int = 1
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        np.random.seed(self.hparams.seed)
      

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.num_workers = num_workers
        self.samplers = None

        self.train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

        self.setup()
    

    def setup(self, stage: Optional[str] = None):
        self.data_train = MIMICDataset(
            transforms=self.train_transforms, 
            data_dir=self.hparams.data_dir, 
            ann_dir=self.hparams.ann_dir, 
            prompt=self.hparams.prompt, 
            split='train'
        )

        self.data_val = MIMICDataset(
            transforms=self.val_transforms, 
            data_dir=self.hparams.data_dir, 
            ann_dir=self.hparams.ann_dir, 
            split='test'
        )

        self.data_test = MIMICDataset(
            transforms=self.val_transforms, 
            data_dir=self.hparams.data_dir, 
            ann_dir=self.hparams.ann_dir, 
            split='test'
        )
    
    def train_dataloader(self) -> DataLoader:
        '''utils.init_distributed_mode(self.args)
        if self.args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            self.samplers = create_sampler(self.data_train, True, num_tasks, global_rank)         
        else:
            self.samplers = None'''

        self.sampler = None
        return DataLoader(
            dataset=self.data_train,
            batch_size= self.hparams.batch_size,
            sampler=self.sampler,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        '''utils.init_distributed_mode(self.args)
        if self.args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            self.samplers = create_sampler(self.data_train, True, num_tasks, global_rank)         
        else:
            self.samplers = None'''

        self.sampler = None
        return DataLoader(
            dataset=self.data_val,
            sampler=self.sampler,
            batch_size= self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        '''utils.init_distributed_mode(self.args)
        if self.args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            self.samplers = create_sampler(self.data_train, True, num_tasks, global_rank)         
        else:
            self.samplers = None'''

        self.sampler = None
        return DataLoader(
            dataset=self.data_test,
            sampler=self.sampler,
            batch_size= self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

if __name__ == "__main__":
     #_ = MIMICDataModule()
    a = MIMICDataModule()

    train_data = a.train_dataloader()
    dataiter = iter(train_data)
    images, labels = dataiter.next()
    print(images.shape) #torch.Size([2, 3, 384, 384])
