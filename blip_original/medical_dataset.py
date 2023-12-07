import json
import os
from socket import IP_DEFAULT_MULTICAST_LOOP
import torch
import pydicom
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from .utils import pre_caption
import os
import tqdm

label_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
              'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
              'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

node = [
    'normal','other finding','heart','cardiomegaly','spine','scoliosis','pleural','effusion','thickening','pneumothorax',
    'bone', 'bone fractures','lung','emphysema','pneumonia','edema','atelectasis','clcatrix','opacity','lesion',
    'mediastinum','hernia','calcinosis','foreign object','airspace','airspace disease','hypoinflation'
]
nodes = '-'.join(node)

node_inds = [0,1,2,3,3,4,4,5,5,5,5,6,6,7,7,7,7,7,7,7,7,8,8,8,9,10,10,10]
node_labels = [0,2,3,1,4,1,5,1,6,6,6,1,7,1,8,8,8,8,8,8,8,1,9,9,10,1,11,11]
node_inds = [each+1 for each in node_inds]
node_labels = [each+1 for each in node_labels]

node_relations = list(range(len(node_inds)))
node_relations = [each+1 for each in node_relations]

skg = {
    'nodes':nodes, 
    'node_inds':node_inds, 
    'node_labels':node_labels, 
    'node_relations': node_relations
}

def check_empty(ann, image_root):
    ann_check_exist = []
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

class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=90, prompt='', dataset='', args=None):
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.dataset = dataset
        self.args = args
        self.ann_check_exist = check_empty(self.ann, self.image_root)

    def __len__(self):
        return len(self.ann_check_exist)
    
    def __getitem__(self, index):    
        ann = self.ann_check_exist[index]
        image_path = ann['image_path']
        '''if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':'''
        path = os.path.join(self.image_root, image_path[0]).replace('.jpg', '.dcm')
        ds = pydicom.dcmread(path)

        ds_image = ds.pixel_array.astype(float)
        #image = Image.open(os.path.join(self.image_root, path_dcm)).convert('RGB')
        image = Image.fromarray(ds_image).convert('RGB')
        image = self.transform(image)
        caption = self.prompt + pre_caption(ann['report'], self.max_words)

        knowledge_skg = skg
        knowledge_tc = ''
        triplet_len = len(ann['triplet'])
        if triplet_len > 30:
            for i in range(30):
                knowledge_tc += ann['triplet'][i]
                if i < 29:
                    knowledge_tc += ' '
        else:
            tri_idx = 0
            for triplet in ann['triplet']:
                knowledge_tc += triplet
                tri_idx += 1
                if tri_idx < triplet_len:
                    knowledge_tc += ' '
        knowledge_tc = pre_caption(knowledge_tc, self.max_words)  #triplet can see each other

        return image, caption, knowledge_skg, knowledge_tc
    
    
class generation_eval(Dataset):
    def __init__(
        self, 
        transform, 
        image_root, 
        ann_root, 
        split, 
        dataset, 
        args=None
    ):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.ann = self.annotation[split]
        self.transform = transform
        self.image_root = image_root
        self.dataset = dataset
        self.args = args
        self.ann_check_exist = check_empty(self.ann, self.image_root)

    def __len__(self):
        return len(self.ann_check_exist)
    
    def __getitem__(self, index):    
        
        ann = self.ann_check_exist[index]

        image_path = ann['image_path']
        '''if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':'''
        path = os.path.join(self.image_root, image_path[0]).replace('.jpg', '.dcm')
        ds = pydicom.dcmread(path)
        ds_image = ds.pixel_array.astype(float)
        #image = Image.open(os.path.join(self.image_root, path_dcm)).convert('RGB')
        image = Image.fromarray(ds_image).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['report'], 90)
        knowledge_skg = skg

        return image, caption, knowledge_skg, image_path
