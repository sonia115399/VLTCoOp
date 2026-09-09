import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import requests
import os
from tqdm import tqdm
import math
import json

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

from clip import clip

from clip.pmcclip import ModifiedResNet

from trainers.CoOp.coop_clip import load_clip_to_cpu
from trainers.prompt_templates import CUSTOM_TEMPLATES, BIOMEDCOOP_TEMPLATES

from transformers import AutoTokenizer, AutoModel

# Directory where the files should be located
directory = "clip/checkpoints"

# File URLs
pmcclip_files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}

# File URLs
pubmedclip_files = {
    "PubMedCLIP_ViT32.pth": "https://huggingface.co/sarahESL/PubMedCLIP/resolve/main/PubMedCLIP_ViT32.pth?download=true",
}


# Function to download a file
def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            # Use tqdm to show the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))  # Update progress bar by the chunk size
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")

def load_generated_prompts(dataset_root, dataset_name):
    # Try different potential paths
    import os
    potential_paths = [
        os.path.join(dataset_root, dataset_name, "generated_prompts.json"),         # data/corn/generated_prompts.json (if root is data)
        os.path.join(dataset_root, "generated_prompts.json"),                       # data/corn/generated_prompts.json (if root is data/corn)
    ]
    
    json_path = None
    for p in potential_paths:
        if os.path.exists(p):
            json_path = p
            break
            
    if not json_path:
        print(f"Warning: generated_prompts.json not found in {potential_paths}. Falling back to default BIOMEDCOOP_TEMPLATES.")
        return None
        
    print(f"Loading generated prompts from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # clip_model = load_clip_to_cpu(cfg)
        clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model
        self.register_model("clip", clip_model)

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
@TRAINER_REGISTRY.register()
class ZeroshotPubMedCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in pubmedclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PubMedCLIP (backbone: ViT-B/32)")
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        checkpoint = torch.load(os.path.join(directory,"PubMedCLIP_ViT32.pth"), map_location=self.device)
        clip_model.load_state_dict(checkpoint['state_dict'])

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model
        self.register_model("clip", clip_model)

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
@TRAINER_REGISTRY.register()
class ZeroshotBiomedCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_model.eval().to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([tokenizer(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts,False)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model.eval()
        self.register_model("clip", clip_model)

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
@TRAINER_REGISTRY.register()
class ZeroshotPMCCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in pmcclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")


        print(f"Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth')))
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth')))
        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'))
        text_projection_layer = nn.Parameter(text_projection_layer)
        self.text_encoder = text_encoder.to(self.device).eval()
        self.text_projection_layer = text_projection_layer.to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        # clip_model.eval().to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        tokenized_prompts = tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        prompts = tokenized_prompts['input_ids'].to(self.device)

        with torch.no_grad():

            output = self.text_encoder(prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
            pooler_output = output.pooler_output
            text_features = pooler_output @ self.text_projection_layer
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.image_encoder = image_encoder.to(self.device).eval()
        self.logit_scale = 4.4292
        self.register_model("image_encoder", image_encoder)
        

    def model_inference(self, image):
        image_features = self.image_encoder(image)
        if isinstance(image_features, dict):
            image_features = image_features['image_features']
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = math.exp(self.logit_scale)
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    templates = BIOMEDCOOP_TEMPLATES

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Load dynamic templates from JSON
        json_templates = load_generated_prompts(cfg.DATASET.ROOT, cfg.DATASET.NAME)
        
        # Helper to get prompts
        def get_prompts_list(cls_name):
            source = "default"
            found_templates = None
            
            if json_templates:
                # 1. Direct match
                if cls_name in json_templates: 
                    found_templates = json_templates[cls_name]
                    source = "JSON (Exact)"
                # 2. Underscore match
                elif cls_name.replace(" ", "_") in json_templates:
                    found_templates = json_templates[cls_name.replace(" ", "_")]
                    source = "JSON (Underscore)"
                # 3. Suffix match (Corn___Common_Rust)
                else:
                    for k in json_templates.keys():
                        if k.endswith(f"___{cls_name}") or k.endswith(f"___{cls_name.replace(' ', '_')}"):
                            found_templates = json_templates[k]
                            source = f"JSON (Suffix: {k})"
                            break
            
            # Fallback to Hardcoded if not found in JSON
            if found_templates is None:
                if cls_name in BIOMEDCOOP_TEMPLATES:
                    found_templates = BIOMEDCOOP_TEMPLATES[cls_name]
                    source = "Hardcoded (Exact)"
                else:
                     # Try finding in Hardcoded with fuzzy logic too
                     for k in BIOMEDCOOP_TEMPLATES.keys():
                        if k.endswith(f"___{cls_name}"):
                            found_templates = BIOMEDCOOP_TEMPLATES[k]
                            source = f"Hardcoded (Suffix: {k})"
                            break

            # Absolute Fallback
            if found_templates is None:
                print(f"[Prompts] Warning: No templates found for class '{cls_name}'. using generic.")
                return [f"a photo of a {cls_name}."] * 10
            
            print(f"[Prompts] Class '{cls_name}' -> Source: {source} | Count: {len(found_templates)}")
            # Print the first few prompts to debug
            for idx, p in enumerate(found_templates[:3]):
                 print(f"   - {p}")
            return found_templates

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)


        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        
        # We need to iterate differently because get_prompts_list returns the full list
        # We can just average the features of the top N prompts for each class directly
        
        final_class_embeddings = []
        for classname in classnames:
            templates = get_prompts_list(classname)
            # Take top N
            templates = templates[:num_temp]
            
            prompts = torch.cat([clip.tokenize(p) for p in templates]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Average
            class_feat = text_features.mean(dim=0)
            class_feat = class_feat / class_feat.norm()
            final_class_embeddings.append(class_feat)
            
        self.text_features = torch.stack(final_class_embeddings)
        self.clip_model = clip_model
        self.register_model("clip", clip_model)

@TRAINER_REGISTRY.register()
class ZeroshotPubMedCLIP2(ZeroshotPubMedCLIP):
    """Prompt ensembling."""

    templates = BIOMEDCOOP_TEMPLATES # Keeping this for reference, but build_model overrides logic

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        # Load dynamic templates from JSON
        json_templates = load_generated_prompts(cfg.DATASET.ROOT, cfg.DATASET.NAME)
        
        # Helper to get prompts
        def get_prompts_list(cls_name):
            if json_templates:
                if cls_name in json_templates: return json_templates[cls_name]
                if cls_name.replace(" ", "_") in json_templates: return json_templates[cls_name.replace(" ", "_")]
                for k in json_templates.keys():
                    if k.endswith(f"___{cls_name}") or k.endswith(f"___{cls_name.replace(' ', '_')}"):
                        return json_templates[k]
            if cls_name in BIOMEDCOOP_TEMPLATES: return BIOMEDCOOP_TEMPLATES[cls_name]
            return [f"a photo of a {cls_name}."] * 10

        # Check for files in the directory and download if necessary
        for filename, url in pubmedclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PubMedCLIP (backbone: ViT-B/32)")
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        checkpoint = torch.load(os.path.join(directory,"PubMedCLIP_ViT32.pth"), map_location=self.device)
        clip_model.load_state_dict(checkpoint['state_dict'])

        for params in clip_model.parameters():
            params.requires_grad_(False)


        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        final_class_embeddings = []
        for classname in classnames:
            templates = get_prompts_list(classname)
            templates = templates[:num_temp]
            
            # Use tokenizer for PubMedCLIP (if it uses CLIP tokenizer, it's clip.tokenize)
            prompts = torch.cat([clip.tokenize(p) for p in templates]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            class_feat = text_features.mean(dim=0)
            class_feat = class_feat / class_feat.norm()
            final_class_embeddings.append(class_feat)

        self.text_features = torch.stack(final_class_embeddings)
        self.clip_model = clip_model
        self.register_model("clip", clip_model)

@TRAINER_REGISTRY.register()
class ZeroshotBiomedCLIP2(ZeroshotBiomedCLIP):
    """Prompt ensembling."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        # Load dynamic templates from JSON
        json_templates = load_generated_prompts(cfg.DATASET.ROOT, cfg.DATASET.NAME)
        
        # Helper to get prompts
        def get_prompts_list(cls_name):
            if json_templates:
                if cls_name in json_templates: return json_templates[cls_name]
                if cls_name.replace(" ", "_") in json_templates: return json_templates[cls_name.replace(" ", "_")]
                for k in json_templates.keys():
                    if k.endswith(f"___{cls_name}") or k.endswith(f"___{cls_name.replace(' ', '_')}"):
                        return json_templates[k]
            if cls_name in BIOMEDCOOP_TEMPLATES: return BIOMEDCOOP_TEMPLATES[cls_name]
            return [f"a photo of a {cls_name}."] * 10
            
        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_model.eval().to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        final_class_embeddings = []
        for classname in classnames:
            templates = get_prompts_list(classname)
            templates = templates[:num_temp]
            
            prompts = torch.cat([tokenizer(p) for p in templates]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            class_feat = text_features.mean(dim=0)
            class_feat = class_feat / class_feat.norm()
            final_class_embeddings.append(class_feat)

        self.text_features = torch.stack(final_class_embeddings)
        self.clip_model = clip_model

@TRAINER_REGISTRY.register()
class ZeroshotPMCCLIP2(ZeroshotPMCCLIP):
    """Prompt ensembling."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        # Load dynamic templates from JSON
        json_templates = load_generated_prompts(cfg.DATASET.ROOT, cfg.DATASET.NAME)
        
        # Helper to get prompts
        def get_prompts_list(cls_name):
            if json_templates:
                if cls_name in json_templates: return json_templates[cls_name]
                if cls_name.replace(" ", "_") in json_templates: return json_templates[cls_name.replace(" ", "_")]
                for k in json_templates.keys():
                    if k.endswith(f"___{cls_name}") or k.endswith(f"___{cls_name.replace(' ', '_')}"):
                        return json_templates[k]
            if cls_name in BIOMEDCOOP_TEMPLATES: return BIOMEDCOOP_TEMPLATES[cls_name]
            return [f"a photo of a {cls_name}."] * 10

        # Check for files in the directory and download if necessary
        for filename, url in pmcclip_files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")


        print(f"Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth')))
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth')))
        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'))
        text_projection_layer = nn.Parameter(text_projection_layer)
        self.text_encoder = text_encoder.to(self.device).eval()
        self.text_projection_layer = text_projection_layer.to(self.device)
        self.image_encoder = image_encoder.to(self.device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')

        for params in self.image_encoder.parameters():
            params.requires_grad_(False)

        for params in self.text_encoder.parameters():
            params.requires_grad_(False)

        num_temp = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Prompt ensembling (n={num_temp})")

        final_class_embeddings = []
        for classname in classnames:
            templates = get_prompts_list(classname)
            templates = templates[:num_temp]
            
            tokenized_prompts = tokenizer(templates, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            prompts = tokenized_prompts['input_ids'].to(self.device)
            output = self.text_encoder(prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
            pooler_output = output.pooler_output
            text_features = pooler_output @ self.text_projection_layer
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            class_feat = text_features.mean(dim=0)
            class_feat = class_feat / class_feat.norm()
            final_class_embeddings.append(class_feat)

        self.text_features = torch.stack(final_class_embeddings)
        self.logit_scale = 4.4292
        self.register_model("image_encoder", image_encoder)

@TRAINER_REGISTRY.register()
class ZeroshotCLIPSelective(ZeroshotCLIP):
    """Selective Prompt Ensemble using Per-Class Image-Text Similarity Selection."""

    templates = BIOMEDCOOP_TEMPLATES

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        # Load dynamic templates from JSON
        print(f"DEBUG: Root={cfg.DATASET.ROOT}, Name={cfg.DATASET.NAME}")
        json_templates = load_generated_prompts(cfg.DATASET.ROOT, cfg.DATASET.NAME)
        
        # Helper to get prompts
        def get_templates_for_class(cls_name):
            source = "default"
            found_templates = None
            
            if json_templates:
                # 1. Direct match
                if cls_name in json_templates: 
                    found_templates = json_templates[cls_name]
                    source = "JSON (Exact)"
                # 2. Underscore match
                elif cls_name.replace(" ", "_") in json_templates:
                    found_templates = json_templates[cls_name.replace(" ", "_")]
                    source = "JSON (Underscore)"
                # 3. Suffix match (Corn___Common_Rust)
                else:
                    for k in json_templates.keys():
                        if k.endswith(f"___{cls_name}") or k.endswith(f"___{cls_name.replace(' ', '_')}"):
                            found_templates = json_templates[k]
                            source = f"JSON (Suffix: {k})"
                            break
            
            # Fallback to Hardcoded if not found in JSON
            if found_templates is None:
                if cls_name in BIOMEDCOOP_TEMPLATES:
                    found_templates = BIOMEDCOOP_TEMPLATES[cls_name]
                    source = "Hardcoded (Exact)"
                else:
                     # Try finding in Hardcoded with fuzzy logic too
                     for k in BIOMEDCOOP_TEMPLATES.keys():
                        if k.endswith(f"___{cls_name}"):
                            found_templates = BIOMEDCOOP_TEMPLATES[k]
                            source = f"Hardcoded (Suffix: {k})"
                            break

            # Absolute Fallback
            if found_templates is None:
                print(f"[Prompts] Warning: No templates found for class '{cls_name}'. using generic.")
                return [f"a photo of a {cls_name}."] * 10
            
            print(f"[Prompts] Class '{cls_name}' -> Source: {source} | Count: {len(found_templates)}")
            # Print the first few prompts to debug
            for idx, p in enumerate(found_templates[:3]):
                 print(f"   - {p}")
            return found_templates

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device).eval()

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # 1. Prepare Validation Loader
        val_loader = self.dm.val_loader
        val_img_feats_by_class = {i: [] for i in range(len(classnames))}
        
        if val_loader:
            print("Extracting validation image features for selection...")
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["img"].to(self.device)
                    labels = batch["label"].to(self.device)
                    feats = clip_model.encode_image(imgs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    for f, l in zip(feats, labels):
                        val_img_feats_by_class[l.item()].append(f)
        else:
            print("Warning: No validation set found. Selection will fall back to using all prompts.")

        # 3. Per-Class Selective Ensemble
        top_k = cfg.TRAINER.BIOMEDCOOP.N_PROMPTS
        print(f"Selecting Top-{top_k} prompts per class based on visual similarity...")

        final_class_embeddings = []

        with torch.no_grad():
            for cls_idx, cls_name in enumerate(classnames):
                templates = get_templates_for_class(cls_name)
                
                # Encode all candidate prompts for this class
                # (N_prompts, Dim)
                prompts_tokenized = torch.cat([clip.tokenize(p) for p in templates]).to(self.device)
                text_feats = clip_model.encode_text(prompts_tokenized)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                # Select best prompts if we have validation images for this class
                support_imgs = val_img_feats_by_class[cls_idx]
                
                if len(support_imgs) > 0:
                    # Stack images: (N_shots, Dim)
                    img_feats = torch.stack(support_imgs)
                    
                    # Compute Similarity Matrix: (N_prompts, N_shots)
                    sim_matrix = text_feats @ img_feats.t()
                    
                    # Score = Average similarity to all support images
                    score_per_prompt = sim_matrix.mean(dim=1) # (N_prompts,)
                    
                    # Get Top K indices
                    k = min(top_k, len(templates))
                    _, top_indices = torch.topk(score_per_prompt, k)
                    
                    selected_feats = text_feats[top_indices]
                else:
                    # Fallback: Use first K prompts if no validation images
                    k = min(top_k, len(templates))
                    selected_feats = text_feats[:k]
                
                # Average selected features to get Class Prototype
                # (Dim,)
                class_proto = selected_feats.mean(dim=0)
                class_proto = class_proto / class_proto.norm()
                final_class_embeddings.append(class_proto)

        # Stack to form classifier weights: (N_classes, Dim)
        self.text_features = torch.stack(final_class_embeddings)
        self.clip_model = clip_model
        self.register_model("clip", clip_model)
