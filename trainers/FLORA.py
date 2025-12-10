import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# from Dassl.dassl.engine import TRAINER_REGISTRY, TrainerX
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from loralib import apply_lora, mark_only_lora_as_trainable, get_lora_parameters

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Cifar10": "a photo of a {}.",
    "Cifar100": "a photo of a {}.",
    "OfficeHome": "a photo of a {}.",
    "Office31": "a photo of a {}.",
    "DomainNet": "a photo of a {}.",
    "PACS": "a photo of a {}.",
    "TerraIncognita": "a photo of a {}.",
    "VLCS": "a photo of a {}."
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'FLORA',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def named_modules_with_index(clip_model: nn.Module):
    assert hasattr(clip_model, "visual") and hasattr(clip_model.visual, "transformer") and hasattr(clip_model, "transformer"), \
        "The model should have both vision and text transformer modules! RN not supported, implement it yourself :)"
    total_vision_blocks = len(clip_model.visual.transformer.resblocks)
    total_text_blocks = len(clip_model.transformer.resblocks)
    for name, module in clip_model.named_modules():
        if "ln_post" in name:
            yield name, module, total_vision_blocks
        if "ln_final" in name:
            yield name, module, total_text_blocks 
        if "ln_pre" in name:
            yield name, module, 0
        splitname = name.split('resblocks.')
        if len(splitname) == 1: # not a resblock
            yield name, module, -1
        else:
            block_idx = int(splitname[-1].split('.')[0])
            yield name, module, block_idx

def trainable_norm_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    trainable_params = []
    for name, module, block_idx in named_modules_with_index(model):
        curr_modality = 'vision' if 'visual' in name else 'text'
        curr_index = vision_start if curr_modality == 'vision' else text_start
        if isinstance(module, torch.nn.LayerNorm) and block_idx >= curr_index and (modality == 'both' or modality == curr_modality):
            trainable_params.extend(list(module.parameters()))
            module.requires_grad_(True)
            print(f"Modality = {modality}, vision_start={vision_start}, text_start={text_start} ==> LayerNorm at {name} is trainable.")
        else:
            module.requires_grad_(False)
    return trainable_params

def trainable_bias_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    trainable_params = []

    for param in model.parameters():
        param.requires_grad_(False)

    for name, module, block_idx in named_modules_with_index(model):
        curr_modality = 'vision' if 'visual' in name else 'text'
        curr_index = vision_start if curr_modality == 'vision' else text_start
        if hasattr(module, "bias") and block_idx >= curr_index and (modality == 'both' or modality == curr_modality):
            module.bias.requires_grad_(True)
            trainable_params.append(module.bias)
            print(f"Modality = {modality}, vision_start={vision_start}, text_start={text_start} ==> Bias at {name}.bias is trainable.")
    
    return trainable_params


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"  
        self.cfg = cfg    
         
        ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        ctx_init = ctx_init.replace("_", " ")
        prompt_prefix = ctx_init
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.cuda()
        clip_model = clip_model.cuda()
    
        with torch.no_grad():
            text_features_zs = clip_model.encode_text(tokenized_prompts)       
            text_features_zs = text_features_zs / text_features_zs.norm(dim=-1, keepdim=True)
        self.text_features_zs = nn.Parameter(text_features_zs.detach(), requires_grad=True)
    
    
    def make_layernorm_trainable(self, clip_model, modality=None, vision_start=0, text_start=0):
        cfg = self.cfg
        if modality is None:
            modality = cfg.LORA.ENCODER
        if cfg.PEFT == 'ln':
            trainable_params = trainable_norm_params(clip_model, modality, vision_start, text_start)
            return trainable_params
        elif cfg.PEFT == 'lora':
            _ = apply_lora(cfg, clip_model, verbose=False)
            mark_only_lora_as_trainable(clip_model)
            trainable_params = get_lora_parameters(clip_model)
            return trainable_params
        elif cfg.PEFT == 'bitfit':
            trainable_params = trainable_bias_params(clip_model, modality, vision_start, text_start)
            return trainable_params


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda")
        self.clip_model = clip_model
        self.dataset = cfg.DATASET.NAME
        self.classnames = classnames
        self.use_second_stage = False
        self.cfg = cfg

    def forward(self, image):  
        cfg = self.cfg            
        image_features = self.image_encoder(image.type(self.dtype))  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        ctx_init = CUSTOM_TEMPLATES[self.dataset]
        ctx_init = ctx_init.replace("_", " ")
        prompt_prefix = ctx_init
        
        classnames = self.classnames
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.cuda()
        
        text_features_zs = self.clip_model.encode_text(tokenized_prompts)
        text_features_zs = text_features_zs / text_features_zs.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_zs = (logit_scale * image_features @ text_features_zs.t())/cfg.TAU
     
        # if self.use_second_stage:
        #     text_features = self.prompt_learner()
        #     logits = (logit_scale * image_features @ text_features.t())/cfg.TAU
        #     return logits
        # else:
        return logits_zs


# @TRAINER_REGISTRY.register()
class FLORA(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FLORA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.FLORA.PREC == "fp32" or cfg.TRAINER.FLORA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()   

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        self.stage_params = self.model.prompt_learner.make_layernorm_trainable(
            self.model.clip_model, modality=cfg.LORA.ENCODER, vision_start=0, text_start=0
        )
        self._stage_param_ids = {id(p) for p in self.stage_params}
        
        print("Model built in Stage-1")
        
        for param in self.stage_params:
            param.requires_grad = True
    

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.stage_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        total_stage_params = sum(p.numel() for p in self.stage_params if p.requires_grad)
        print(f"[Stage 1] Number of trainable parameters: {total_stage_params:,}")

        self.scaler = GradScaler() if cfg.TRAINER.FLORA.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.FLORA.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)