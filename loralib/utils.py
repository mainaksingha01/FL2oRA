import os

import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_lora(cfg, clip_model, verbose=True):
    list_lora_layers = []
    if cfg.LORA.ENCODER == 'text' or cfg.LORA.ENCODER == 'both':
        indices = INDEX_POSITIONS_TEXT[cfg.LORA.POSITION]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if verbose: print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=cfg.LORA.PARAMS, r=cfg.LORA.R, lora_alpha=cfg.LORA.ALPHA, dropout_rate=cfg.LORA.DROPOUT_RATE)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if cfg.LORA.ENCODER == 'vision' or cfg.LORA.ENCODER == 'both':
        indices = INDEX_POSITIONS_VISION[cfg.MODEL.BACKBONE.NAME][cfg.LORA.POSITION]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if verbose: print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=cfg.LORA.PARAMS, r=cfg.LORA.R, lora_alpha=cfg.LORA.ALPHA, dropout_rate=cfg.LORA.DROPOUT_RATE)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def save_lora(cfg, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in cfg.LORA.PARAMS:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in cfg.LORA.PARAMS:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in cfg.LORA.PARAMS:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in cfg.LORA.PARAMS:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': cfg.LORA.R,
        'alpha': cfg.LORA.ALPHA,
        'encoder': cfg.LORA.ENCODER,
        'params': cfg.LORA.PARAMS,
        'position': cfg.LORA.POSITION
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = cfg.MODEL.BACKBONE.NAME.replace('/', '').replace('-', '').lower()
    save_dir = f'{cfg.LORA.SAVE_PATH}/{backbone}/{cfg.EXP_NAME}/{cfg.DATASET.NUM_SHOTS}shots/seed{cfg.SEED}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{cfg.EXP_NAME}_lora.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(cfg, list_lora_layers):
    # to manage names like ViT-B/16
    backbone = cfg.MODEL.BACKBONE.NAME.replace('/', '').replace('-', '').lower()
    load_path = f'{cfg.LORA.SAVE_PATH}/{backbone}/{cfg.EXP_NAME}/{cfg.DATASET.NUM_SHOTS}shots/seed{cfg.SEED}/{cfg.EXP_NAME}_lora.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != cfg.LORA.R:
        raise ValueError(
            f"r mismatch: expected {cfg.LORA.R}, found {metadata['r']}")
    if metadata['alpha'] != cfg.LORA.ALPHA:
        raise ValueError(
            f"alpha mismatch: expected {cfg.LORA.ALPHA}, found {metadata['alpha']}")
    if metadata['encoder'] != cfg.LORA.ENCODER:
        raise ValueError(
            f"Encoder mismatch: expected {cfg.LORA.ENCODER}, found {metadata['encoder']}")
    if metadata['params'] != cfg.LORA.PARAMS:
        raise ValueError(
            f"Params mismatch: expected {cfg.LORA.PARAMS}, found {metadata['params']}")
    if metadata['position'] != cfg.LORA.POSITION:
        raise ValueError(
            f"Position mismatch: expected {cfg.LORA.POSITION}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in cfg.LORA.PARAMS and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in cfg.LORA.PARAMS and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in cfg.LORA.PARAMS and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in cfg.LORA.PARAMS and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')
