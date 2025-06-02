"""
统一的属性插件接口
每个插件函数接受 obj_dict → 返回 1-D numpy / torch 标量或 one-hot
"""

import numpy as np
import torch

REGISTRY = {}

def register(name):
    def _decor(f):
        REGISTRY[name] = f
        return f
    return _decor

# ─── 基础插件示例 ───────────────────────────────
@register("size")
def attr_size(obj):
    return np.array([obj["size"]], dtype=np.float32) / 900.0  # 归一化

@register("holes")
def attr_holes(obj):
    onehot = np.zeros(9, dtype=np.float32)
    onehot[min(obj["holes"], 8)] = 1.0
    return onehot

@register("is_rect")
def attr_is_rect(obj):
    """
    返回 1.0 表示对象像素恰好填满其外接矩形（矩形对象），
         0.0 表示存在空洞或形状不全覆盖外接框。
    """
    xmin, ymin, xmax, ymax = obj["bbox"]          # (x0,y0,x1,y1)
    bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)
    return np.array([float(bbox_area == obj["size"])], dtype=np.float32)

# TODO: 添加更多插件
# ──────────────────────────────────────────────

def build_attr_tensor(obj_dicts, keys=None):
    """
    obj_dicts : List[dict]  from extract_objects
    keys      : List[str]   要激活的插件
    return    : Tensor (N_obj, D_total)
    """
    keys = keys or list(REGISTRY.keys())
    feats = []
    for obj in obj_dicts:
        cols = [REGISTRY[k](obj) for k in keys]
        feats.append(np.concatenate(cols, axis=0))
    return torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)
