# utils/object_adapter.py
import numpy as np
import torch
from objutil import all_pureobjects_from_grid                # 直接复用你现有实现

# ── 全局可调整参数 ───────────────────────────────────────────────
# param ＝ 3-bool 组合列表，跟你以前 main 流程里的保持一致即可
DEFAULT_PARAM_COMBINATIONS = [
    (False, False, False),
    (True,  False, False),
    (False, True,  False),
    (False, False, True),
]

def _obj_to_mask(obj, h=30, w=30):
    """obj 是 set((color,(r,c))) → 返回 bool mask[h,w]"""
    mask = np.zeros((h, w), dtype=bool)
    for _, (r, c) in obj:
        mask[r, c] = True
    return mask

def extract_objects_from_grid(grid,
                              pair_id: int = 0,
                              in_or_out: str = "in",
                              param=None,
                              background_color=None,
                              canvas_size: int = 30):
    """
    兼容 CompressARC 的统一接口

    Returns
    -------
    obj_dicts : List[dict]   [{id,color,size,bbox}, …]
    masks     : torch.BoolTensor (N,canvas_size,canvas_size)
    attrs     : dict[str, torch.Tensor]
    """
    if param is None:
        param = DEFAULT_PARAM_COMBINATIONS

    # ① 调用旧函数，得到 frozenset(obj)
    obj_set = all_pureobjects_from_grid(
        param, pair_id, in_or_out, grid,
        [grid.shape[0], grid.shape[1]],
        background_color=background_color,
    )

    obj_dicts, mask_list, colors, sizes = [], [], [], []
    for idx, obj in enumerate(obj_set):
        mask = _obj_to_mask(obj, canvas_size, canvas_size)
        bbox_r = [r for _, (r, _) in obj]
        bbox_c = [c for _, (_, c) in obj]
        main_color = max((col for col, _ in obj), key=lambda col: sum(c == col for c, _ in obj))

        obj_dicts.append({
            "id": idx,
            "color": main_color,
            "size": len(obj),
            "bbox": (min(bbox_r), min(bbox_c), max(bbox_r), max(bbox_c)),
        })
        mask_list.append(mask)
        colors.append(main_color)
        sizes.append(len(obj))

    masks = (torch.from_numpy(np.stack(mask_list, 0))
             if mask_list else torch.zeros((0, canvas_size, canvas_size), dtype=torch.bool))

    attrs = {
        "color": torch.tensor(colors, dtype=torch.long),
        "size":  torch.tensor(sizes,  dtype=torch.float),
    }
    return obj_dicts, masks, attrs
