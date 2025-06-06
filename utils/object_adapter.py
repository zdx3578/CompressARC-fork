# utils/object_adapter.py
import numpy as np
import torch
from utils.objutil import all_pureobjects_from_grid
# try:
#     from utils.objutil import all_pureobjects_from_grid  # external dependency
# except ImportError:  # Fallback to a simple local implementation
#     # from utils.objutil_sub import all_pureobjects_from_grid
#     raise ImportError("Please install the required dependencies for CompressARC.")

# ── 全局可调整参数 ───────────────────────────────────────────────
# param ＝ 3-bool 组合列表，跟你以前 main 流程里的保持一致即可
DEFAULT_PARAM_COMBINATIONS00 = [
    (False, False, False),
    (True,  False, False),
    (False, True,  False),
    (False, False, True),
]

DEFAULT_PARAM_COMBINATIONS = [
    (True,  False, False),
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

    obj_dicts, mask_list, colors, sizes , holes = [],  [], [], [], []
    for idx, obj in enumerate(obj_set):
        mask = _obj_to_mask(obj, canvas_size, canvas_size)
        bbox_r = [r for _, (r, _) in obj]
        bbox_c = [c for _, (_, c) in obj]
        main_color = max((col for col, _ in obj), key=lambda col: sum(c == col for c, _ in obj))
        hole_cnt = _flood_fill_holes(mask)

        obj_dicts.append({
            "id": idx,
            "color": main_color,
            "size": len(obj),
            "bbox": (min(bbox_r), min(bbox_c), max(bbox_r), max(bbox_c)),
            "holes": hole_cnt
        })
        mask_list.append(mask)
        colors.append(main_color)
        sizes.append(len(obj))
        holes.append(hole_cnt)

        # hole_cnt = _flood_fill_holes(mask)
        # obj_dicts[-1]["holes"] = hole_cnt
        # attrs["holes"] = torch.tensor([*attrs.get("holes", []), hole_cnt], dtype=torch.long)



    masks = (torch.from_numpy(np.stack(mask_list, 0))
             if mask_list else torch.zeros((0, canvas_size, canvas_size), dtype=torch.bool)).to(torch.get_default_device())

    attrs = {
        "color": torch.tensor(colors, dtype=torch.long).to(torch.get_default_device()),
        "size":  torch.tensor(sizes,  dtype=torch.float).to(torch.get_default_device()),
        "holes": torch.tensor(holes,  dtype=torch.long).to(torch.get_default_device()),
    }
    return obj_dicts, masks, attrs

def _flood_fill_holes(mask: np.ndarray) -> int:
    """返回 mask 中封闭空洞的个数（4-连通背景连通域，排除外边界）。"""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    holes = 0
    for x in range(H):
        for y in range(W):
            if mask[x, y] or visited[x, y]:
                continue
            # 背景像素
            stack = [(x, y)]
            touches_border = False
            while stack:
                i, j = stack.pop()
                if not (0 <= i < H and 0 <= j < W):
                    continue
                if mask[i, j] or visited[i, j]:
                    continue
                visited[i, j] = True
                if i in (0, H-1) or j in (0, W-1):
                    touches_border = True
                stack.extend([(i-1,j),(i+1,j),(i,j-1),(i,j+1)])
            if not touches_border:
                holes += 1
    print(f"[DEBUG] Found {holes} holes in mask of shape {mask.shape}")
    return holes


def assert_holes_consistency(obj_dicts, masks):
    """Recompute holes from ``masks`` and assert they match ``obj_dicts``."""
    for i, mask in enumerate(masks):
        recomputed = _flood_fill_holes(mask.cpu().numpy())
        print(f"obj {i}: color={obj_dicts[i]['color']} holes={recomputed}")
        assert recomputed == obj_dicts[i]["holes"], (
            f"Hole count mismatch for object {i}: "
            f"expected {obj_dicts[i]['holes']}, got {recomputed}"
        )
