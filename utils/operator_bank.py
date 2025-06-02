"""
可微算子插件库
每个算子签名:   output_grid = op(grid, mask, params_tensor)
               grid, mask : (H,W) torch.Tensor
               params_tensor: shape (P,) 由 RuleLayer 预测
"""
import torch
import torch.nn.functional as F

OP_BANK = {}
def register(name):
    def _decor(f):
        OP_BANK[name] = f
        return f
    return _decor

# ─── 示例算子 ──────────────────────────────────
@register("fill_line")
def op_fill_line(canvas, mask, params):
    """
    params[0]: color 0-9  (softmax 外部完成)
    params[1]: axis 0=rows 1=cols (Gumbel-St)
    """
    color  = params[0].long().clamp(0,9)
    axis   = (params[1] > 0).long()   # 0 or 1
    indices = mask.nonzero(as_tuple=False)
    if indices.numel()==0:
        return canvas
    if axis == 0:
        canvas[:, indices[:,1]] = color
    else:
        canvas[indices[:,0]] = color
    return canvas

@register("draw_cross")
def op_draw_cross(canvas, mask, params):
    """
    params[0]: color, params[1]: width (0-4)
    """
    color = params[0].long().clamp(0,9)
    w     = params[1].long().clamp(0,4)
    if mask.sum()==0:
        return canvas
    center = mask.float().mean(0).round().long()
    cx, cy = center.tolist()
    canvas[max(cx-w,0):cx+w+1, :] = color
    canvas[:, max(cy-w,0):cy+w+1] = color
    return canvas
# TODO: 更多算子
