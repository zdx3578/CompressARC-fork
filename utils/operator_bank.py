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


@register("recolor_mask")
def op_recolor_mask(canvas, mask, params, temp=0.6):
    logits = params[:10]                                   # (10,)
    y      = torch.nn.functional.gumbel_softmax(
                logits, tau=temp, hard=True)               # one-hot (hard ST)
    palette = torch.arange(10, device=logits.device, dtype=logits.dtype)
    color_val = (y * palette).sum()                        # still differentiable

    # --- dtype & bounds ---
    color_val = color_val.round().clamp(0, 9).long()       # ★ 修复

    if canvas.dim() == 1 and canvas.numel() == mask.numel():
        canvas = canvas.view_as(mask)
    canvas = canvas.clone()
    canvas[mask] = color_val
    return canvas


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
    params[0]: color id 0‥9   （softmax、argmax 外部完成）
    params[1]: width 0‥4
    """
    color = params[0].long().clamp(0, 9)
    w     = params[1].long().clamp(0, 4)

    if mask.sum() == 0:
        return canvas                        # 空掩码直接返回

    # 取前景像素坐标，再求行、列平均 → 中心点
    coords  = mask.nonzero(as_tuple=False)   # (N, 2)  [[r,c],...]
    center  = coords.float().mean(dim=0).round().long()   # (2,)
    cx, cy  = center.tolist()

    # 画十字：横竖宽度 w
    canvas[max(cx-w, 0): cx+w+1, :] = color
    canvas[:, max(cy-w, 0): cy+w+1] = color
    return canvas


# TODO: 更多算子





@register("noop")
def op_noop(canvas, mask, params):
    """什么也不做，占位算子"""
    return canvas


@register("noop1")
def op_noop1(canvas, mask, params):
    """什么也不做，占位算子"""
    return canvas

@register("noop2")
def op_noop2(canvas, mask, params):
    """什么也不做，占位算子"""
    return canvas

@register("noop3")
def op_noop3(canvas, mask, params):
    """什么也不做，占位算子"""
    return canvas

@register("noop4")
def op_noop4(canvas, mask, params):
    """什么也不做，占位算子"""
    return canvas

