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

@register("recolor_mask")
def op_recolor_mask(canvas, mask, params):
    """
    重上色算子：把 mask 内像素改为 params 指定颜色
    canvas : (H,W) int tensor  *或* 展平 1-D
    mask   : (H,W) bool tensor
    params : (...>=1)  第 0 维为颜色 logits
    """
    # --- 取颜色 id ---
    color = params[0].softmax(0).argmax().long().clamp(0, 9)

    # --- 确保形状匹配 ---
    if canvas.dim() == 1:
        if canvas.numel() == mask.numel():
            canvas = canvas.view_as(mask)
        else:  # 兜底：直接 reshape 失败，用 debug
            raise ValueError(f"[recolor_mask] shape mismatch: "
                             f"canvas.numel={canvas.numel()} vs mask={mask.shape}")

    # --- 上色 ---
    canvas = canvas.clone()          # 避免就地影响上游
    canvas[mask] = color
    return canvas



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

