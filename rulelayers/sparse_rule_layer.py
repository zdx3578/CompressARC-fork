import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.operator_bank import OP_BANK

class SparseRuleLayer(nn.Module):
    """
    输入: attr_tensor (N_obj, D) , obj_masks (N_obj,H,W)
    输出: patched_grid (H,W)
    """
    def __init__(self, attr_dim, n_colors, K_ops=8,  temp=1.0):
        super().__init__()
        # self.K = K_ops
        self.attr_dim = attr_dim
        self.temp = 0.3  #temp
        self.n_params = n_colors
        self.selector_temp = 0.3

        self.K = min(K_ops, len(OP_BANK))
        self.op_names = list(OP_BANK.keys())[:self.K]
        self.selector = nn.Linear(attr_dim, self.K)
        self.param_head = nn.Linear(attr_dim, self.K * self.n_params)

        # 用于控制recolor_mask打印输出
        self.debug_buffer = []
        self.max_debug_per_line = 5


        # self.selector = nn.Linear(attr_dim, K_ops)       # 选算子
        # self.param_head = nn.Linear(attr_dim, K_ops*4)   # 每算子最多 4 个参数
        # self.op_names = list(OP_BANK.keys())[:K_ops]     # 固定顺序


    def forward(self, canvas, attr_tensor, obj_masks):
        """
        canvas      : (H,W) int tensor
        attr_tensor : (N,D) float
        obj_masks   : (N,H,W) bool
        """
        # 重置debug buffer
        self.debug_buffer = []

        N = attr_tensor.size(0)
        sel_logits = self.selector(attr_tensor) / 0.3 #self.temp      # (N,K)
        sel_prob   = F.gumbel_softmax(sel_logits, tau=self.temp, hard=True)  # (N,K)

        params_raw = 5 * self.param_head(attr_tensor)                # (N,K*P)
        # params_raw
        params_raw = params_raw.view(N, self.K, self.n_params)

        # print('!!debug111')
        # print(canvas)

        # 遍历对象

        for i in range(N):
            k = sel_prob[i].argmax().item()
            op_name = self.op_names[k]
            op_func = OP_BANK[op_name]
            mask_i  = obj_masks[i]
            p_i     = params_raw[i,k]
            H, W     = mask_i.shape
            if op_name == "recolor_mask":
                color_id = p_i.softmax(dim=0).argmax().item()
                debug_msg = f"obj={i} op=recolor_mask color={color_id}"
                self.debug_buffer.append(debug_msg)

                # 当缓冲区满了或者是最后一个对象时，打印并清空
                if len(self.debug_buffer) >= self.max_debug_per_line or i == N - 1:
                    print(f"[DBG Rule] {' | '.join(self.debug_buffer)}")
                    self.debug_buffer = []
            if canvas.dim() == 1 or canvas.numel() < mask_i.numel():
                print(f"[DEBUG] before {op_name}: canvas.shape={canvas.shape}, "
                    f"mask.shape={mask_i.shape}, obj_id={i}")
            if canvas.dim() == 1:
                print(f"[DBG] before {self.op_names[k]}  obj={i}  flat_len={canvas.numel()}")

            # print('!!debug')
            # print(canvas)

            # ---------- 取基准画布，永远 2-D ----------
            if canvas.dim() != 2 or canvas.numel() != H * W:
                # 无论之前被压扁成什么，都重建为 (H,W)
                canvas_2d = canvas.new_full((H, W), fill_value=0)
                if canvas.numel() == H * W:
                    canvas_2d.copy_(canvas.view(H, W))
                canvas = canvas_2d                     # 保证后续正常
            # ---------- 运行算子 ----------
            # canvas_tmp = op_func(canvas.clone(), mask_i, p_i)
            # rulelayers/sparse_rule_layer.py  内 forward() 关键行
            if mask_i.sum() == 0:        # 对象空洞/出界，直接跳过
                continue
            canvas_tmp = op_func(canvas.clone(), mask_i, p_i,   # ❌ 旧
                                temp=self.temp, hard=self.hard)  # ✅ 新


            # 再保险：算子若返回奇形，也强制 view
            if canvas_tmp.shape != (H, W):
                if canvas_tmp.numel() == H * W:
                    canvas_tmp = canvas_tmp.view(H, W)
                else:
                    raise ValueError(
                        f"[{op_func.__name__}] invalid shape {canvas_tmp.shape}"
                    )

            # ---------- 合并结果 ----------
            canvas = torch.where(mask_i, canvas_tmp, canvas)

            # 确保颜色值在有效范围内 (0-9)
            canvas = torch.clamp(canvas, 0, 9)

        # 如果还有未打印的debug信息，在这里打印
        if self.debug_buffer:
            print(f"[DBG Rule] {' | '.join(self.debug_buffer)}")
            self.debug_buffer = []

        return canvas
