import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.operator_bank import OP_BANK

class SparseRuleLayer(nn.Module):
    """
    输入: attr_tensor (N_obj, D) , obj_masks (N_obj,H,W)
    输出: patched_grid (H,W)
    """
    def __init__(self, attr_dim, K_ops=8, hidden=64, temp=1.0):
        super().__init__()
        self.K = K_ops
        self.attr_dim = attr_dim
        self.temp = temp

        self.selector = nn.Linear(attr_dim, K_ops)       # 选算子
        self.param_head = nn.Linear(attr_dim, K_ops*4)   # 每算子最多 4 个参数

        self.op_names = list(OP_BANK.keys())[:K_ops]     # 固定顺序
        self.n_params = 4

    def forward(self, canvas, attr_tensor, obj_masks):
        """
        canvas      : (H,W) int tensor
        attr_tensor : (N,D) float
        obj_masks   : (N,H,W) bool
        """
        N = attr_tensor.size(0)
        sel_logits = self.selector(attr_tensor) / self.temp      # (N,K)
        sel_prob   = F.gumbel_softmax(sel_logits, tau=self.temp, hard=True)  # (N,K)

        params_raw = self.param_head(attr_tensor)                # (N,K*P)
        params_raw = params_raw.view(N, self.K, self.n_params)

        # 遍历对象
        for i in range(N):
            k = sel_prob[i].argmax().item()
            op_name = self.op_names[k]
            op_func = OP_BANK[op_name]
            mask_i  = obj_masks[i]
            p_i     = params_raw[i,k]
            canvas  = op_func(canvas, mask_i, p_i)

        return canvas
