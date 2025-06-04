import time
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
# from layers.sparse_rule_layer import SparseRuleLayer

from rulelayers.sparse_rule_layer import SparseRuleLayer
from utils import attr_registry
from utils.attr_registry import build_attr_tensor


import os
import sys

debugstep = 40
reconstrucstep = 300

maxsteps = 2000

def debug_train_predictions(task, logits, pred_idx, train_step, folder, task_name, rule_layer=None, USE_RULE_LAYER=False):
    """
    调试输出训练样例上的预测效果
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if (train_step+1) % debugstep != 0:  # 只在特定步数输出
        return

    print(f"\n=== DEBUG: Train Examples Prediction at Step {train_step+1} ===")

    # 创建调试图像
    n_train = task.n_train
    if n_train == 0:
        return

    fig, axes = plt.subplots(n_train, 4, figsize=(16, 4*n_train))
    if n_train == 1:
        axes = axes.reshape(1, -1)

    for train_idx in range(n_train):
        # 1. 原始输入
        input_grid = task.problem[train_idx, :, :, 0].cpu().numpy()
        axes[train_idx, 0].imshow(input_grid, cmap='tab10', vmin=0, vmax=9)
        axes[train_idx, 0].set_title(f'Train {train_idx}: Input')
        axes[train_idx, 0].axis('off')

        # 2. 目标输出
        target_grid = task.problem[train_idx, :, :, 1].cpu().numpy()
        axes[train_idx, 1].imshow(target_grid, cmap='tab10', vmin=0, vmax=9)
        axes[train_idx, 1].set_title(f'Train {train_idx}: Target')
        axes[train_idx, 1].axis('off')

        # 3. 网络原始预测（应用规则前）
        raw_pred = logits[train_idx, :, :, :, 1].argmax(dim=0).cpu().numpy()
        axes[train_idx, 2].imshow(raw_pred, cmap='tab10', vmin=0, vmax=9)
        axes[train_idx, 2].set_title(f'Train {train_idx}: Raw Net Pred')
        axes[train_idx, 2].axis('off')

        # 4. 最终预测（应用规则后）
        final_pred = pred_idx[train_idx, :, :, 1].cpu().numpy()
        axes[train_idx, 3].imshow(final_pred, cmap='tab10', vmin=0, vmax=9)
        rule_status = "(+Rule)" if USE_RULE_LAYER and rule_layer is not None and train_step >= reconstrucstep else "(NoRule)"
        axes[train_idx, 3].set_title(f'Train {train_idx}: Final {rule_status}')
        axes[train_idx, 3].axis('off')

        # 计算准确率
        target_flat = target_grid.flatten()
        raw_flat = raw_pred.flatten()
        final_flat = final_pred.flatten()

        raw_acc = np.mean(target_flat == raw_flat)
        final_acc = np.mean(target_flat == final_flat)

        print(f"Train Example {train_idx}:")
        print(f"  Raw Network Accuracy: {raw_acc:.3f}")
        print(f"  Final Accuracy: {final_acc:.3f}")
        if USE_RULE_LAYER and rule_layer is not None and train_step >= reconstrucstep:
            print(f"  Rule Improvement: {final_acc - raw_acc:+.3f}")

    plt.tight_layout()
    debug_fname = folder + f'DEBUG_train_pred_{task_name}_step_{train_step+1}.png'
    plt.savefig(debug_fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Debug visualization saved: {debug_fname}")

    # 如果使用规则层，输出规则分析
    if USE_RULE_LAYER and rule_layer is not None and train_step >= reconstrucstep:
        debug_rule_analysis(task, rule_layer, train_step)

def debug_rule_analysis(task, rule_layer, train_step):
    """
    分析规则层的行为
    """
    print(f"\n=== Rule Layer Analysis at Step {train_step+1} ===")

    with torch.no_grad():
        for train_idx in range(task.n_train):
            print(f"\nTrain Example {train_idx}:")

            # 获取属性和掩码
            attr_tensor = task.output_attr_tensor[train_idx]  # (N_obj, D)
            obj_masks = task.output_obj_masks[train_idx]      # (N_obj, H, W)

            if attr_tensor.shape[0] == 0:
                print("  No objects detected")
                continue

            # 规则选择概率
            selector_logits = rule_layer.selector(attr_tensor)  # (N_obj, K_ops)
            selector_probs = selector_logits.softmax(dim=-1)
            chosen_ops = selector_probs.argmax(dim=-1).cpu().tolist()

            print(f"  Objects: {attr_tensor.shape[0]}")
            print(f"  Chosen Operations: {chosen_ops}")

            # 颜色参数
            if hasattr(rule_layer, 'param_head'):
                param_logits = rule_layer.param_head(attr_tensor)
                if param_logits.shape[-1] > 0:  # 有颜色参数
                    color_logits = param_logits[:, 0]  # 第一个参数通常是颜色
                    predicted_colors = color_logits.softmax(-1).argmax(-1).cpu().tolist()
                    print(f"  Predicted Colors: {predicted_colors}")

            # 打印每个对象的规则选择概率（top-2）
            for obj_idx in range(min(3, attr_tensor.shape[0])):  # 只显示前3个对象
                probs = selector_probs[obj_idx].cpu().numpy()
                top2_indices = np.argsort(probs)[-2:][::-1]
                print(f"    Object {obj_idx}: Op{top2_indices[0]}({probs[top2_indices[0]]:.3f}), Op{top2_indices[1]}({probs[top2_indices[1]]:.3f})")


# 权重保存和加载功能
def save_checkpoint(model, optimizer, train_step, task_name, folder, rule_layer=None):
    """保存训练检查点"""
    checkpoint = {
        'train_step': train_step,
        'task_name': task_name
    }

    # ARCCompressor使用weights_list而不是state_dict
    if hasattr(model, 'weights_list'):
        checkpoint['model_weights_list'] = [w.clone() for w in model.weights_list]
    else:
        # 如果是标准的nn.Module
        checkpoint['model_state_dict'] = model.state_dict()

    # 保存优化器状态
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # 如果有rule_layer，也要保存
    if rule_layer is not None:
        checkpoint['rule_layer_state_dict'] = rule_layer.state_dict()

    checkpoint_path = os.path.join(folder, f'checkpoint_step_{train_step}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {train_step}: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, device, rule_layer=None):
    """加载训练检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载ARCCompressor的weights_list
    if hasattr(model, 'weights_list') and 'model_weights_list' in checkpoint:
        loaded_weights = checkpoint['model_weights_list']
        if len(loaded_weights) == len(model.weights_list):
            for i, weight in enumerate(loaded_weights):
                model.weights_list[i].data.copy_(weight.data)
        else:
            print(f"Warning: Weight list length mismatch. Model: {len(model.weights_list)}, Checkpoint: {len(loaded_weights)}")
    elif 'model_state_dict' in checkpoint:
        # 如果是标准的nn.Module
        model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 如果有rule_layer，也要加载
    if rule_layer is not None and 'rule_layer_state_dict' in checkpoint:
        rule_layer.load_state_dict(checkpoint['rule_layer_state_dict'])

    train_step = checkpoint['train_step']
    print(f"Checkpoint loaded from step {train_step}: {checkpoint_path}")
    return train_step

def find_latest_checkpoint(folder, task_name):
    """查找最新的检查点文件"""
    checkpoint_files = [f for f in os.listdir(folder) if f.startswith('checkpoint_step_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None

    # 按步数排序，找到最新的
    steps = [int(f.split('_')[2].split('.')[0]) for f in checkpoint_files]
    latest_step = max(steps)
    return os.path.join(folder, f'checkpoint_step_{latest_step}.pth')

possible_pypaths = [
    '/kaggle/input/3-28arcdsl'
    '/kaggle/input/3-28arcdsl/forpopper2',
    '/kaggle/input/3-28arcdsl/bateson',
    '/Users/zhangdexiang/github/VSAHDC/arcv2',
    '/Users/zhangdexiang/github/VSAHDC/arcv2/forpopper2',
    '/Users/zhangdexiang/github/VSAHDC',
    '/home/zdx/github/VSAHDC/arcv2',
    '/home/zdx/github/VSAHDC/arcv2/forpopper2',
    '/home/zdx/github/VSAHDC',
    '/home/zdx/github/VSAHDC/arcMrule',
    '/home/zdx/github/VSAHDC/arcMrule/diffstar',
    '/another/path/to/check'
]

# 遍历路径列表，检查并按需加载
for path in possible_pypaths:
    if os.path.exists(path):
        print(f"Adding path to sys.path: {path}")
        sys.path.append(path)
    else:
        print(f"Path does not exist, skipping: {path}")

# 打印最终的 sys.path 以确认结果
print("Current sys.path:")
for p in sys.path:
    print(p)

torch.set_default_device('cuda')

"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)

# USE_RULE_LAYER = True


def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.
    """
    logprobs = []
    for offset in range(mask.shape[0]-length+1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset:offset+length])
        logprob = logprob - torch.sum(mask[offset+length:])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs

def take_step(task, model, optimizer, train_step, train_history_logger, folder, task_name):
    """
    Runs a forward pass of the model on the ARC-AGI task.
    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """

    if (train_step == 0) or ((train_step + 1) % debugstep == 0):                   # 只跑一次
        idx_sample = 0                        # 第一个 train 样例

        # -- ① holes one-hot -----------------------------
        start = attr_registry.key_index("holes")
        dim   = 9                             # 若改成 5，改这里
        holes_vec = task.output_attr_tensor[idx_sample][:, start:start+dim]
        print("[CHK] holes row0-4 =", holes_vec[:5].tolist())

        # -- ② selector argmax ---------------------------
        sel_logits = model.rule_layer.selector(task.output_attr_tensor[idx_sample])
        sel = sel_logits.softmax(-1).argmax(-1)            # (N_obj,)
        print("[CHK] selector argmax =", sel[:5].tolist())

        # -- ③ color id of chosen op ---------------------
        raw_param = model.rule_layer.param_head(task.output_attr_tensor[idx_sample])
        #! K = model.rule_layer.K_ops
        K = 8
        # P = raw_param.shape[1] // K


        # P = 10                                       # 每 op 10 维
        P = model.rule_layer.n_params
        raw_param = raw_param.view(-1, K, P)         # (N_obj,K,10)
        color_logits = raw_param[
            torch.arange(len(sel), device=raw_param.device), sel, :
        ]                                            # (N_obj,10)
        color_ids = color_logits.softmax(-1).argmax(-1).cpu().tolist()
        # print("Predicted colors per obj:", color_ids[:5])
        print(f"[DBG {train_step}] selector idx per obj:", sel.tolist())
        print(f"[DBG {train_step}] color id per obj  :", color_ids[:5])



    optimizer.zero_grad()
    # optimizer.zero_grad()
    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits
    pred_idx = logits.argmax(dim=1)

    # ── 2) RuleLayer 仅对“网络输出”做后处理 ────────────
    rule_layer = getattr(model, "rule_layer", None)
    USE_RULE_LAYER = getattr(model, "use_rule", False)
    if USE_RULE_LAYER and rule_layer is not None and train_step >= reconstrucstep:
        # a) 取网络初步颜色索引
        # pred_idx = logits.argmax(dim=1)                    # (N,H,W)

        # b) 逐样例裁剪掩码并调用 RuleLayer
        # for idx in range(task.n_examples):
        n_out = len(task.output_obj_masks)   # 仅 train 样例有 output
        for idx in range(n_out):
            pred_out = pred_idx[idx, :, :, 1].clone()   # (H,W) long
            H, W = pred_out.shape

            raw_mask = task.output_obj_masks[idx]
            mask_i   = raw_mask[:, :H, :W] if raw_mask.ndim == 3 else raw_mask[:H, :W]

            patched = rule_layer(
                pred_out.clone(),                     # (H,W)
                task.output_attr_tensor[idx],              # (Ni,D)
                mask_i.to(pred_idx.device)                 # (Ni,H,W) or (H,W)
            )
            # 确保patched的值在有效范围内
            patched = torch.clamp(patched, 0, 9)

            mask_union = mask_i.any(dim=0)              # (H,W)
            pred_out[mask_union] = patched[mask_union]

            # 写回 (H,W,2) 张量的输出通道
            pred_idx[idx, :, :, 1] = pred_out

        # c) one-hot + straight-through 写回 logits
        # one_hot 得到 (N,H,W,2,C) → 重新排轴 (N,C,H,W,2)
        # 检查pred_idx的值是否在有效范围内
        max_val = pred_idx.max().item()
        min_val = pred_idx.min().item()
        num_classes = logits.shape[1]

        if max_val >= num_classes or min_val < 0:
            print(f"Warning: pred_idx values out of range! min={min_val}, max={max_val}, num_classes={num_classes}")
            # 将超出范围的值钳制到有效范围
            pred_idx = torch.clamp(pred_idx, 0, num_classes - 1)

        patched_onehot = torch.nn.functional.one_hot(
            pred_idx, logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        logits = logits + (patched_onehot - patched_onehot.detach())

    # ── logits 现已包含 RuleLayer 修改，下方重构误差照旧 ──
    else:
        # 若<100步，仅用原 logits 预测，RuleLayer 先不介入
        pred_idx = logits.argmax(dim=1)   # 供后面简单 CE


    # Compute the total KL loss
    total_KL = 0
    for KL_amount in KL_amounts:
        total_KL = total_KL + torch.sum(KL_amount)

    # Compute the reconstruction error
    reconstruction_error = 0
    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            # Determine whether the grid size is already known.
            # If not, there is an extra term in the reconstruction error, corresponding to
            # the probability of reconstructing the correct grid size.
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            if grid_size_uncertain:
                coefficient = 0.01**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
            problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
            # Account for probability of getting right grid size, if grid size is not known
            if grid_size_uncertain:
                x_log_partitions = []
                y_log_partitions = []
                for length in range(1, x_mask.shape[1]+1):
                    x_log_partitions.append(mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                for length in range(1, y_mask.shape[1]+1):
                    y_log_partitions.append(mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            # Given that we have the correct grid size, get the reconstruction error of getting the colors right
            logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition  # given the correct grid size,
                    logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                    target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y
                    logprob = logprob - 1 * ( torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')  ) # calculate the error for the colors.
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient  # Aggregate for all possible grid sizes
            reconstruction_error = reconstruction_error - logprob

    # ---------- sparsity penalty ----------
    if USE_RULE_LAYER and rule_layer is not None:
        # 以首样例的属性张量估算稀疏度
        lam_sched = 0.0 if train_step < 150 else \
                    3e-4 * min(1.0, (train_step-150)/450)
        sparsity_penalty = lam_sched * rule_layer.selector(
            task.output_attr_tensor[0]).abs().mean()
    else:
        sparsity_penalty = 0.0

    #  4) 重新计算 reconstruction loss
    # ────────────────────────────────────────────────
    # ground-truth 颜色索引

    # ---------- reconstruction error on output frame ----------
    # logits_out  = logits[..., 1]                     # (N,C,H,W)
    # target_idx  = task.problem[:, :, :, 1].to(logits.device)  # (N,H,W)
    # reconstruction_error = torch.nn.functional.cross_entropy(
    #         logits_out, target_idx, reduction='sum')


    if train_step < reconstrucstep:         gamma, beta, lam = 15, 0.51, 0.0
    elif train_step < 800:       # linear anneal
        frac  = (train_step)/1110
        gamma = 10
        beta  = 1  + 0.5*frac
        lam   = (5e-4 ) * frac
    else:
        gamma = 10
        beta  = 2
        lam   = 1e-3

    loss = gamma * reconstruction_error + beta * total_KL + lam * sparsity_penalty



    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # debug_train_predictions(
    #     task, logits, pred_idx, train_step,
    #     folder=task_name + '/',  # 需要传入folder参数
    #     task_name=task_name,     # 需要传入task_name参数
    #     rule_layer=rule_layer,
    #     USE_RULE_LAYER=USE_RULE_LAYER
    )

    if (train_step+1) % debugstep == 0:
        print(f"\n=== Train Examples Debug at Step {train_step+1} ===")

        with torch.no_grad():
            for train_idx in range(task.n_train):
                # 原始网络预测（规则应用前）
                raw_pred = logits[train_idx, :, :, :, 1].argmax(dim=0)
                # 最终预测（规则应用后）
                final_pred = pred_idx[train_idx, :, :, 1]
                # 目标答案
                target = task.problem[train_idx, :, :, 1].to(logits.device)

                # 计算准确率
                raw_acc = (raw_pred == target).float().mean().item()
                final_acc = (final_pred == target).float().mean().item()

                print(f"  Train {train_idx}: Raw={raw_acc:.3f}, Final={final_acc:.3f}, Δ={final_acc-raw_acc:+.3f}")

                # 显示颜色差异统计
                if raw_acc != final_acc:
                    raw_colors = torch.unique(raw_pred).cpu().tolist()
                    final_colors = torch.unique(final_pred).cpu().tolist()
                    target_colors = torch.unique(target).cpu().tolist()
                    print(f"    Raw colors: {raw_colors}")
                    print(f"    Final colors: {final_colors}")
                    print(f"    Target colors: {target_colors}")

        # 规则分析
        rule_layer = getattr(model, "rule_layer", None)
        USE_RULE_LAYER = getattr(model, "use_rule", False)
        if USE_RULE_LAYER and rule_layer is not None and train_step >= reconstrucstep:
            print(f"  === Rule Layer Analysis ===")
            with torch.no_grad():
                for idx in range(min(task.n_train, len(task.output_attr_tensor))):
                    attr_tensor = task.output_attr_tensor[idx].to('cuda')
                    if attr_tensor.shape[0] > 0:
                        # 规则选择概率
                        sel_probs = rule_layer.selector(attr_tensor).softmax(-1)
                        chosen_ops = sel_probs.argmax(-1).cpu().tolist()
                        max_probs = sel_probs.max(-1)[0].cpu().tolist()

                        print(f"    Train {idx} Objects: {len(chosen_ops)}")
                        print(f"    Chosen ops: {chosen_ops}")
                        print(f"    Confidence: {[f'{p:.2f}' for p in max_probs]}")

                        # 颜色预测
                        if hasattr(rule_layer, 'param_head'):
                            param_logits = rule_layer.param_head(attr_tensor)
                            if param_logits.shape[-1] > 0:
                                color_probs = param_logits[:, 0].softmax(-1)
                                pred_colors = color_probs.argmax(-1).cpu().tolist()
                                print(f"    Pred colors: {pred_colors}")


    # ──────────── Debug & 可视化 (每 debugstep step) ───────────────
    logits_before_ST = logits.clone()
    if (train_step > 5) and ((train_step+1 ) % debugstep == 0) :
        # 1) 取训练样例 idx=0 的网络原输出 & 规则修正后
        raw_out   = logits_before_ST[0, :, :, 1].argmax(0)   # (H,W)
        patched   = logits[0, :, :, 1].argmax(0)             # (H,W)

        # 2) 直接画两张图到 png
        import matplotlib.pyplot as plt, torchvision
        fig, ax = plt.subplots(1, 2, figsize=(4,2))
        ax[0].imshow(raw_out.cpu());     ax[0].set_title("raw")
        ax[1].imshow(patched.cpu());     ax[1].set_title("patched")
        for a in ax: a.axis("off")
        plt.tight_layout()
        debug_fname = (folder + f"{task_name}_trainPred_at_{train_step+1}.png")
        plt.savefig(debug_fname); plt.close()

        # 3) 打印 RuleLayer selector 与颜色 logits
        if USE_RULE_LAYER and model.rule_layer is not None:
            sel = model.rule_layer.selector(
                    task.output_attr_tensor[0]).softmax(-1).argmax(-1)
            col = model.rule_layer.param_head(
                    task.output_attr_tensor[0])[:,0].softmax(-1).argmax(-1)
            print(f" color logits shape:", model.rule_layer.param_head(
                task.output_attr_tensor[0]).shape)
            print(f"[DBG {train_step}] selector idx per obj:", sel.tolist())
            print(f"[DBG {train_step}] color pred per obj :", col.tolist())

        # 4) 现有工具：直接把 logger sample 图也存一下
        visualization.plot_solution(
            train_history_logger,
            fname = folder + f"{task_name}_at_{train_step+1}_steps.png")


    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             loss)



if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train ARC model with checkpoint support')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint path to resume from')
    parser.add_argument('--task_name', type=str, default='0a2355a6', help='Task name to train')
    parser.add_argument('--save_steps', nargs='+', type=int, default=[2,400, 800, 1200, 1600, 2000],
                       help='Steps at which to save checkpoints')
    args = parser.parse_args()

    start_time = time.time()
    torch.set_default_device('cuda')
    task_nums = list(range(1000))
    split = "training"  # "training", "evaluation, or "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Preprocess all tasks, make models, optimizers, and loggers. Make plots.
    # tasks = preprocessing.preprocess_tasks(split, task_nums)

    task_name = args.task_name
    folder = task_name + '/'
    print('Performing a training run on task', task_name,
          'and placing the results in', folder)
    os.makedirs(folder, exist_ok=True)

    task = preprocessing.preprocess_tasks(split, [task_name])[0]
    tasks = [task]

    models = []
    optimizers = []
    train_history_loggers = []
    for task in tasks:
        model = arc_compressor.ARCCompressor(task)
        models.append(model)


        USE_RULE_LAYER = True        # 改 False → 彻底关闭规则层
        if USE_RULE_LAYER:
            attr_dim   = build_attr_tensor(task.input_obj_dicts[0]).shape[1]
            rule_layer = SparseRuleLayer(attr_dim, n_colors=task.n_color_channels, K_ops=8, temp=1.0).to(device)
            optimizer  = torch.optim.Adam(
                model.weights_list + list(rule_layer.parameters()),
                lr=0.01
            )
        else:
            rule_layer = None
            optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))

        # optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))

        model.rule_layer = rule_layer
        model.use_rule   = USE_RULE_LAYER

        optimizers.append(optimizer)
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

        # ── 在每个 task 训练完、进入下一个 task 前加 ─────────────────



    # Get the solution hashes so that we can check for correctness
    true_solution_hashes = [task.solution_hash for task in tasks]

    folder = task_name + '/'

    # Train the models one by one
    for i, (task, model, optimizer, train_history_logger) in enumerate(zip(tasks, models, optimizers, train_history_loggers)):
        n_iterations = maxsteps
        start_step = 0

        # 检查是否需要从检查点恢复
        if args.resume or args.checkpoint:
            if args.checkpoint:
                checkpoint_path = args.checkpoint
            else:
                checkpoint_path = find_latest_checkpoint(folder, task_name)

            if checkpoint_path:
                start_step = load_checkpoint(checkpoint_path, model, optimizer, device)
                print(f"Resuming training from step {start_step}")
            else:
                print("No checkpoint found, starting from beginning")

        print(f"\nTraining task {i+1}/{len(tasks)}: {task_name} (from step {start_step})")

        # 创建tqdm进度条
        pbar = tqdm(range(start_step, n_iterations), desc=f"Training {task_name}",
                   unit="step", ncols=100, leave=True, initial=start_step, total=n_iterations)

        for train_step in pbar:
            take_step(task, model, optimizer, train_step, train_history_logger,folder, task_name)

            # 更新进度条显示信息
            if train_step % 10 == 0:  # 每10步更新一次显示
                # 获取最新的损失信息
                last_loss = getattr(train_history_logger, 'losses', [0])[-1] if hasattr(train_history_logger, 'losses') and train_history_logger.losses else 0
                pbar.set_postfix({
                    'step': train_step + 1,
                    'loss': f"{last_loss:.4f}" if isinstance(last_loss, (int, float)) else "N/A"
                })

            # 在指定步数保存检查点
            if (train_step + 1) in args.save_steps:
                # pass
                save_checkpoint(model, optimizer, train_step + 1, task_name, folder)

            if (train_step+1) % debugstep == 0:
                visualization.plot_solution(train_history_logger,
                    fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.png')
                # visualization.plot_solution(train_history_logger,
                #     fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.pdf')

        pbar.close()  # 确保进度条正确关闭

        visualization.plot_solution(train_history_logger)
        solution_selection.save_predictions(train_history_loggers[:i+1])
        solution_selection.plot_accuracy(true_solution_hashes)

        if USE_RULE_LAYER:
                print(f"\n[DEBUG]  Rule inspection for task {task_name}")

                # 取第一条样例（idx=0），也可以随机取
                attr0  = task.input_attr_tensor[0].to(device)     # (N_obj,D)
                masks0 = task.input_obj_masks[0].to(device)       # (N_obj,H,W)

                with torch.no_grad():
                    # selector logits → 概率
                    sel_logits = model.rule_layer.selector(attr0)          # (N_obj,K)
                    sel_probs  = sel_logits.softmax(dim=-1)
                    chosen_op  = sel_probs.argmax(dim=-1).cpu().tolist()   # 每对象选哪算子
                    print("Chosen op indices:", chosen_op)                 # 0..K-1

                    # 颜色 logits （第 0 参数）
                    param_logits = model.rule_layer.param_head(attr0)
                    param_logits = param_logits.view(attr0.shape[0], model.rule_layer.K, model.rule_layer.n_params)
                    # 获取选中操作的颜色参数
                    sel_indices = sel_probs.argmax(dim=-1)
                    colors = param_logits[torch.arange(len(sel_indices)), sel_indices, 0]
                    colors = colors.softmax(-1).argmax(-1).cpu().tolist()
                    print("Predicted colors per object:", colors)


        if USE_RULE_LAYER:
            attr0  = task.input_attr_tensor[0].to(device)
            with torch.no_grad():
                probs = model.rule_layer.selector(attr0).softmax(-1)
                print("Chosen op idx per object:", probs.argmax(-1).tolist())
                color_logits = model.rule_layer.param_head(attr0)[:, 0]
                colors = color_logits.softmax(-1).argmax(-1).tolist()
                print("Predicted colors:", colors)

    # Write down how long it all took
    with open('timing_result.txt', 'w') as f:
        f.write("Time elapsed in seconds: " + str(time.time() - start_time))
