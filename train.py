import time

import numpy as np
import torch
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
from utils.attr_registry import build_attr_tensor

import os
import sys

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

USE_RULE_LAYER = True


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

def take_step(task, model, optimizer, train_step, train_history_logger):
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

    optimizer.zero_grad()

    # rule_layer = getattr(model, "rule_layer", None)
    # USE_RULE_LAYER = getattr(model, "use_rule", False)

    # ---------- RuleLayer 叠加 ----------
    rule_layer = getattr(model, "rule_layer", None)
    USE_RULE_LAYER = getattr(model, "use_rule", False)

    if USE_RULE_LAYER and rule_layer is not None:
        canvas       = task.problem[:, :, :, 0]        # (N,H,W) 颜色索引
        union_masks  = task.input_obj_masks            # List[Tensor]
        attr_tensors = task.input_attr_tensor          # List[Tensor]

        for idx in range(task.n_examples):
            # 当前样例真实网格大小
            H, W = canvas[idx].shape                   # (H,W)

            # 取并裁剪掩码  → mask_i shape = (Ni,H,W)  或 (H,W)
            raw_mask = union_masks[idx]
            if raw_mask.ndim == 3:                     # (Ni,30,30)
                mask_i = raw_mask[:, :H, :W]
            else:                                      # (30,30)
                mask_i = raw_mask[:H, :W]

            # 调用 RuleLayer
            patched = rule_layer(
                canvas[idx].clone(),                   # (H,W)
                attr_tensors[idx],                     # (Ni,D)
                mask_i.to(canvas.device)               # 对齐设备
            )

            # 前景区域换色
            mask_union = mask_i.any(dim=0)             # (H,W)
            canvas[idx][mask_union] = patched[mask_union]

        # 写回 Task.problem (仅输入帧)
        task.problem[:, :, :, 0] = canvas
    # ---------- RuleLayer 结束 ----------


    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits

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
                    logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')  # calculate the error for the colors.
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient  # Aggregate for all possible grid sizes
            reconstruction_error = reconstruction_error - logprob

    # loss = total_KL + 10*reconstruction_error
    if USE_RULE_LAYER:
        sparsity_penalty = 1e-4 * rule_layer.selector(
            attr_tensors[0]).abs().mean()
    else:
        sparsity_penalty = 0.0

    # loss = 2*reconstruction_error + total_KL + sparsity_penalty

    if train_step < 990:
        gamma = 13
        beta  = 1.0
        lam   = 1e-4
    elif train_step < 1500:
        gamma = 5
        beta  = 2.0
        lam   = 3e-4
    else:
        gamma = 2
        beta  = 4.0
        lam   = 5e-4

    loss = gamma * reconstruction_error + beta * total_KL + lam * sparsity_penalty



    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

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
    start_time = time.time()
    torch.set_default_device('cuda')
    task_nums = list(range(1000))
    split = "training"  # "training", "evaluation, or "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"


    # Preprocess all tasks, make models, optimizers, and loggers. Make plots.
    # tasks = preprocessing.preprocess_tasks(split, task_nums)


    task_name = '0a2355a6'
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
            rule_layer = SparseRuleLayer(attr_dim, K_ops=8, temp=1.0).to(device)
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
        n_iterations = 2000
        print(f"\nTraining task {i+1}/{len(tasks)}: {task_name}")

        # 创建tqdm进度条
        pbar = tqdm(range(n_iterations), desc=f"Training {task_name}",
                   unit="step", ncols=100, leave=True)

        for train_step in pbar:
            take_step(task, model, optimizer, train_step, train_history_logger)

            # 更新进度条显示信息
            if train_step % 10 == 0:  # 每10步更新一次显示
                # 获取最新的损失信息
                last_loss = getattr(train_history_logger, 'losses', [0])[-1] if hasattr(train_history_logger, 'losses') and train_history_logger.losses else 0
                pbar.set_postfix({
                    'step': train_step + 1,
                    'loss': f"{last_loss:.4f}" if isinstance(last_loss, (int, float)) else "N/A"
                })

            if (train_step+1) % 99 == 0:
                visualization.plot_solution(train_history_logger,
                    fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.png')
                visualization.plot_solution(train_history_logger,
                    fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.pdf')

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
                    param_logits = model.rule_layer.param_head(attr0)[:,0] # (N_obj,K)
                    colors = param_logits.softmax(-1).argmax(-1).cpu().tolist()
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
