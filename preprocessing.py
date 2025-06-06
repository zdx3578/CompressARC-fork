import json
import numpy as np
import torch
import multitensor_systems

np.random.seed(0)
torch.manual_seed(0)

class Task:
    """
    A class that helps deal with task-specific operations such as preprocessing,
    grid shape handling, solution processing, etc. Sets up the task-specific
    multitensor system to be used to construct the network.
    """
    def __init__(self, task_name, problem, solution):
        self.task_name = task_name
        self.n_train = len(problem['train'])
        self.n_test = len(problem['test'])
        self.n_examples = self.n_train + self.n_test
        self.unprocessed_problem = problem

        self.input_obj_dicts,  self.output_obj_dicts  = [], []
        self.input_obj_masks,  self.output_obj_masks  = [], []
        self.input_obj_attrs,  self.output_obj_attrs  = [], []

        self.input_attr_tensor, self.output_attr_tensor = [], []


        self.n_obj_channels = 1

        self.shapes = self._collect_problem_shapes(problem)
        self._predict_solution_shapes()
        self._construct_multitensor_system(problem)
        self._compute_mask()
        self._create_problem_tensor(problem)

        self.solution = self._create_solution_tensor(solution) if solution else None
        if solution is None:
            self.solution_hash = None





    def _collect_problem_shapes(self, problem):
        """
        Extract input/output shapes for each example.
        """
        shapes = []
        for split_name in ['train', 'test']:
            for example in problem[split_name]:
                in_shape = list(np.array(example['input']).shape)
                out_shape = list(np.array(example['output']).shape) if 'output' in example else None
                shapes.append([in_shape, out_shape])
        return shapes

    def _predict_solution_shapes(self):
        """
        Predict output shapes when not explicitly provided.
        """
        self.in_out_same_size = all(tuple(inp) == tuple(out) for inp, out in self.shapes[:self.n_train])
        self.all_in_same_size = len({tuple(shape[0]) for shape in self.shapes}) == 1
        self.all_out_same_size = len({tuple(shape[1]) for shape in self.shapes if shape[1]}) == 1

        if self.in_out_same_size:
            for shape in self.shapes[self.n_train:]:
                shape[1] = shape[0]
        elif self.all_out_same_size:
            default_shape = self.shapes[0][1]
            for shape in self.shapes[self.n_train:]:
                shape[1] = default_shape
        else:
            max_x, max_y = self._get_max_dimensions()
            for shape in self.shapes[self.n_train:]:
                shape[1] = [max_x, max_y]

    def _get_max_dimensions(self):
        max_x, max_y = 0, 0
        for in_out_pair in self.shapes:
            for shape in in_out_pair:
                if shape:
                    max_x = max(max_x, shape[0])
                    max_y = max(max_y, shape[1])
        return max_x, max_y

    def _construct_multitensor_system(self, problem):
        """
        Build tensor system with appropriate sizes.
        """
        self.n_x = max(shape[i][0] for shape in self.shapes for i in range(2))
        self.n_y = max(shape[i][1] for shape in self.shapes for i in range(2))

        colors = {color
                  for split in ['train', 'test']
                  for example in problem[split]
                  for grid in [example['input'], example.get('output', [])]
                  for row in grid
                  for color in row}
        colors.add(0)  # Always include black as background

        self.colors = list(sorted(colors))
        self.n_colors = len(self.colors) - 1
        self.n_color_channels = self.n_colors + 1   # 颜色 + bg
        self.total_channels  = self.n_color_channels + self.n_obj_channels

        self.multitensor_system = multitensor_systems.MultiTensorSystem(
            self.n_examples, self.n_colors, self.n_x, self.n_y, self
        )





    def _create_problem_tensor(self, problem):
        """
        Build self.problem (C,H,W,2)  +  缓存 input/output 对象信息
        """
        # --- ① 预分配张量 ---
        self.problem = np.zeros(
            (self.n_examples, self.total_channels, self.n_x, self.n_y, 2), dtype=np.int8
        )

        # --- ② 遍历样例 ---
        for subsplit, n_examples in [('train', self.n_train), ('test', self.n_test)]:
            for example_num, example in enumerate(problem[subsplit]):
                new_idx = example_num if subsplit == 'train' else self.n_train + example_num

                for mode in ('input', 'output'):
                    # 测试集没有 output
                    if subsplit == 'test' and mode == 'output':
                        continue

                    # ---------- 对象提取 ----------
                    original_grid = example.get(
                        mode, np.zeros(self.shapes[new_idx][1])
                    )                               # 2-D list/ndarray (H,W)
                    # Debug: show grid if necessary. Commented out to reduce
                    # verbosity during preprocessing.
                    # print(original_grid)
                    if isinstance(original_grid, list):
                        original_grid = np.array(original_grid)
                    if original_grid.ndim != 2:
                        raise ValueError(f"Expected 2D grid, got {original_grid.ndim}D array.")
                    if original_grid.shape[0] == 0 or original_grid.shape[1] == 0:
                        raise ValueError(f"Grid shape {original_grid.shape} is invalid, must be non-empty.")

                    from utils.object_adapter import extract_objects_from_grid
                    obj_d, obj_m, obj_a = extract_objects_from_grid(
                        grid=np.array(original_grid),  # 必须二维
                        pair_id=new_idx,
                        in_or_out=mode
                    )
                    if mode == 'input':
                        self.input_obj_dicts.append(obj_d)
                        self.input_obj_masks.append(obj_m)
                        self.input_obj_attrs.append(obj_a)
                    else:
                        self.output_obj_dicts.append(obj_d)
                        self.output_obj_masks.append(obj_m)
                        self.output_obj_attrs.append(obj_a)
                    # --------------------------------

                    from utils.attr_registry import build_attr_tensor
                    # ...
                    obj_attrs_tensor = build_attr_tensor(obj_d)   # (N,D)
                    if mode=='input':
                        self.input_attr_tensor.append(obj_attrs_tensor)
                    else:
                        self.output_attr_tensor.append(obj_attrs_tensor)

                    # Debug: inspect hole features for the first few objects of
                    # the first training example.
                    if subsplit == 'train' and example_num == 0 and len(obj_d) > 0:
                        from utils.attr_registry import key_index
                        h_start = key_index('holes')
                        h_vec = obj_attrs_tensor[:, h_start:h_start+9]
                        print(f"[DEBUG] Example {new_idx} {mode} holes one-hot:", h_vec[:5].tolist())



                    # ---------- 原 pipeline ----------
                    grid_tensor = self._create_grid_tensor(original_grid)   # (C0,H,W)
                    H, W = grid_tensor.shape[1:]

                    # ---------- union mask ----------
                    if obj_m.shape[0] > 0:
                        union_mask = obj_m.any(dim=0).to(torch.int8).cpu().numpy()
                    else:
                        union_mask = np.zeros((H, W), dtype=np.int8)

                    union_mask = union_mask[:H, :W]

                    mode_num = 0 if mode == 'input' else 1

                    # ---------- 写入 ----------
                    self.problem[new_idx,
                                :self.n_color_channels,
                                :H, :W,
                                mode_num] = grid_tensor

                    self.problem[new_idx,
                                self.n_color_channels,   # 对象掩码通道
                                :H, :W,
                                mode_num] = union_mask



        color_tensor = self.problem[:, :self.n_color_channels, ...]
        color_idx = np.argmax(color_tensor, axis=1)          # (N,H,W,2)
        self.problem = torch.from_numpy(color_idx).to(torch.get_default_device())



    def _create_grid_tensor(self, grid):
        return np.array([
            [[1 if self.colors.index(color) == ref_color else 0
              for color in row]
             for row in grid]
            for ref_color in range(self.n_colors + 1)
        ])

    def _create_solution_tensor(self, solution):
        """
        Convert solution grids to tensors for crossentropy evaluation.
        """
        solution_tensor = np.zeros((self.n_test, self.n_colors + 1, self.n_x, self.n_y))
        solution_tuple = ()

        for example_num, grid in enumerate(solution):
            solution_tuple += (tuple(map(tuple, grid)),)
            grid_tensor = self._create_grid_tensor(grid)
            # unfortunately sometimes the solution tensor will be bigger than (n_x, n_y), and in these cases
            # we'll never get the solution.
            min_x, min_y = min(grid_tensor.shape[1], self.n_x), min(grid_tensor.shape[2], self.n_y)
            solution_tensor[example_num, :, :min_x, :min_y] = grid_tensor[:, :min_x, :min_y]

        self.solution_hash = hash(solution_tuple)
        return torch.from_numpy(np.argmax(solution_tensor, axis=1)).to(torch.get_default_device())

    def _compute_mask(self):
        """
        Compute masks for activations and cross-entropies.
        """
        self.masks = np.zeros((self.n_examples, self.n_x, self.n_y, 2))

        for example_num, (in_shape, out_shape) in enumerate(self.shapes):
            for mode_num, shape in enumerate([in_shape, out_shape]):
                if shape:
                    x_mask = np.arange(self.n_x) < shape[0]
                    y_mask = np.arange(self.n_y) < shape[1]
                    self.masks[example_num, :, :, mode_num] = np.outer(x_mask, y_mask)

        self.masks = torch.from_numpy(self.masks).to(torch.get_default_dtype()).to(torch.get_default_device())


# def preprocess_tasks(split, task_nums_or_task_names):
#     """
#     Preprocess tasks by loading problems and solutions.
#     """
#     with open(f'dataset/arc-agi_{split}_challenges.json', 'r') as f:
#         problems = json.load(f)

#     solutions = None if split == "test" else json.load(open(f'dataset/arc-agi_{split}_solutions.json'))

#     task_names = list(problems.keys())

#     return [Task(task_name,
#                  problems[task_name],
#                  solutions.get(task_name) if solutions else None)
#             for task_name in task_names
#             if task_name in task_nums_or_task_names or task_names.index(task_name) in task_nums_or_task_names]


def preprocess_tasks(split, task_nums_or_task_names):
    """
    Preprocess tasks by loading problems and solutions.
    """
    with open(f'dataset/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)

    solutions = None if split == "test" else json.load(open(f'dataset/arc-agi_{split}_solutions.json'))

    print(f"Loaded {len(problems)} tasks from {split} split.")

    task_names = list(problems.keys())
    selected_task_names = []

    # 将输入统一为列表格式
    if isinstance(task_nums_or_task_names, (str, int)):
        task_nums_or_task_names = [task_nums_or_task_names]

    print(f"Searching for tasks: {task_nums_or_task_names}")

    # 收集有效的任务名称
    for item in task_nums_or_task_names:
        if isinstance(item, str):
            if item in task_names:
                selected_task_names.append(item)
            else:
                print(f"Warning: Task '{item}' not found in {split} split")
        elif isinstance(item, int):
            if 0 <= item < len(task_names):
                selected_task_names.append(task_names[item])
            else:
                print(f"Warning: Task index {item} out of range (0-{len(task_names)-1})")

    if not selected_task_names:
        print(f"Error: No valid tasks found from input {task_nums_or_task_names}")
        print(f"Available tasks: {task_names[:5]}... (total: {len(task_names)})")
        return []

    print(f"Selected tasks: {selected_task_names}")

    return [Task(task_name,
                problems[task_name],
                solutions.get(task_name) if solutions else None)
           for task_name in selected_task_names]