## 《Evidential Deep Learning to Quantify Classification Uncertainty》示例代码教学文档（PyTorch 版）

> 对应仓库：`pytorch-classification-uncertainty-master`

---

### 1. 论文与代码的总体关系

#### 1.1 论文在解决什么问题？

传统分类神经网络的做法是：

- 最后一层输出一个向量 `z`（logits），经 过 `softmax(z)` 得到类别概率 `p`；
- 用 `CrossEntropyLoss`（交叉熵）训练，让正确类别的概率最大化。

**问题**：i

- Softmax 强制所有输入都“属于某个已知类别”，即使这个输入其实是**分布外样本（Out-of-Distribution, OOD）**；
- 在 OOD 或极端样本上，网络依然会给出**很高的概率**，却没有“不确定”这一选项。

**论文核心想法**：

- 不直接输出概率，而是输出**每个类别的证据（evidence）**： \( e_k \ge 0 \)；
- 通过 \( \alpha_k = e_k + 1 \) 构造 Dirichlet 分布 \( Dir(\alpha) \)，把网络预测看成对“类别概率分布”的主观意见；
- 使用新的损失函数（3 种变体：Eq.3, Eq.4, Eq.5），在拟合数据的同时，引导：
  - 对**正确样本**：证据多、不确定性低；
  - 对**错误样本或 OOD 样本**：证据少、不确定性高。

这样就能在保持分类性能的同时，为每个预测提供一个**不确定性估计**。

#### 1.2 这个 PyTorch 示例在做什么？

本仓库用 MNIST 手写数字分类作为示例，实现了论文中 Evidential Deep Learning（EDL） 的核心思想：

- 模型：一个简单的 `LeNet` 网络（两个卷积层 + 两个全连接层）
- 两种训练模式：
  - **标准模式**：普通 softmax + 交叉熵。
  - **不确定性模式**：输出证据，构造 Dirichlet 分布，用论文中的三种 Evidential Loss：
    - `edl_mse_loss`：Expected Mean Square Error（论文 Eq.5）；
    - `edl_digamma_loss`：Expected Cross Entropy（论文 Eq.4）；
    - `edl_log_loss`：Negative Log of Expected Likelihood（论文 Eq.3）。
- 实验：
  - 旋转一张 MNIST 中的“数字 1”，查看不同旋转角度下的预测标签、概率、不确定性；
  - 比较 in-distribution 图像（digit one）和 完全无关的 Yoda 图片上的不确定性表现。

接下来我们按**文件结构 → 关键函数 → 对应论文公式**的顺序来讲解。

---

### 2. 项目结构与文件角色

仓库主要文件：

- `main.py`：命令行入口，负责解析参数，决定训练/测试/展示数据，并选择损失函数；
- `data.py`：下载并构造 MNIST 数据集和 `DataLoader`；
- `lenet.py`：定义 LeNet 模型结构；
- `losses.py`：实现 Evidential Deep Learning 的三种损失和 KL 正则等；
- `train.py`：训练循环，支持标准 CE 和 Evidential Loss；
- `test.py`：测试脚本，包括旋转实验和单图测试 + 可视化；
- `helpers.py`：辅助函数（CUDA 设备、one-hot 编码、图片旋转）；
- `README.md`：英文说明与结果图示；
- `data/`、`results/`、`images/`：输入图片和输出结果存放目录。

学习建议顺序：

1. 先看 `main.py`，理解整个流程；
2. 再看 `lenet.py` 和 `data.py`，了解模型和数据；
3. 深入 `losses.py`，对照论文公式理解 Evidential Loss；
4. 看 `train.py` 如何调用这些 loss，并如何计算证据/不确定性；
5. 最后看 `test.py`，理解旋转实验与可视化；
6. 配合 `README.md` 中的图片理解实验现象。

---

### 3. `main.py`：程序入口与模式切换

#### 3.1 命令行参数解析

`main()` 中使用 `argparse` 定义了 3 种互斥模式：

- `--train`：训练网络；
- `--test`：加载训练好的模型做旋转与单图测试；
- `--examples`：只展示几张 MNIST 样本并保存到 `images/examples.jpg`。

训练相关参数：

- `--epochs`：训练轮数；
- `--dropout`：是否在全连接层使用 dropout；
- `--uncertainty`：是否启用“不确定性建模”（即 Evidential 模式）；
- 当 `--uncertainty` 为真时，三个互斥选项：
  - `--mse`：使用 Expected MSE Loss（Eq.5）；
  - `--digamma`：使用 Expected Cross Entropy（Eq.4）；
  - `--log`：使用 Negative Log of Expected Likelihood（Eq.3）。

逻辑要点：

- 如果 `--uncertainty` 为真，但没指定 `--mse/--digamma/--log`，则报错；
- 如果 `--uncertainty` 为假，则直接采用普通 `nn.CrossEntropyLoss()`。

#### 3.2 训练流程（`--train`）

在 `elif args.train:` 分支中：

1. 设置训练轮数、是否使用不确定性、类别数：

   - `num_epochs = args.epochs`
   - `use_uncertainty = args.uncertainty`
   - `num_classes = 10`
2. 初始化模型：`model = LeNet(dropout=args.dropout)`；
3. 根据是否使用不确定性选择损失：

   - 若 `use_uncertainty`：
     - `criterion = edl_digamma_loss` / `edl_log_loss` / `edl_mse_loss`;
   - 否则：`criterion = nn.CrossEntropyLoss()`。
4. 优化器与学习率策略：

   - `optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.005)`;
   - `exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)`；
5. 将模型放到 GPU/CPU 对应设备：`device = get_device(); model.to(device)`；
6. 调用 `train_model(...)`：

   - 传入 `uncertainty=use_uncertainty`，训练循环内部会据此选择普通还是 evidential 逻辑；
   - `criterion` 已经是对应的损失函数对象或函数。
7. 训练完成后，保存 state dict 到不同文件：

   - 普通模式：`./results/model.pt`；
   - 不确定性 + MSE：`./results/model_uncertainty_mse.pt`；
   - 不确定性 + Digamma：`./results/model_uncertainty_digamma.pt`；
   - 不确定性 + Log：`./results/model_uncertainty_log.pt`。

这部分是把论文中的**三种损失函数**抽象成命令行可选项，训练细节全交给 `train_model` 和 `losses.py`。

#### 3.3 测试流程（`--test`）

在 `elif args.test:` 分支中：

1. 初始化模型和优化器：
   - `model = LeNet()`
   - `optimizer = Adam(model.parameters())`
2. 根据 `--uncertainty` 与具体损失类型加载对应权重：
   - 有不确定性：
     - `--digamma` ➜ `model_uncertainty_digamma.pt`；同时设置输出文件名 `rotate_uncertainty_digamma.jpg`；
     - `--log` ➜ `model_uncertainty_log.pt`；
     - `--mse` ➜ `model_uncertainty_mse.pt`；
   - 无不确定性：
     - 加载 `model.pt`，输出图片为 `rotate.jpg`。
3. 加载 state dict、设为 eval 模式；
4. 调用：
   - `rotating_image_classification(model, digit_one, filename, uncertainty=use_uncertainty)`；
   - `test_single_image(model, './data/one.jpg', uncertainty=use_uncertainty)`；
   - `test_single_image(model, './data/yoda.jpg', uncertainty=use_uncertainty)`。

这正对应论文中：

- 对“旋转数字 1”进行实验，查看错误区域的不确定性变化；
- 对“Yoda 等明显 OOD 图像”进行测试，查看模型是否给出高不确定性。

---

### 4. `data.py`：数据加载

#### 4.1 MNIST 数据集

使用 `torchvision.datasets.MNIST`：

- `data_train`：训练集，`train=True`，自动下载到 `./data/mnist`；
- `data_val`：验证集，`train=False`；
- 变换：`transforms.ToTensor()`，将 28×28 灰度图转换到 `[1,28,28]` 张量，并归一化到 [0,1]。

#### 4.2 DataLoader

- `dataloader_train`：batch_size=1000，shuffle=True，用于训练；
- `dataloader_val`：batch_size=1000，用于验证；

封装成：

```python
dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}
```

#### 4.3 提取旋转实验用的数字 1

```python
digit_one, _ = data_val[5]
```

- 取验证集中第 6 张图片（索引 5），通常是一个数字“1”；
- 后面 `test.py` 用这个 `digit_one` 做旋转实验。

这一部分没有特别的论文细节，就是标准的 MNIST 配置。

---

### 5. `lenet.py`：网络结构与“证据输出”

#### 5.1 LeNet 结构

`LeNet` 是一个非常经典和简单的卷积网络：

- 输入：`1×28×28` 灰度图像；
- `conv1`: `Conv2d(1, 20, kernel_size=5)`；
- `conv2`: `Conv2d(20, 50, kernel_size=5)`；
- `fc1`: `Linear(20000, 500)`；
- `fc2`: `Linear(500, 10)`（10 类输出）。

前向传播：

1. 卷积 + 池化 + ReLU：
   - `x = F.relu(F.max_pool2d(self.conv1(x), 1))`
   - `x = F.relu(F.max_pool2d(self.conv2(x), 1))`
2. 展平：`x = x.view(x.size()[0], -1)`；
3. 全连接 + ReLU：`x = F.relu(self.fc1(x))`；
4. 可选 dropout：`if self.use_dropout: x = F.dropout(x, training=self.training)`；
5. 输出层：`x = self.fc2(x)`；

**重要点**：

- 这里**没有 softmax**，输出的是原始 logits；
- 在普通模式下，直接送入 `CrossEntropyLoss` 即可（内部带 log-softmax）；
- 在不确定性模式下，这些 logits 会通过 `relu_evidence` 转换为证据，再加 1 得到 Dirichlet 参数 \(\alpha\)。

和论文的对应：

- 论文中说“我们学习的是证据（evidence）而不是概率”，在代码中就是：
  - 网络最后一层输出的 `x` ➜ `evidence = ReLU(x)` ➜ `alpha = evidence + 1`。

---

### 6. `helpers.py`：辅助函数

三个简单实用的函数：

1. `get_device()`：自动选择 `cuda:0` 或 `cpu` 用作训练设备；
2. `one_hot_embedding(labels, num_classes=10)`：将整数标签转成 one-hot 向量，供 evidential loss 使用；
3. `rotate_img(x, deg)`：将 28×28 图像按照给定角度旋转（使用 `scipy.ndimage.rotate`），用于旋转实验。

其中，`one_hot_embedding` 与论文中 `y` 的 one-hot 表示直接对应。

---

### 7. `losses.py`：Evidential Loss 的核心实现

这是理解 EDL 的重心，与论文公式紧密对应。

#### 7.1 证据函数（Evidence）

```python
def relu_evidence(y):
    return F.relu(y)
```

- 将网络输出 `y` 经过 ReLU 截断负数：\( e_k = \max(0, y_k) \)；
- 然后在各个 loss 中统一做 `alpha = evidence + 1`：
  - 确保每个 \(\alpha_k \ge 1\)，即每个类别至少有 1 单位“基线证据”。

你可以把 `alpha` 理解为“对每个类别的伪计数（pseudo-count）”，类似贝叶斯中的先验 + 观察次数。

#### 7.2 Dirichlet 的 KL 散度：`kl_divergence`

```python
def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], ...)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl
```

数学上，这是 Dirichlet 分布之间 KL 散度的解析表达式，计算的是：

\[
\text{KL}(Dir(\alpha) \parallel Dir(\mathbf{1}))
\]

- `ones` 表示对称先验 Dirichlet(1, 1, ..., 1)；
- KL 项在损失中主要作用：
  - 约束预测不要过度自信；
  - 尤其对**预测错误**的样本，鼓励其证据向均匀先验收缩（即 \(\alpha_k \to 1\)），从而提高不确定性。

#### 7.3 MSE 型损失（Eq.5）：`loglikelihood_loss` 与 `mse_loss` / `edl_mse_loss`

##### 7.3.1 `loglikelihood_loss`

```python
def loglikelihood_loss(y, alpha, device=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood
```

对应论文 Eq.5：

- \(y\)：one-hot 标签（真实类别为 1，其余为 0）；
- \(\hat{p} = \alpha / S\)：Dirichlet 分布的期望概率；
- 第一项：
  - \(\| y - \hat{p} \|^2\)：偏差项（期望平方误差）；
- 第二项：
  - Dirichlet 引入的概率方差项，量化模型固有的不确定性；

两者和起来就是论文中所谓的 **Expected Mean Square Error**：

\[
\mathbb{E}_{p \sim Dir(\alpha)}[\|y - p\|^2]
\]

##### 7.3.2 `mse_loss` 与 KL annealing

```python
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = min(1.0, epoch_num / annealing_step)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div
```

解释：

- `annealing_coef`：随 epoch 增大，逐渐从 0 过渡到 1：
  - 前期训练更看重拟合数据（loglikelihood）；
  - 后期逐步增强 KL 正则，防止过拟合与过度自信；
- `kl_alpha = (alpha - 1) * (1 - y) + 1`：
  - 若类别为真实类别：`y_k=1`，则 `alpha'_k = 1`；
  - 若类别为非真实类别：`y_j=0`，则 `alpha'_j = alpha_j`；
  - 等价于**只对非真实类别的证据进行正则**，把它们往先验 Dir(1,1,...,1) 拉。

##### 7.3.3 `edl_mse_loss`

```python
def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss
```

- `output`：网络最后一层未经过 softmax 的输出；
- `target`：one-hot 标签（由 `one_hot_embedding` 生成）；
- 内部完成：`output → evidence → alpha`，然后调用上面的 `mse_loss`。

#### 7.4 Log 和 Digamma 型损失（Eq.3、Eq.4）：`edl_log_loss`、`edl_digamma_loss`

定义了一个通用函数 `edl_loss`：

```python
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1.0, epoch_num / annealing_step)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div
```

- 当 `func = torch.log` 时，对应负对数期望似然（Eq.3）；
- 当 `func = torch.digamma` 时，对应期望交叉熵（Eq.4）。

具体封装：

```python
def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    )
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    )
    return loss
```

和 MSE 版本相同的特点：

- 先把输出转成证据和 Dirichlet 参数；
- 再加上 KL 正则，并用逐步增大的系数（annealing）控制其权重。

---

### 8. `train.py`：训练循环与不确定性统计

#### 8.1 函数签名

```python
def train_model(
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    uncertainty=False,
):
```

- `uncertainty`：是否启用 Evidential 模式；
- `criterion`：在普通模式下是 `nn.CrossEntropyLoss`，在 EDL 模式下是 `edl_*_loss` 函数。

#### 8.2 训练/验证循环

标准 PyTorch 训练模式：

- 外层循环 `for epoch in range(num_epochs)`；
- 内层对 `phase in ["train", "val"]` 分别执行：
  - 设置 `model.train()` 或 `model.eval()`；
  - 迭代 `dataloaders[phase]` 得到 `(inputs, labels)`；
  - 前向传播，计算损失；
  - 若是训练相位则反向传播 + `optimizer.step()`；
  - 记录每个 epoch 的平均 loss 与 accuracy。

#### 8.3 普通模式 vs 不确定性模式

在 batch 内部前向阶段：

- **普通模式**：

  ```python
  outputs = model(inputs)
  _, preds = torch.max(outputs, 1)
  loss = criterion(outputs, labels)  # CrossEntropyLoss
  ```
- **不确定性模式**：

  ```python
  y = one_hot_embedding(labels, num_classes)
  y = y.to(device)
  outputs = model(inputs)
  _, preds = torch.max(outputs, 1)
  loss = criterion(outputs, y.float(), epoch, num_classes, 10, device)
  ```

  其中：

  - `criterion` 即 `edl_mse_loss` / `edl_log_loss` / `edl_digamma_loss`；
  - `10` 是 `annealing_step`，表示 KL 权重在前 10 个 epoch 内线性从 0 增加到 1；
  - 其余参数如 `epoch`、`num_classes` 交给 loss 内部使用。

#### 8.4 训练中对证据与不确定性的分析（对应论文实验）

在 `uncertainty=True` 时，代码还会计算一些额外指标：

```python
match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
acc = torch.mean(match)  # 当前 batch 准确率

# 证据与不确定性
evidence = relu_evidence(outputs)
alpha = evidence + 1
u = num_classes / torch.sum(alpha, dim=1, keepdim=True)  # 不确定性 u

total_evidence = torch.sum(evidence, 1, keepdim=True)
mean_evidence = torch.mean(total_evidence)
mean_evidence_succ = torch.sum(total_evidence * match) / torch.sum(match + 1e-20)
mean_evidence_fail = torch.sum(total_evidence * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)
```

含义：

- `u = K / S`：整体不确定性（S 是 sum(alpha)）；
  - 证据越多（S 越大），不确定性越低；
  - 证据越少（S 越小），不确定性越高；
- `mean_evidence_succ`：对**预测正确**的样本，其平均总证据；
- `mean_evidence_fail`：对**预测错误**的样本，其平均总证据；

论文中很多图（如“错误样本的证据小、不确定性高”）就是基于类似的统计。

---

### 9. `test.py`：旋转实验与单图不确定性可视化

`test.py` 定义了两个核心函数：

- `test_single_image(model, img_path, uncertainty=False, device=None)`；
- `rotating_image_classification(model, img, filename, uncertainty=False, ...)`。

#### 9.1 单张图片测试：`test_single_image`

流程：

1. 读取图片 `img_path`，转为灰度图 `L`，再 resize 到 `28×28`，转为 tensor；
2. 若 `uncertainty=True`：

   ```python
   output = model(img_variable)
   evidence = relu_evidence(output)
   alpha = evidence + 1
   uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
   prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
   ```

   - 概率：Dirichlet 的期望 \(\mathbb{E}[p_k] = \alpha_k / S\)；
   - 不确定性：\(u = K/S\)。
3. 若 `uncertainty=False`：

   ```python
   output = model(img_variable)
   prob = F.softmax(output, dim=1)
   ```
4. 打印结果：预测类别、概率、不确定性（若有）；
5. 画出图像 + 概率条形图，并保存到 `./results/{图片文件名}`。

对应 README 中“MNIST One Digit vs Random Yoda”那张图：

- 对 in-distribution 的 MNIST 数字 1：
  - 某个类别（比如 1）概率高；
  - 不确定性 \(\approx 0.15\) 左右；
- 对 OOD 的 Yoda 图片：
  - 概率更趋近均匀；
  - 不确定性 \(\approx 1.0\) 接近最大。

#### 9.2 旋转实验：`rotating_image_classification`

核心思想：

- 选定一个 digit one 图片，从 0° 到 180° 按固定步长旋转；
- 对每个旋转角度进行前向推理，记录：
  - 预测类别；
  - 各类概率；
  - 若启用不确定性，还记录 `u` 的变化；
- 将所有角度的图片拼接起来，并画出随角度变化的概率曲线和不确定性曲线。

关键片段（不确定性模式）：

```python
output = model(img_variable)
evidence = relu_evidence(output)
alpha = evidence + 1
uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
_, preds = torch.max(output, 1)
prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
classifications.append(preds[0].item())
lu.append(uncertainty.mean())
```

最后绘图：

- 上方子图：把不同角度的数字 1 横向拼接；
- 中间子图：用 table 的形式显示每个角度的预测类别序列；
- 下方子图：不同类别概率随旋转角度的曲线；
  - 若 `uncertainty=True`，再加上一条“不确定性曲线”。

对应 README 中的三张图：

- `rotate.jpg`：标准 softmax 网络，没有不确定性曲线；
- `rotate_uncertainty_mse.jpg`：MSE EDL 网络，显示在错误区域证据低、不确定性高；
- `rotate_uncertainty_digamma.jpg`、`rotate_uncertainty_log.jpg`：其他两个损失下的表现差异。

---

### 10. 实际运行与对照学习建议

#### 10.1 安装与运行

1. 安装依赖（参考 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

2. 查看命令行帮助：

```bash
python main.py --help
```

3. 训练普通 softmax 网络：

```bash
python main.py --train --dropout --epochs 10
```

4. 测试普通网络（旋转 + 单图）：

```bash
python main.py --test
```

会生成 `./results/rotate.jpg` 等文件。

5. 训练带不确定性的网络（以 MSE 为例）：

```bash
python main.py --train --dropout --uncertainty --mse --epochs 50
```

6. 测试带不确定性的网络：

```bash
python main.py --test --uncertainty --mse
```

生成 `rotate_uncertainty_mse.jpg`、`one.jpg`、`yoda.jpg` 等结果图。

#### 10.2 建议的阅读顺序

1. **对照论文公式与代码**：
   - 打开论文的 Eq.3 / Eq.4 / Eq.5；
   - 一边看 `losses.py` 中 `edl_mse_loss`, `edl_log_loss`, `edl_digamma_loss` 等实现，一边核对每一项来源；
2. **观察实验现象**：
   - 将 `rotate.jpg` 与 `rotate_uncertainty_*.jpg` 对比，关注：
     - 错误区域的概率是否变小？
     - 不确定性曲线是否明显升高？
3. **单图测试**：
   - 观察 `one.jpg` 与 `yoda.jpg` 的预测概率与不确定性差异；
   - 思考：对于你自己手画的图片、噪声图像、不同数据集图片，EDL 模型会给出怎样的不确定性？

#### 10.3 进一步的扩展思路

- 将本示例从 MNIST 迁移到 CIFAR-10，验证在更复杂数据上的表现；
- 替换网络结构为更深的 CNN（如 ResNet），保持 `losses.py` 不变；
- 在真实的开放集识别任务（Open Set Recognition）中接入 Evidential Loss，观察对 OOD 检测的影响；
- 研究不同证据函数（`relu_evidence` / `exp_evidence` / `softplus_evidence`）对训练稳定性与不确定性的影响。

---

### 11. 小结

- 这个示例工程完整实现了论文《Evidential Deep Learning to Quantify Classification Uncertainty》的核心思想，重点在于：
  - **输出证据而不是概率**，再通过 Dirichlet 分布表达主观信念；
  - 使用包含 KL 正则的 **三种 Evidential Loss**，让模型在错误/分布外样本上具有更高不确定性；
  - 通过旋转实验与 OOD 图片测试，直观展示 EDL 相比标准 softmax 的优势。
- 对代码完全理解后，你可以：
  - 将其应用到自己的数据集和任务上；
  - 尝试不同的 evidential 变体；
  - 与其他不确定性估计方法（如 Bayesian NN、MC Dropout）进行比较。

如果你希望对某个具体文件（例如 `losses.py`）做成**逐行中文注释版**，可以在 IDE 里 @ 对应文件，我可以再帮你写一份更细致的“代码旁注”文档。
