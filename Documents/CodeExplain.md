# 项目代码功能模块解读
_Exported on 2025/11/11 at GMT+8 22:39:02 from Cursor (2.0.69)_

---

**User**

请对项目代码进行解读，输出一份详细的代码功能模块说明和关键算法模块解释，方便我学习该项目代码

---

**Cursor**

我来帮你解读这个项目的代码。让我先读取主要的代码文件。

让我继续读取更多核心代码文件：

现在我已经读取了所有核心代码文件，让我为你整理一份详细的项目代码解读文档：

# XVR 项目代码详细解读

## 一、项目概述

**XVR (X-ray to Volume Registration)** 是一个基于 PyTorch 的 2D/3D 医学图像配准系统，核心功能是将 2D X射线图像与 3D CT/MR 体积数据进行精确配准。

**核心特点：**
- 使用深度学习的位姿回归模型进行初始化
- 结合可微分渲染进行迭代优化
- 5分钟内完成患者特定模型训练
- 达到亚毫米级配准精度

---

## 二、核心功能模块

### 2.1 CLI 命令行接口 (`cli.py`)

**功能：** 提供统一的命令行入口

**主要命令：**
- `train` - 从头训练位姿回归模型
- `finetune` - 微调预训练模型
- `register` - 执行 2D/3D 配准（三种初始化方式）
- `animate` - 生成优化轨迹动画
- `dcm2nii` - DICOM 格式转换

```python
# 命令层级结构
cli (主入口)
├── train       # 训练
├── restart     # 重启训练
├── finetune    # 微调
├── register    # 配准
│   ├── model   # 使用预训练模型初始化
│   ├── dicom   # 使用DICOM参数初始化
│   └── fixed   # 使用固定位姿初始化
├── animate     # 动画生成
└── dcm2nii     # 格式转换
```

---

### 2.2 训练模块 (`commands/train.py`)

**核心算法：位姿回归模型训练**

**训练流程：**

```python
def train_model(config, run):
    # 1. 加载所有CT体积数据
    volumes = [read(filepath) for filepath in niftis]
    
    # 2. 初始化模块
    model, drr, transforms, optimizer, scheduler = initialize(config, subject)
    
    # 3. 损失函数
    imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
    geodesic = DoubleGeodesicSE3(config["sdd"])
    
    # 4. 训练循环
    for epoch in range(n_epochs):
        for batch in range(n_batches_per_epoch):
            # 随机采样位姿
            pose = get_random_pose(config)
            
            # 渲染DRR并增强
            img, _, _ = render(drr, pose, volume, contrast)
            img = augmentations(img)
            
            # 预测位姿并渲染
            pred_pose = model(img)
            pred_img, _, _ = render(drr, pred_pose, volume, contrast)
            
            # 计算损失
            mncc = imagesim(img, pred_img)
            dgeo = geodesic(pose, pred_pose)
            loss = 1 - mncc + weight_geo * dgeo
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**关键技术点：**
1. **随机位姿采样** - 在6自由度空间（3个旋转角+3个平移）均匀采样
2. **对比度增强** - 随机对比度范围 [1.0, 10.0]
3. **图像增强** - 包括自动对比度、模糊、噪声、反转等
4. **复合损失** - 图像相似度（NCC）+ 测地距离（Geodesic）
5. **学习率策略** - Warmup + Cosine 衰减

---

### 2.3 微调模块 (`commands/finetune.py`)

**功能：** 在患者特定CT上微调预训练模型

**与训练模块的区别：**
- 加载预训练权重 `model.load_state_dict(model_state_dict)`
- 单个CT体积（不是多个）
- 更少的训练轮次（默认10 epochs）
- 可以缩放探测器平面 `rescale` 参数

```python
# 探测器缩放调整
config["height"] = int(config["height"] * rescale)
config["delx"] /= rescale
```

---

### 2.4 配准模块 (`registrar.py`)

**核心：三种配准器 + 多尺度优化算法**

#### 2.4.1 配准器基类 (`_RegistrarBase`)

**核心流程：**

```python
def run(self, i2d, beta):
    # 1. 获取初始位姿
    gt, intrinsics, init_pose = self.initialize_pose(i2d)
    
    # 2. 更新DRR内参
    self.drr.set_intrinsics_(**intrinsics)
    
    # 3. 多尺度配准优化
    for scale in scales:
        # 缩放探测器
        reg.drr.rescale_detector_(scale)
        
        # 初始化优化器
        optimizer = torch.optim.Adam([
            {"params": [reg.rotation], "lr": lr_rot / step_size_scalar},
            {"params": [reg.translation], "lr": lr_xyz / step_size_scalar}
        ], maximize=True)
        
        # 学习率调度器
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        
        # 迭代优化
        for itr in range(max_n_itrs):
            pred_img = reg()
            loss = imagesim(img, pred_img, beta=beta)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # 检查plateau
            if n_plateaus == max_n_plateaus:
                break
    
    return init_pose, final_pose
```

**关键算法：多尺度配准**

**多尺度策略：**
- 从低分辨率开始（如 1/8 分辨率）
- 逐步提升到全分辨率
- 每个尺度独立优化直到收敛
- 学习率随尺度递减

**收敛判定：**
- `ReduceLROnPlateau` - 损失平台期检测
- `patience=10` - 允许10次迭代无改进
- `max_n_plateaus=3` - 最多3次学习率衰减后进入下一尺度

#### 2.4.2 三种配准器

**1. RegistrarModel - 基于深度学习模型**
```python
def initialize_pose(self, i2d):
    gt = read_xray(i2d)  # 读取X射线
    init_pose = predict_pose(model, config, gt)  # 模型预测
    init_pose = _correct_pose(init_pose, warp)  # 坐标系校正
    return gt, init_pose
```

**2. RegistrarDicom - 基于DICOM参数**
```python
def initialize_pose(self, i2d):
    gt = read_xray(i2d)
    # 从DICOM提取C-arm角度
    alpha = ds.PositionerPrimaryAngle
    beta = ds.PositionerSecondaryAngle
    sid = ds.DistanceSourceToPatient
    init_pose = convert([alpha, beta, 0], [0, sid, 0])
    return gt, init_pose
```

**3. RegistrarFixed - 固定初始位姿**
```python
def initialize_pose(self, i2d):
    gt = read_xray(i2d)
    init_pose = convert(rot, xyz)  # 用户指定的固定位姿
    return gt, init_pose
```

---

### 2.5 DRR渲染器 (`renderer.py`)

**功能：** 初始化可微分数字重建射线成像(DRR)模块

```python
def initialize_drr(volume, mask, labels, ...):
    # 1. 读取CT体积
    subject = read(volume, mask, labels, orientation)
    
    # 2. 初始化DRR
    drr = DRR(
        subject,
        sdd=sdd,           # 射线源到探测器距离
        height=height,     # 探测器高度
        delx=delx,         # 像素物理尺寸
        renderer=renderer  # "siddon" 或 "trilinear"
    ).to("cuda")
    
    return drr
```

**两种渲染算法：**
- **Siddon** - 精确的射线追踪算法
- **Trilinear** - 三线性插值（更快）

---

### 2.6 DICOM处理 (`dicom.py`)

**核心功能：X射线图像读取与预处理**

```python
def read_xray(filename, crop, subtract_background, linearize, reducefn):
    # 1. 解析DICOM元数据
    img, sdd, delx, dely, x0, y0, pf_to_af = _parse_dicom(filename)
    
    # 2. 预处理
    img = _preprocess_xray(img, crop, subtract_background, linearize, reducefn)
    
    return img, sdd, delx, dely, x0, y0, pf_to_af
```

**预处理步骤：**

```python
def _preprocess_xray(img, crop, subtract_background, linearize, reducefn):
    # 1. 裁剪边缘（移除collimator伪影）
    if crop != 0:
        img = center_crop(img, (height - crop, width - crop))
    
    # 2. 归一化到 [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    
    # 3. 减去背景
    if subtract_background:
        background = img.flatten().mode().values.item()
        img = torch.clamp(img - background, -1, 0) + 1
    
    # 4. 线性化（从指数空间转换）
    if linearize:
        img = img.max().log() - img.log()
    
    # 5. 多帧处理
    if img.ndim == 5:
        img = img.max(dim=2).values  # 最大强度投影
    
    return img
```

**关键技术点：**
- **线性化** - X射线遵循Beer-Lambert定律：`I = I₀ * exp(-μx)`，转换为 `-log(I/I₀) = μx`
- **RAO图像重定向** - 检测 `PatientOrientation` 并翻转

---

### 2.7 评估模块 (`metrics.py`)

**功能：** 计算4种配准误差指标

```python
class Evaluator:
    def __call__(self, true_pose, pred_pose):
        # 1. 平均投影误差 (mPE) - 2D空间
        x = drr.perspective_projection(pred_pose, fiducials)
        y = drr.perspective_projection(true_pose, fiducials)
        mpe = (delx * (x - y)).norm(dim=-1).mean(dim=-1)
        
        # 2. 平均反投影误差 (mRPE) - 3D空间
        x = drr.inverse_projection(pred_pose, x)
        y = drr.inverse_projection(true_pose, y)
        mrpe = (x - y).norm(dim=-1).mean(dim=-1)
        
        # 3. 平均目标配准误差 (mTRE) - 基准点误差
        x = pred_pose(fiducials)
        y = true_pose(fiducials)
        mtre = (x - y).norm(dim=-1).mean(dim=-1)
        
        # 4. 双重测地距离 (dgeo) - SE(3)流形距离
        dgeo = geodesic(true_pose, pred_pose)
        
        return [mpe, mrpe, mtre, dgeo]
```

**误差指标含义：**
- **mPE** - 基准点在探测器平面上的投影距离（像素）
- **mRPE** - 基准点在3D空间的反投影距离（毫米）
- **mTRE** - 变换后基准点在3D空间的欧氏距离（毫米）
- **dgeo** - 位姿在SE(3)流形上的测地距离（旋转+平移）

---

### 2.8 可视化模块 (`visualization.py`)

**功能：** 配准结果可视化

```python
def plot_registration(drr, fiducials, gt, pred_pose, true_pose):
    # 1. 计算配准误差
    mtre = evaluator(true_pose, pred_pose)[2]
    
    # 2. 生成DRR和基准点投影
    pred_pts = drr.perspective_projection(pred_pose, fiducials)
    true_pts = drr.perspective_projection(true_pose, fiducials)
    pred_img = drr(pred_pose)
    true_img = drr(true_pose)
    error = true_img - pred_img
    
    # 3. 绘制3个子图
    # - 预测DRR + 基准点
    # - 真实X射线 + 基准点
    # - 误差图（热力图）
```

---

## 三、关键算法模块详解

### 3.1 位姿参数化 (`utils.py`)

**随机位姿生成：**

```python
def get_random_pose(config):
    # 在指定范围内均匀采样
    alpha = uniform(alphamin, alphamax, batch_size)
    beta = uniform(betamin, betamax, batch_size)
    gamma = uniform(gammamin, gammamax, batch_size)
    tx = uniform(txmin, txmax, batch_size)
    ty = uniform(tymin, tymax, batch_size)
    tz = uniform(tzmin, tzmax, batch_size)
    
    # 转换为SE(3)刚体变换
    rot = torch.concat([alpha, beta, gamma], dim=1) / 180 * π
    xyz = torch.concat([tx, ty, tz], dim=1)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
```

**位姿表示：**
- **旋转** - Euler角（ZXY顺序）或其他SO(3)参数化
- **平移** - 毫米单位的3D向量
- **组合** - SE(3) = SO(3) ⋉ ℝ³

### 3.2 图像相似度度量

**组合度量：**

```python
imagesim = lambda x, y, beta: beta * sim1(x, y) + (1 - beta) * sim2(x, y)

# sim1: 多尺度归一化互相关
sim1 = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])

# sim2: 梯度归一化互相关
sim2 = GradientNormalizedCrossCorrelation2d(patch_size=11, sigma=10)
```

**NCC公式：**
```
NCC(I, J) = Σ[(I - μᵢ)(J - μⱼ)] / √[Σ(I - μᵢ)² · Σ(J - μⱼ)²]
```

### 3.3 学习率调度

**WarmupCosineSchedule：**

```python
def lr_lambda(self, step):
    if step < warmup_steps:
        # 线性warmup
        return float(step) / float(max(1.0, warmup_steps))
    
    # Cosine衰减
    progress = (step - warmup_steps) / (t_total - warmup_steps)
    return 0.5 * (1.0 + cos(π * cycles * 2.0 * progress))
```

**训练阶段：**
1. **Warmup** - 前5个epoch线性增加（0 → lr）
2. **Cosine衰减** - 剩余epoch余弦函数衰减（lr → 0）

### 3.4 坐标系变换 (`utils.py`)

**刚体变换校正：**

```python
def get_4x4(mat, img, invert=False):
    # 1. 读取ANTs变换
    transform = ants.read_transform(mat)
    R = transform.parameters[:9].reshape(3, 3)  # 旋转矩阵
    t = transform.parameters[9:]                # 平移向量
    c = transform.fixed_parameters               # 旋转中心
    
    # 2. 计算全局平移
    global_t = -R @ c + t + c
    
    # 3. 构建4x4齐次矩阵
    M = [[R, global_t],
         [0, 1]]
    
    # 4. 方向矩阵校正
    D = direction_matrix(img)
    Tinv = translation_to_center(img)
    
    # 5. 组合变换
    T = Tinv @ D @ M @ D⁻¹
    
    # 6. 投影到SO(3)
    return project_onto_SO3(T)
```

**SO(3)投影（SVD分解）：**

```python
def project_onto_SO3(T):
    A = T[:3, :3]
    U, S, V = A.svd()
    
    # 强制det(R) = +1
    S = torch.ones_like(S)
    S[-1] = (U @ V.T).det()
    
    # 正交化
    R = U @ diag(S) @ V.T
    return RigidTransform(R)
```

---

## 四、数据流与依赖关系

### 4.1 训练数据流

```
CT体积 → read() → DRR渲染器
                    ↓
          随机位姿采样 → render()
                    ↓
                  增强 → XrayAugmentations
                    ↓
           ResNet18/34 → 位姿回归
                    ↓
              预测DRR ← render()
                    ↓
        NCC + Geodesic Loss
                    ↓
           反向传播 + 优化
```

### 4.2 配准数据流

```
DICOM → read_xray() → 预处理X射线
           ↓
    初始位姿估计 (3种方式)
           ↓
    多尺度迭代优化
           ↓
    Scale 1/8 → 优化至plateau
    Scale 1/4 → 优化至plateau
    Scale 1/2 → 优化至plateau
    Scale 1/1 → 优化至plateau
           ↓
       最终配准位姿
```

### 4.3 核心依赖库

```python
# 深度学习
torch, torchvision          # PyTorch生态
diffdrr                     # 可微分DRR渲染器
timm                        # ResNet预训练模型

# 医学图像
pydicom                     # DICOM读取
torchio, ants               # 医学图像处理

# 数据增强
kornia                      # 可微分图像变换

# 可视化
matplotlib                  # 绘图
wandb                       # 实验跟踪
```

---

## 五、性能优化技术

### 5.1 训练加速
1. **批量DRR渲染** - GPU并行渲染（batch_size=116）
2. **自适应梯度裁剪** - `adaptive_clip_grad` 稳定训练
3. **混合精度** - 支持 float32（主要用于医学图像精度要求）

### 5.2 配准加速
1. **多尺度策略** - 从粗到细，减少迭代次数
2. **早停机制** - plateau检测避免过度优化
3. **学习率自适应** - `ReduceLROnPlateau` 动态调整

### 5.3 内存优化
1. **按需加载CT** - 训练时随机选择体积
2. **梯度检查点** - 节省中间激活内存
3. **探测器缩放** - 低分辨率阶段减少显存占用

---

## 六、使用示例

### 6.1 训练模型

```bash
xvr train \
  -i /path/to/CTs/ \
  -o /output/models/ \
  --r1 -20 20 --r2 -20 20 --r3 -5 5 \
  --tx -50 50 --ty -50 50 --tz -50 50 \
  --sdd 1020 --height 256 --delx 0.194 \
  --n_epochs 1000
```

### 6.2 配准X射线

```bash
xvr register model \
  /path/to/xray.dcm \
  -v /path/to/ct.nii.gz \
  -c /path/to/model.pth \
  -o /output/results/ \
  --scales 8,4,2 \
  --saveimg
```

---

## 七、总结

**核心创新点：**
1. **深度学习初始化** - ResNet回归6-DoF位姿，避免陷入局部最优
2. **可微分渲染** - 端到端梯度反传，无需手工特征
3. **多尺度优化** - 粗到细策略，提升收敛速度和精度
4. **组合损失函数** - 图像相似度 + 几何约束

**适用场景：**
- 术中导航（实时X射线与术前CT配准）
- 放疗定位（验证患者位置）
- 介入手术（C-arm与CT融合）

**关键代码逻辑：** 训练阶段学习X射线→位姿映射，配准阶段用学习的初始化 + 迭代优化达到亚毫米精度。

```python
# 命令层级结构
cli (主入口)
├── train       # 训练
├── restart     # 重启训练
├── finetune    # 微调
├── register    # 配准
│   ├── model   # 使用预训练模型初始化
│   ├── dicom   # 使用DICOM参数初始化
│   └── fixed   # 使用固定位姿初始化
├── animate     # 动画生成
└── dcm2nii     # 格式转换
```

```python
def train_model(config, run):
    # 1. 加载所有CT体积数据
    volumes = [read(filepath) for filepath in niftis]
    
    # 2. 初始化模块
    model, drr, transforms, optimizer, scheduler = initialize(config, subject)
    
    # 3. 损失函数
    imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
    geodesic = DoubleGeodesicSE3(config["sdd"])
    
    # 4. 训练循环
    for epoch in range(n_epochs):
        for batch in range(n_batches_per_epoch):
            # 随机采样位姿
            pose = get_random_pose(config)
            
            # 渲染DRR并增强
            img, _, _ = render(drr, pose, volume, contrast)
            img = augmentations(img)
            
            # 预测位姿并渲染
            pred_pose = model(img)
            pred_img, _, _ = render(drr, pred_pose, volume, contrast)
            
            # 计算损失
            mncc = imagesim(img, pred_img)
            dgeo = geodesic(pose, pred_pose)
            loss = 1 - mncc + weight_geo * dgeo
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

```python
# 探测器缩放调整
config["height"] = int(config["height"] * rescale)
config["delx"] /= rescale
```

```python
def run(self, i2d, beta):
    # 1. 获取初始位姿
    gt, intrinsics, init_pose = self.initialize_pose(i2d)
    
    # 2. 更新DRR内参
    self.drr.set_intrinsics_(**intrinsics)
    
    # 3. 多尺度配准优化
    for scale in scales:
        # 缩放探测器
        reg.drr.rescale_detector_(scale)
        
        # 初始化优化器
        optimizer = torch.optim.Adam([
            {"params": [reg.rotation], "lr": lr_rot / step_size_scalar},
            {"params": [reg.translation], "lr": lr_xyz / step_size_scalar}
        ], maximize=True)
        
        # 学习率调度器
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        
        # 迭代优化
        for itr in range(max_n_itrs):
            pred_img = reg()
            loss = imagesim(img, pred_img, beta=beta)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # 检查plateau
            if n_plateaus == max_n_plateaus:
                break
    
    return init_pose, final_pose
```

```python
def initialize_pose(self, i2d):
    gt = read_xray(i2d)  # 读取X射线
    init_pose = predict_pose(model, config, gt)  # 模型预测
    init_pose = _correct_pose(init_pose, warp)  # 坐标系校正
    return gt, init_pose
```

```python
def initialize_pose(self, i2d):
    gt = read_xray(i2d)
    # 从DICOM提取C-arm角度
    alpha = ds.PositionerPrimaryAngle
    beta = ds.PositionerSecondaryAngle
    sid = ds.DistanceSourceToPatient
    init_pose = convert([alpha, beta, 0], [0, sid, 0])
    return gt, init_pose
```

```python
def initialize_pose(self, i2d):
    gt = read_xray(i2d)
    init_pose = convert(rot, xyz)  # 用户指定的固定位姿
    return gt, init_pose
```

```python
def initialize_drr(volume, mask, labels, ...):
    # 1. 读取CT体积
    subject = read(volume, mask, labels, orientation)
    
    # 2. 初始化DRR
    drr = DRR(
        subject,
        sdd=sdd,           # 射线源到探测器距离
        height=height,     # 探测器高度
        delx=delx,         # 像素物理尺寸
        renderer=renderer  # "siddon" 或 "trilinear"
    ).to("cuda")
    
    return drr
```

```python
def read_xray(filename, crop, subtract_background, linearize, reducefn):
    # 1. 解析DICOM元数据
    img, sdd, delx, dely, x0, y0, pf_to_af = _parse_dicom(filename)
    
    # 2. 预处理
    img = _preprocess_xray(img, crop, subtract_background, linearize, reducefn)
    
    return img, sdd, delx, dely, x0, y0, pf_to_af
```

```python
def _preprocess_xray(img, crop, subtract_background, linearize, reducefn):
    # 1. 裁剪边缘（移除collimator伪影）
    if crop != 0:
        img = center_crop(img, (height - crop, width - crop))
    
    # 2. 归一化到 [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    
    # 3. 减去背景
    if subtract_background:
        background = img.flatten().mode().values.item()
        img = torch.clamp(img - background, -1, 0) + 1
    
    # 4. 线性化（从指数空间转换）
    if linearize:
        img = img.max().log() - img.log()
    
    # 5. 多帧处理
    if img.ndim == 5:
        img = img.max(dim=2).values  # 最大强度投影
    
    return img
```

```python
class Evaluator:
    def __call__(self, true_pose, pred_pose):
        # 1. 平均投影误差 (mPE) - 2D空间
        x = drr.perspective_projection(pred_pose, fiducials)
        y = drr.perspective_projection(true_pose, fiducials)
        mpe = (delx * (x - y)).norm(dim=-1).mean(dim=-1)
        
        # 2. 平均反投影误差 (mRPE) - 3D空间
        x = drr.inverse_projection(pred_pose, x)
        y = drr.inverse_projection(true_pose, y)
        mrpe = (x - y).norm(dim=-1).mean(dim=-1)
        
        # 3. 平均目标配准误差 (mTRE) - 基准点误差
        x = pred_pose(fiducials)
        y = true_pose(fiducials)
        mtre = (x - y).norm(dim=-1).mean(dim=-1)
        
        # 4. 双重测地距离 (dgeo) - SE(3)流形距离
        dgeo = geodesic(true_pose, pred_pose)
        
        return [mpe, mrpe, mtre, dgeo]
```

```python
def plot_registration(drr, fiducials, gt, pred_pose, true_pose):
    # 1. 计算配准误差
    mtre = evaluator(true_pose, pred_pose)[2]
    
    # 2. 生成DRR和基准点投影
    pred_pts = drr.perspective_projection(pred_pose, fiducials)
    true_pts = drr.perspective_projection(true_pose, fiducials)
    pred_img = drr(pred_pose)
    true_img = drr(true_pose)
    error = true_img - pred_img
    
    # 3. 绘制3个子图
    # - 预测DRR + 基准点
    # - 真实X射线 + 基准点
    # - 误差图（热力图）
```

```python
def get_random_pose(config):
    # 在指定范围内均匀采样
    alpha = uniform(alphamin, alphamax, batch_size)
    beta = uniform(betamin, betamax, batch_size)
    gamma = uniform(gammamin, gammamax, batch_size)
    tx = uniform(txmin, txmax, batch_size)
    ty = uniform(tymin, tymax, batch_size)
    tz = uniform(tzmin, tzmax, batch_size)
    
    # 转换为SE(3)刚体变换
    rot = torch.concat([alpha, beta, gamma], dim=1) / 180 * π
    xyz = torch.concat([tx, ty, tz], dim=1)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
```

```python
imagesim = lambda x, y, beta: beta * sim1(x, y) + (1 - beta) * sim2(x, y)

# sim1: 多尺度归一化互相关
sim1 = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])

# sim2: 梯度归一化互相关
sim2 = GradientNormalizedCrossCorrelation2d(patch_size=11, sigma=10)
```

```plaintext
NCC(I, J) = Σ[(I - μᵢ)(J - μⱼ)] / √[Σ(I - μᵢ)² · Σ(J - μⱼ)²]
```

```python
def lr_lambda(self, step):
    if step < warmup_steps:
        # 线性warmup
        return float(step) / float(max(1.0, warmup_steps))
    
    # Cosine衰减
    progress = (step - warmup_steps) / (t_total - warmup_steps)
    return 0.5 * (1.0 + cos(π * cycles * 2.0 * progress))
```

```python
def get_4x4(mat, img, invert=False):
    # 1. 读取ANTs变换
    transform = ants.read_transform(mat)
    R = transform.parameters[:9].reshape(3, 3)  # 旋转矩阵
    t = transform.parameters[9:]                # 平移向量
    c = transform.fixed_parameters               # 旋转中心
    
    # 2. 计算全局平移
    global_t = -R @ c + t + c
    
    # 3. 构建4x4齐次矩阵
    M = [[R, global_t],
         [0, 1]]
    
    # 4. 方向矩阵校正
    D = direction_matrix(img)
    Tinv = translation_to_center(img)
    
    # 5. 组合变换
    T = Tinv @ D @ M @ D⁻¹
    
    # 6. 投影到SO(3)
    return project_onto_SO3(T)
```

```python
def project_onto_SO3(T):
    A = T[:3, :3]
    U, S, V = A.svd()
    
    # 强制det(R) = +1
    S = torch.ones_like(S)
    S[-1] = (U @ V.T).det()
    
    # 正交化
    R = U @ diag(S) @ V.T
    return RigidTransform(R)
```

```plaintext
CT体积 → read() → DRR渲染器
                    ↓
          随机位姿采样 → render()
                    ↓
                  增强 → XrayAugmentations
                    ↓
           ResNet18/34 → 位姿回归
                    ↓
              预测DRR ← render()
                    ↓
        NCC + Geodesic Loss
                    ↓
           反向传播 + 优化
```

```plaintext
DICOM → read_xray() → 预处理X射线
           ↓
    初始位姿估计 (3种方式)
           ↓
    多尺度迭代优化
           ↓
    Scale 1/8 → 优化至plateau
    Scale 1/4 → 优化至plateau
    Scale 1/2 → 优化至plateau
    Scale 1/1 → 优化至plateau
           ↓
       最终配准位姿
```

```python
# 深度学习
torch, torchvision          # PyTorch生态
diffdrr                     # 可微分DRR渲染器
timm                        # ResNet预训练模型

# 医学图像
pydicom                     # DICOM读取
torchio, ants               # 医学图像处理

# 数据增强
kornia                      # 可微分图像变换

# 可视化
matplotlib                  # 绘图
wandb                       # 实验跟踪
```

```shellscript
xvr train \
  -i /path/to/CTs/ \
  -o /output/models/ \
  --r1 -20 20 --r2 -20 20 --r3 -5 5 \
  --tx -50 50 --ty -50 50 --tz -50 50 \
  --sdd 1020 --height 256 --delx 0.194 \
  --n_epochs 1000
```

```shellscript
xvr register model \
  /path/to/xray.dcm \
  -v /path/to/ct.nii.gz \
  -c /path/to/model.pth \
  -o /output/results/ \
  --scales 8,4,2 \
  --saveimg
```

