# 第 22 章 高性能计算智能与预测技术

## 22.1 写作动机

由 Lawrence Berkeley National Lab 的 Horst Simon 和 Kesheng Wu 撰写。本章从美国国家实验室的视角介绍 ML 在大规模科学计算中的实践。Berkeley Lab 在 ML/HPC 领域积淀近 90 年——发现 16 种新元素、为 MRI/PET 奠基,远早于硅谷热潮。

## 22.2 监管对 2010 闪崩的回应

2010-05-06,SP500 在数分钟内崩跌近 9%——"Flash Crash"。SEC/CFTC 后续报告指出:
- 高频交易者在异常波动时撤出做市单。
- 流动性瞬间消失。
- ML/HPC 工具(VPIN、有符号订单流分析)能提前数小时识别危机征兆。

## 22.3 背景

国家实验室 vs. 大学/公司:
- **大学**: 个人研究员 + 学生,主题碎片化。
- **公司**: 商业目标驱动,短期视角。
- **国家实验室**: 团队协作 + 长期 + 跨学科。

正是金融业理想的"meta-strategy" 模式(第 1 章)。

## 22.4 HPC 硬件

| 类型 | 适合 |
|------|------|
| **CPU 集群** | 通用计算;Slurm 调度 |
| **GPU (NVIDIA)** | 深度学习训练;CUDA |
| **量子退火 (D-Wave)** | 组合优化(第 21 章) |
| **FPGA** | 低延迟交易、HFT 数据预处理 |

## 22.5 HPC 软件

### 22.5.1 Message Passing Interface (MPI)
跨节点通信的事实标准。Python 用 `mpi4py`。

### 22.5.2 Hierarchical Data Format 5 (HDF5)
分层文件格式,支持并行 I/O。Pandas `to_hdf` / `read_hdf` 底层就是 HDF5。

### 22.5.3 In Situ 处理
"边产生边处理"——避免把 PB 级原始数据写盘后再读回来。
对 tick 数据流尤其重要:不存储全部 tick,在线计算 VPIN、SADF、滚动统计等。

### 22.5.4 Convergence (HPC + AI)
传统 HPC = 模拟物理过程;AI = 从数据学习。两者正在合并——同一台超算同时跑模拟器和 ML 模型。

## 22.6 应用案例

### 22.6.1 超新星探测
LSST/Vera Rubin 望远镜每晚 30 TB 数据,ML 模型 in-situ 筛选超新星候选。

### 22.6.2 核聚变等离子体中的 Blob
ITER/EAST 装置中等离子体不稳定结构的实时检测——ML+物理模型混合。

### 22.6.3 日内电力负荷峰值
电网调度的预测性维护。

### 22.6.4 2010 闪崩 ★

VPIN(第 19 章) 已被 SEC 用作早期预警:
- 闪崩前 30 分钟 VPIN 显著上升。
- 对应做市商撤出 → 流动性蒸发 → 闪崩触发。

### 22.6.5 VPIN 校准
HPC 让我们在每个交易日数十亿 tick 上重新校准 VPIN 桶大小,最优化预测能力。

### 22.6.6 用 NUFFT 揭示高频事件

不规则到达的 tick 不能直接 FFT。**Non-Uniform FFT (NUFFT)**:

$$F(f_k) = \sum_t x_t e^{-2\pi i f_k t}$$

在不规则采样网格上做谱分析,识别隐藏的周期性(如 1ms 周期的算法 footprint)。

## 22.7 总结与号召

国家实验室对金融工业贡献巨大但被低估。Marcos 鼓励金融机构与国家实验室合作——这是"meta-strategy"的最大化形式。

## 22.8 致谢

略

## 22.9 本章核心要点

1. **HPC + ML = 金融未来**——单机时代结束。
2. SEC 已采用 VPIN 作为闪崩预警工具——ML 进入监管。
3. **In-situ processing** 是 tick 级数据的标配。
4. **NUFFT** 揭示不规则采样下的隐藏周期。
5. 与国家实验室合作是高级量化研究的捷径。

## 配套代码示例

`python_demos/chapter_22_hpc_intelligence.py` 实现:
- VPIN calibration
- NUFFT
- Streaming moments (in-situ 范例)
- MPI workflow stub
