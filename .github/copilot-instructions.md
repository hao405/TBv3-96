# Copilot 使用说明（针对 TBv3-96 / TimeBridge 项目）

目的：帮助 AI 编码代理快速理解本仓库的结构、主要设计决策、常用运行/调试命令及代码风格约定，便于生成正确且可运行的修改。

1) 项目大局观
- 架构分层：
  - `data_provider/`：统一的数据加载、缩放与逆缩放逻辑（训练/验证/测试）。
  - `experiments/`：实验控制器（训练/验证/测试流程）。主要变体：`exp_long_term_forecasting.py`（训练循环、度量与保存模型）。
  - `model/`：模型组装（`TimeBridge.py` 为主模型）。
  - `layers/`：模型里更细粒度的构件（PatchEmbed、TSMixer、Attention 等）。
  - `utils/`：训练工具（EarlyStopping、学习率调整、metrics）。
- 数据流：`run.py` -> `experiments/*.py` 调用 `data_provider.data_factory.data_provider` 得到 dataloader -> 模型前向 -> loss（time + freq）-> 优化 -> checkpoint 存盘（`./checkpoints/<setting>/checkpoint.pth`）。

2) 快速上手命令（必须是可复现的示例）
- 安装依赖：
  - pip install -r requirements.txt
- 获取数据：将数据解压到项目根目录，确保存在 `dataset/`（与 README 中结构一致）。
- 在类 Unix 环境（仓库 README）：
  - sh ./scripts/TimeBridge.sh
- Windows / PowerShell 示例（等价运行）：
  - python .\run.py --is_training 1 --model_id "exp1" --model TimeBridge --data ETT-small --root_path .\dataset\ETT-small\ --data_path ETTh1.csv --seq_len 96 --pred_len 96 --enc_in 7

3) 关键文件（必读，AI 需要参考这些文件来作改动）
- `run.py`：入口；定义大量命令行参数（超参在此暴露）。
- `experiments/exp_long_term_forecasting.py`：训练/验证/测试流程；实现复合损失 `time_freq_mae`（time + freq 的 MAE 混合，权重由 `--alpha` 控制）。
- `experiments/exp_basic.py`：设备选择（通过设置 CUDA_VISIBLE_DEVICES）、模型字典注册位置。
- `model/TimeBridge.py`：模型的 assemble/forward；注意 `forecast` 返回经均值/方差恢复后的预测；使用两个并行的 PatchEmbed+Encoder 分支 (mean / std) 并通过 reparametrize 生成输出。
- `layers/Embed.py`、`layers/SelfAttention_Family.py`、`layers/Transformer_EncDec.py`：实现了 PatchEmbed、TSMixer、IntAttention、PatchSampling、CointAttention 等模块，是网络修改的首选位置。
- `data_provider/data_factory.py`：负责 dataset 划分、scale、inverse_transform；某些 dataset（PEMS、Solar）会跳过时间编码（见代码注释）。
- `utils/tools.py`：包含 EarlyStopping、学习率调整、visual（可视化）等工具。

4) 项目特定约定与常见模式（必须遵守）
- 参数驱动：绝大多数行为通过 `run.py` 的命令行参数控制（例如 `--label_len`、`--pred_len`、`--seq_len`、`--enc_in`、`--features` 等）。修改默认行为优先添加/更改参数而非改动调用点。
- 特征模式：`--features` 可为 `M` / `S` / `MS`，在预测切片时用 `f_dim = -1 if features=='MS' else 0` 选择通道维度（见 experiments 中多处）。
- decoder 输入构造：训练/测试里通用做法是构造 `dec_inp = cat(batch_y[:, :label_len, :], zeros)` 作为 channel-decoder 的输入；不要删除或绕过此构造，除非确认新 decoder 能替代它。
- 损失：主实验使用 time + frequency 的 MAE 混合（`alpha` 控制）。直接修改损失需在 `experiments/exp_long_term_forecasting.py::time_freq_mae` 确保兼容频域实现（使用 torch.fft.rfft）。
- Checkpoint 与结果文件位置：模型保存到 `./checkpoints/<setting>/checkpoint.pth`，测试结果追加到 `result_long_term_forecast.txt`。
- 多 GPU：通过 `--use_multi_gpu` 与 `--devices` 启用，模型使用 `nn.DataParallel` 包装（在 Exp_Basic._acquire_device 设置 CUDA_VISIBLE_DEVICES）。

5) 调试与常见问题
- 若看不到 GPU，请检查 `--use_gpu`、`--gpu` 与 `--use_multi_gpu`；Exp_Basic 会设置环境变量 `CUDA_VISIBLE_DEVICES`。Windows 上请确保 CUDA 驱动正确安装。
- PEMS / Solar 数据集不使用时间编码（在 exp 文件中有明确判断），因此 `batch_x_mark` / `batch_y_mark` 会被置为 None。
- 运行失败常见原因：数据路径不正确、dataset 未解压、或 `enc_in`/`pred_len` 不匹配模型。修改模型输出层时注意 decoder 的输出形状：最终解码器将输出 [B, pred_len, C]。

6) 对 AI 代理的建议（如何安全地改动）
- 优先改动 `layers/` 中小模块并在 `model/TimeBridge.py` 做最小 glue 变更；在修改前阅读 `PatchEmbed` 和编码器输出的 tensor 形状。
- 修改训练超参时，优先通过 `run.py` 参数添加或调整，而非硬编码默认值。
- 添加新数据集时，扩展 `data_provider/data_factory.py`，保证 `scale` 与 `inverse_transform` 的一致行为。
- 不要移除 `time_freq_mae` 的频域分支，除非提交同时更新实验设置和 README 以说明变化。

---
有需要补充的具体部分或你想让我把某些文件的细节也写进说明吗？请指出不清楚或想补充的项。
