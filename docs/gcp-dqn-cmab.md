# GCP 分支上的 DQN 与 Cut–FPT 交叉特征

本工作分支基于 `origin/gcp` 的 `30c0774`，移植来源为 `alan-test` 的 `320caab`。
保留 GCP 的协议实现、动作目录、numeric/one_hot 编码、factorized/combined 策略及加速器。

## CMAB 交叉特征

在 `benchmark/fabfile.py` 对应实验的 `bench_params` 中设置：

```python
'rl_algo': 'cmab',
'cmab_policy': 'rf_ts',
'cmab_action_encoding': 'numeric',
'enable_cmab_cut_fpt_cross_feature': True,
```

这会追加 12 维 one-hot 特征：cut = 2/3/4 × timeout = 0/100/200/300。
默认关闭，关闭时原有特征不变；也支持在 GCP 的完整动作 one_hot 编码后追加。
不适用于 factorized、combined 或其他算法；误配会明确报错。
交叉特征 checkpoint 单独保存到 `~/checkpoints/cmab_<encoding>_cut_fpt/`。
开关、编码、动作目录或交叉特征 schema 不匹配的 checkpoint 会拒绝加载。

## 在线 DQN

```python
'rl_algo': 'dqn',
'enable_cmab_cut_fpt_cross_feature': False,
'dqn_action_port': 19100,
'dqn_checkpoint_load_mode': 'resume',
'cmab_resume_from': None,
'rl_max_training_iterations': 2500,
```

沿用原来的 GCP benchmark 启动方式。node0 运行一个 DQN 训练器，所有 primary（包括 node0）运行动作接收器。
接收器启动并通过端口就绪检查后才启动训练器；远程实验要求每台主机一个 primary。
本地实验使用 19100、19101……避免端口冲突。
远程节点之间需要能访问配置的 TCP 端口；本次移植不修改云端网络规则。

DQN 保留源分支的 `Q(state, action)` 网络、动作特征、replay buffer、target network、epsilon 探索及 checkpoint。
GCP 当前动作目录是 96 个动作，未改变其参数取值范围。
训练样本必须同时满足：全部广播成功、epoch 连续、奖励有效、未发现本地 abandon 信号，且 GCP 的 `param_apply_ok` 为 true。
接收器 ACK 只表示参数已写入并发出信号，不能替代协议层的生效确认。
在线 checkpoint 在 node0 的 `<metrics目录的父目录>/dqn_checkpoints/state_action_q_v1/`。

## 离线数据 → 离线训练 → 在线微调

先在 CMAB 实验中启用 node0 数据导出：

```python
'enable_cmab_transition_export': True,
'cmab_transition_export_dir': '/tmp/autopilot_offline_data',
'cmab_environment_label': 'gcp-A',
```

导出包含 manifest、transitions、epoch action 记录；每次运行使用独立目录。
只导出连续且参数确实生效的训练样本，导出失败不会中断 CMAB。
数据目录内另存每次定期 checkpoint 的最新 CMAB 快照。长期实验请将 `/tmp` 改为持久化目录。

在安装了 `Agent/requirements.txt` 依赖的 Python 环境下，从仓库根目录执行：

```bash
python Agent/rl/train_dqn_offline.py \
  --dataset-root /tmp/autopilot_offline_data \
  --environment gcp-A \
  --gradient-steps 5000 \
  --checkpoint-output /tmp/dqn-offline.pt
```

支持多个 `--environment`、按环境/运行均衡采样和 `--init-checkpoint` 增量训练。
然后把生成的 checkpoint 放到 node0 上，并在下一次在线 DQN 实验设置：

```python
'rl_algo': 'dqn',
'cmab_resume_from': '/path/on/node0/dqn-offline.pt',
'dqn_checkpoint_load_mode': 'finetune',
```

`finetune` 保留网络权重，重置经验回放、计数和优化器；`resume` 恢复同一训练任务的状态。
GCP 启动器沿用已有的远程路径语义，不会自动上传本地 checkpoint。
旧 checkpoint 需要状态维度、动作目录及网络 schema 一致；不兼容时明确报错。

## 运行入口与验证

GCP `remote.py` 和本地 `local.py` 已接入上述选项。原 GCP 版本的 CloudLab 控制器被注释；
本次仅在显式选择 DQN、交叉特征或离线数据导出时启动 CloudLab 控制器，其他情况保留原行为。
没有引入源分支的 coverage_round_robin、KernelUCB 改写、reward monitor 或协议修改。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s Agent/rl/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=benchmark python -m unittest discover -s benchmark/tests
```

测试覆盖特征/模型恢复、原有 CMAB 策略、数据导出隔离、离线训练到微调、
DQN 的 GCP 参数生效检查、动作去重、命令参数和多节点接收器配置。
云端多节点真实运行仍需在部署环境验证；本地测试不代表已经完成分布式实验。
