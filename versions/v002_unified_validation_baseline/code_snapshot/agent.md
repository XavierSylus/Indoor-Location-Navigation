# Indoor Location Kaggle Agent Guide

本项目是 Kaggle `Indoor Location & Navigation` 比赛的工作区。核心目标不是堆提交文件，而是持续管理实验版本、验证路线、记录判断，并让每一次 Kaggle 提交都能复盘。

所有 agent 和人工操作都必须围绕一个原则：每个正式版本都要能回答“为什么做、改了什么、怎么复现、验证结果如何、是否值得继续”。

## Project Facts

比赛 slug 固定为：

```text
indoor-location-navigation
```

当前事实源：

- `README.md`: 项目状态和推荐主线。
- `EXPERIMENT_LEDGER.md`: 提交面向实验总账。
- `CONTINUATION_GUIDE.md`: 后续路线说明。
- `data_processing/processed/*.json`: 已验证的长期证据。

当前最佳历史记录：

| item | value |
| --- | --- |
| submission | `submission_step3_beamsearch.csv` |
| route family | `step3 global beam` |
| Public LB | `6.64 m` |
| Private LB | `7.90 m` |
| decision | `historical` |

后续版本不能仅凭单次 smoke run 或孤立 LB 结果替代当前最佳记录。新路线必须有统一验证报告、配置、seed、commit 和清晰 judgement。

## Directory Layout

推荐从现在开始使用以下结构管理新实验：

```text
.
├── agent.md
├── README.md
├── EXPERIMENT_LEDGER.md
├── configs/
├── data_processing/
│   └── processed/
├── models/
├── scripts/
├── src/
├── tests/
├── versions/
│   ├── _template/
│   │   └── README.md
│   └── v001_short_name/
│       ├── README.md
│       ├── config.yml
│       ├── run_manifest.json
│       ├── reproduce.ps1
│       ├── validation_summary.json
│       ├── metrics.json
│       ├── train.log
│       ├── submission.csv
│       └── code_snapshot/
│           ├── SHA256SUMS.txt
│           └── ...
├── submissions/
└── tmp/
```

现有根目录下的历史 `submission_*.csv` 保留为历史证据，不要覆盖。新提交文件统一复制到 `submissions/`，文件名必须带版本号。

## Folder Meaning

- `versions/`: 每个正式实验版本独立存放，不覆盖旧版本。
- `versions/vXXX_short_name/README.md`: 单版本复盘核心。
- `submissions/`: 准备提交或已提交到 Kaggle 的 CSV 副本。
- `configs/`: 所有超参数和路径配置。禁止把重要超参数硬编码到脚本里。
- `data_processing/`: 数据处理代码统一目录。
- `data_processing/processed/`: 长期验证报告、指标、诊断结果。
- `scripts/`: 训练、推理、验证、提交辅助脚本入口。
- `tmp/`: 临时输出，可清理但必须先列清单并等待用户批准。

## Version Naming

版本目录格式：

```text
versions/vXXX_short_name/
```

规则：

- `vXXX` 必须递增，三位数补零。
- `short_name` 使用小写英文、数字和下划线。
- 一个版本只验证一个主要假设。
- bugfix 也要开新版本，因为它会改变实验判断。

示例：

```text
versions/v001_rebuild_baseline/
versions/v002_unified_holdout/
versions/v003_topk_rerank_probe/
versions/v004_site_whitelist_beam/
```

## Version Reproducibility Contract

每个正式版本目录必须包含：

```text
versions/vXXX_short_name/
├── README.md
├── config.yml
├── run_manifest.json
├── reproduce.ps1
├── validation_summary.json
├── metrics.json
├── train.log
├── submission.csv
└── code_snapshot/
    ├── SHA256SUMS.txt
    └── ...
```

硬规则：

- 没有 `run_manifest.json` 的版本不能标记为 `done`，不能作为后续 base。
- 没有 `reproduce.ps1` 的版本不能称为可复现版本。
- 没有 `code_snapshot/` 的版本只能作为历史参考，不能作为可复现候选。
- `run_manifest.json` 必须记录 exact command、git commit、random seed、Python 环境、config path、input data path、output path、validation report、experiment_type、iteration_mode。
- `code_snapshot/` 必须保存本轮真正用到的入口脚本和必要共享模块；后续主干代码可以演进，但不能污染历史版本复现。

## Experiment Types

`experiment_type` 只能使用以下值：

- `cv_probe`: 本地验证探针。默认不消耗 Kaggle LB。
- `lb_direction_probe`: 低风险 LB 方向探针，只回答一个方向问题。
- `leaderboard_attempt`: 冲榜尝试。必须由本地统一验证或方向探针支撑。

`iteration_mode` 只能使用以下值：

- `add_module`: 新增独立模块，例如新特征、新模型族、新验证器。
- `assemble_module`: 组装已有模块，例如 absolute baseline + site whitelist beam。
- `tune_module`: 只调整已有模块的参数、权重、seed、fold 或阈值。

约束：

- 一个版本只能有一个 `experiment_type` 和一个 `iteration_mode`。
- 不允许在同一版本里同时新增模块、组装模块、调参。
- `leaderboard_attempt` 默认必须是 `assemble_module` 或 `tune_module`。
- 未经本地验证的新模块不能直接塞进冲榜版本。

## Current Mainline

当前推荐主线是 absolute-position-first：

1. 先维护统一 holdout validation。
2. 分析 absolute baseline 的误差来源。
3. 只在固定验证集上反复改善的 site 上重新引入 beam。
4. PDR、global beam、gating、top-k rerank 都必须先作为探针验证。

当前不推荐：

- 不要把 global `PDR + beam + gating` 作为默认路线继续提交。
- 不要混用不同 holdout 集合比较方法。
- 不要用单个站点或极小样本的改善宣布新主线。

## Agent Workflow

每次新实验必须按以下流程执行：

1. 读取 `EXPERIMENT_LEDGER.md`，确认当前 best、recent rejected、mainline decision。
2. 读取目标 base version 的 `README.md`，如果没有 version 目录，则读取当前事实源文件。
3. 明确本轮只验证一个主要假设。
4. 写出方案，说明数据流变化、全局副作用、验证方式，并等待用户批准。
5. 创建 `versions/vXXX_short_name/`。
6. 复制或写入本轮 `config.yml`。
7. 运行训练、验证、推理。
8. 保存 log、metrics、validation summary、submission。
9. 复制提交候选到 `submissions/vXXX_short_name.csv`。
10. 生成 `run_manifest.json`、`reproduce.ps1`、`code_snapshot/SHA256SUMS.txt`。
11. 更新版本 `README.md`，必须写 `Judgement` 和 `Next`。
12. 更新 `EXPERIMENT_LEDGER.md`，记录 seed、commit、config、validation report、submission file。

## Per-Version README

每个版本必须有固定结构：

```md
# vXXX_short_name

## Goal
这个版本验证什么。

## Base
基于哪个版本或哪条历史路线，为什么。

## Data Flow
输入、处理、模型、后处理、输出文件如何流动。

## Changes
- 关键改动 1
- 关键改动 2

## Validation
- Holdout definition:
- Local metric:
- Public LB:
- Private LB:
- CV/LB gap or explanation:

## Files
- Config:
- Script:
- Metrics:
- Validation report:
- Submission:

## Side Effects
说明对已有主线、配置、缓存、模型、提交文件的影响。

## Judgement
这次实验说明了什么。必须写结论，不能只写分数。

## What Worked
-

## What Failed
-

## Next
下一步最值得做什么。
```

## EXPERIMENT_LEDGER Rules

每个正式版本结束后必须写入 `EXPERIMENT_LEDGER.md`。

必填字段：

- date
- submission file
- route family
- config path
- validation report
- git commit
- seed
- holdout definition
- public leaderboard score
- private leaderboard score
- decision
- notes

`decision` 只能是：

- `mainline`
- `exploratory`
- `rejected`
- `historical`

未知历史字段写 `unknown`，不要猜。

## Kaggle API Setup

本地提交依赖 Kaggle CLI：

```powershell
kaggle --version
kaggle competitions files -c indoor-location-navigation
kaggle competitions submissions -c indoor-location-navigation -v
```

Kaggle 凭证只允许放在用户目录：

```text
%USERPROFILE%\.kaggle\kaggle.json
```

禁止把以下文件放入项目或提交到 git：

- `kaggle.json`
- `access_token`
- `.env`
- 任何包含 Kaggle token 的文件

如果项目根目录已经存在历史 `kaggle.json`，不要继续依赖它；后续应改用用户目录凭证。

## Local Kaggle Submission Interface

用户明确批准某一个版本后，agent 可以直接从本地调用 Kaggle API 提交，不需要用户进入官网。

提交命令模板：

```powershell
kaggle competitions submit -c indoor-location-navigation -f submissions/vXXX_short_name.csv -m "vXXX_short_name cv=<local_cv> seed=<seed>"
```

查询提交结果：

```powershell
kaggle competitions submissions -c indoor-location-navigation -v
```

如果需要下载比赛数据：

```powershell
kaggle competitions download -c indoor-location-navigation -p indoor-location-navigation
```

提交前必须满足：

- `submissions/vXXX_short_name.csv` 已存在。
- 对应 `versions/vXXX_short_name/README.md` 已完成。
- `validation_summary.json` 已记录 row count、columns、site_path_timestamp 顺序、坐标有限性、NaN/inf count。
- `run_manifest.json` 已记录 git commit、seed、config、命令、环境。
- `EXPERIMENT_LEDGER.md` 已记录本地验证结果。
- agent 已向用户展示本次提交申请，并等待明确授权。

提交申请必须写清楚：

- competition slug: `indoor-location-navigation`
- version: `vXXX_short_name`
- file: `submissions/vXXX_short_name.csv`
- message
- local validation metric
- seed
- config path
- expected cost: 消耗 1 次 Kaggle 提交额度

用户必须明确说“提交 vXXX”或给出等价明确授权。仅批准写代码、生成文件或验证结果，不等于批准提交 Kaggle。

提交后必须执行：

1. 查询最新提交状态。
2. 记录 Kaggle ref、status、public score、private score 可见时的结果。
3. 回写 `EXPERIMENT_LEDGER.md`。
4. 回写版本 `README.md` 的 `Validation` 和 `Judgement`。
5. 如果提交失败或分数异常，停止继续提交，先定位根因。

## Submission Validation

提交 CSV 必须满足 Kaggle 赛制格式：

- 列名与 sample submission 一致。
- 行数与 sample submission 一致。
- key 顺序与 sample submission 一致。
- x/y/floor 等预测列全部为有限值。
- 不允许 NaN、inf、空字符串。
- 不允许手工改动 sample submission 的 ID。

建议提交前生成 `validation_summary.json`：

```json
{
  "submission_file": "submissions/vXXX_short_name.csv",
  "row_count": 0,
  "columns": [],
  "id_order_matches_sample": true,
  "nan_count": 0,
  "inf_count": 0,
  "coordinate_summary": {},
  "config_path": "configs/...",
  "git_commit": "...",
  "seed": 42
}
```

## Kaggle Submission Safety Rules

- 禁止一次提交多个候选。
- 禁止为了“试一下”消耗 LB。
- 禁止没有本地 validation summary 就申请提交。
- 禁止同一个无效文件换 message 重复提交。
- 禁止把未记录 seed、config、commit 的文件作为正式提交。
- 禁止 Kaggle API token 进入项目记录。
- 如果当天提交额度接近用尽，必须先停止并告知用户。

## Decision Rules

- 本地验证改善、LB 改善：可考虑晋升为新 base。
- 本地验证改善、LB 变差：优先检查验证泄漏、holdout 代表性和过拟合。
- 本地验证变差、LB 改善：记录但谨慎，不直接晋升。
- 本地验证和 LB 都变差：标记为 `rejected`，除非带来重要认知。
- 连续 3 个版本同方向无收益：停止该路线。
- 如果 CV/LB gap 持续扩大，优先修验证协议，不继续堆模型。

## Cleanup Rules

解决问题后必须盘点过程中产生的临时文件、一次性脚本和失效冗余代码。

严禁未经允许删除任何文件。删除前必须展示：

- 待删除路径
- 为什么无用
- 是否可复现生成
- 删除风险

只有得到用户明确批准后才能删除。

## KISS Rules

- 不引入复杂实验平台，先用目录、Markdown、CSV/JSON 管理。
- 不让 notebook 或控制台日志成为唯一记录。
- 不覆盖旧版本。
- 不把多个主假设塞进一个版本。
- 不只记录分数，必须记录判断。
- 不为了整洁重写历史，历史是证据。

## Next Work Suggestions

当前最值得优先推进：

1. 建立 `versions/`、`submissions/`、`tmp/` 的新实验结构。
2. 为当前最佳历史提交补一个 `historical` version README。
3. 基于 `configs/unified_validation.yml` 生成统一验证报告。
4. 对 absolute baseline 做错误分解。
5. 只在稳定改善的 site 上测试 beam whitelist。
