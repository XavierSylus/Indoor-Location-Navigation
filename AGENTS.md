# Project Agent Rules

## Correction-Derived Rules

- 自动工作流搭建完成后，后续迭代必须优先围绕分数提升假设，而不是继续扩展工具本身。
- 项目迭代核心不是“换模型、调参数、ensemble、看分数”，而是用实验结果反推误差来源，再创造新的信息源去攻击该误差来源。
- 每次新实验迭代开始前，必须先将上一状态作为可追溯版本提交并成功推送到 `https://github.com/XavierSylus/Indoor-Location-Navigation`；备份不得包含 Kaggle 凭证、原始竞赛数据、模型权重或可再生的大型缓存。
- 每次备份提交在上传前必须先写明标注标题，统一使用 `backup(<目标版本>): <状态说明>` 格式，标题必须能够辨识该备份对应的下一轮版本。
- 当用户明确要求在 Kaggle 训练时，禁止在本地读取竞赛数据执行任何训练或训练 smoke；本地仅允许语法检查、单元测试、静态数据契约检查和上传包校验，实际训练与训练输出生成必须在 Kaggle 完成。

## Kaggle Iteration Rules

- 每个版本必须能回答：为什么做、改了什么、结果如何、下一步怎么走。
- 每个版本只验证一个主要假设；如果需要同时新增模块、组装模块和调参，必须拆成多个版本。
- 版本迭代类型必须明确区分：
  - `add_module`: 新增独立信息源或能力，用来证明新模块是否有价值。
  - `assemble_module`: 只组合已有有效组件，不在同一版本里发明新模型或新规则。
  - `tune_module`: 只调整已有模块的参数、权重、seed、fold、阈值或正则。
- 实验类型必须明确区分：
  - `cv_probe`: 本地验证探针，默认不消耗 Kaggle LB。
  - `lb_direction_probe`: 方向枪，只回答一个低风险 LB 方向问题。
  - `leaderboard_attempt`: 冲榜尝试，必须由本地验证或方向枪支撑。
- 探针实验的目标是降低不确定性，不是一次拿最优分。分数不好但带来明确判断，也必须记录为有效信息。
- integration 版本只能在至少一个方向枪或多个独立探针支持同一方向后执行；必须写清楚吸收了哪些 component、丢弃了哪些 component，以及为什么这些信号可以共存。
- 失败路线不能通过堆叠复杂度掩盖。若同方向连续 3 个版本无收益，停止该路线并回到误差诊断。
- 面向 4m 目标的诊断版本不能只做 taxonomy 分类，必须输出 score waterfall 和 excess-over-4m 贡献账本，量化每类误差分别贡献了多少、修掉后理论上能到哪里。
- v004 的目标固定升级为 `v004_error_source_score_waterfall`：回答当前分数距离 4m 的主要误差贡献来自哪里，并据此给出 v005 的唯一推荐方向。
- 不同评价口径必须先桥接再比较。LB MPE、holdout waypoint MAE、delta-leg MAE、reranker CV MAE 不能混为一谈；必须用 `metric_bridge.csv` 标记 scope、n_points、split_type、uses_gt、comparable_to_submission。
- floor error 是一级误差来源，必须单独统计；即使当前 floor 看起来很准，也必须显式报告 floor 错误对尾部误差的贡献。
- 面向 4m 的主要优化指标必须包含 `excess_over_4m = max(error - 4, 0)`，并按 site/floor/path/group 汇总 tail-error 贡献。
- 任何 residual、bias、path-shift 或 oracle 修复只能作为 diagnostic-only，除非经过 out-of-fold 或 submission-safe 规则验证，不得直接转成可提交规则。

## Experiment DNA Rules

- DNA 是每个方案的可继承“基因组”。`run_manifest.json` 负责回答“用哪份代码、哪条命令、哪份数据复跑”，`dna.json` 负责回答“这个方案由哪些可替换、可比较、可蒸馏的 gene 组成”。
- 每个正式版本必须保存 `dna.json`；没有 `dna.json` 的版本不能作为后续 base，也不能称为已纳入实验资产体系。
- 每个方案必须映射到多个 gene，至少包括：
  - `base`: 继承自哪些版本或外部方案。
  - `strategy`: 本方案要攻击的核心假设或误差来源。
  - `feature_component`: 特征、OOF、artifact、后处理等可迁移组件。
  - `model_component`: 模型族、推理组件或可复用模型信号。
  - `change_operator`: 相对父方案的新增、替换、微调或组装方式。
  - `validation`: CV、LB、诊断报告、hidden-safe 输出校验等证据路径。
  - `execution`: 复现命令、配置入口、随机种子和运行环境。
  - `submission`: test prediction、Kaggle 输出或提交候选资产。
- 新版本不是“复制一个方案”，而是“继承父 DNA，并新增、替换或微调少数 gene”。如果一个方案同时替换多个独立 gene，必须拆成多个版本，除非它是明确的 `assemble_module` 版本。
- 复盘时必须先比较 DNA，再比较分数：判断哪个 gene 贡献了 CV、LB、hidden-safe、稳定性或失败信号。
- 开源方案被吸收后也必须生成 DNA。外部来源、可保留组件、丢弃组件和泄漏风险必须沉淀为 gene，不能只保留 notebook 链接或口头结论。
- CSV 或 ledger 中必须暴露 DNA 索引字段，例如 `dna_id`、`dna_parent`、`dna_file`、`dna_genes`；完整基因结构只放在 `dna.json`，表格只放紧凑摘要。

## Next Step DNA Planning Rules

- 当用户说“下一步”“继续”“next step”或同义短句，并且没有明确指定只看状态、只写计划或只提交 LB 时，agent 不应反问用户选择哪条路，而必须先读取当前 ledger、版本 README/DNA、最近成功版本和最近失败版本的证据，再自行判断下一轮动作。
- 下一轮动作只能在三类中选择一种：
  - 创新新的 gene：当最近路线连续失败、CV/LB gap 暴露新误差来源、现有 gene 无法解释失败，或外部方案暗示新 component 时使用；通常落账为 `experiment_type=cv_probe`、`iteration_mode=add_module`，默认不提交 LB。
  - 微调已有 gene：当某个 DNA 已有本地 CV、OOF、残差相关性、可见代理或 LB 证据，但幅度、符号、cap、gate、fold、seed、正则、样本权重或分段策略仍不稳定时使用；通常落账为 `iteration_mode=tune_module`，一次只调整少数明确 gene。
  - 集成 gene 冲榜：当多个已验证 component 可以组合，或当前 leader 需要用已证实 DNA 做低风险组装时使用；通常落账为 `iteration_mode=assemble_module`。只有在已有 validation summary、hidden-safe 证据和用户明确授权后，才能消耗 LB。
- “下一步”不是只写建议，而是推进一个新版本。agent 必须选择下一个 `vXXX_short_name`，创建或更新对应 `versions/vXXX_short_name/`，并产出至少一个可追溯 artifact：本地 CV、诊断报告、可验证输出、远端输出校验或明确失败证据。
- 每次选择前，必须用当前 ledger 判断最值得投入下一轮算力的瓶颈：当前最好 LB、最好本地 CV、最近 LB 负向证据、最近正向 DNA、最近失败路线、是否已有提交授权。
- 决策输出必须包含：下一步判定、选择理由、本轮版本、父 DNA、本轮 gene 变化、验证方式。
- 如果判断为“集成 gene 冲榜”但没有提交授权，agent 只能完成本地或远端校验版本，并将其记录为候选，不得自行消耗 LB。

## Submission Rules

- 当前项目 Kaggle slug 固定为 `indoor-location-navigation`。不得把其他比赛的 slug、notebook/code competition 提交流程或 kernel 规则套用到本项目。
- Kaggle 凭证只能放在用户目录，例如 `%USERPROFILE%\.kaggle\kaggle.json`，不得作为项目文件管理。
- 每次 Kaggle 提交前必须展示：version、competition slug、submission file、message、local validation、seed、config path、预计消耗 1 次提交额度。
- 只有用户明确授权“提交某个具体版本/文件”后，才能调用 Kaggle submit。
- 禁止一次提交多个候选；禁止为了“试一下”消耗 LB；禁止没有 validation summary 就申请提交；禁止同一个无效文件换 message 重复提交。

## Recordkeeping Rules

- 正式版本必须保存 `dna.json`、`run_manifest.json`、`reproduce.ps1`、`validation_summary.json` 和 `code_snapshot/SHA256SUMS.txt`。
- `run_manifest.json` 必须记录 exact command、git commit、seed、config path、input/output path、validation report、experiment_type 和 iteration_mode。
- 关键结论必须沉淀到版本 README 和 `EXPERIMENT_LEDGER.md`，不能只存在于 notebook、控制台输出或聊天记录。
- 旧版本可以标记为 historical/backfill，但不能伪造成完整可复现实验。
