# UniMol Perovskite MCP Server

基于 [bohr-agent-sdk](https://pypi.org/project/bohr-agent-sdk/)（`dp.agent`）的 MCP 服务器，对外暴露一个工具：**`Unimol_Predict_Perovskite_Additive`** —— 输入有机添加剂分子的 SMILES 列表，返回每个分子对钙钛矿电池 PCE 的预测变化量（ΔPCE）。

模型为本仓库训练管线产出的 UniMol 微调权重 `fixed_hyper_param`（回归任务，数据集为 343 条实验记录的 PCE 提升值，见仓库根 README）。

## 工具说明

```
Unimol_Predict_Perovskite_Additive(smiles_list: List[str]) -> Dict[str, float]
```

- 输入：SMILES 列表（仅有机分子；RDKit 校验失败的条目会被自动剔除）
- 输出：`{SMILES: 预测ΔPCE}`，每条预测值叠加 `inductive_bias = 0.2`
- 推理使用 CUDA（`use_cuda: True`），需要 GPU（Bohrium 上原配置为 `1 * NVIDIA T4_16g`）

`utils.py` 中另附训练侧包装 `UniMolTrain` / `UniMolPredict`（基于 `unimol_tools` 的 `MolTrain` / `MolPredict`），自定义超参见 `_default_params()`（unimolv1/84m、kfold=5、scaffold 分组、anomaly_clean、freeze_layers 25/25 等），可用于复训。

## 权重下载

```bash
./download_weights.sh ./weights          # ~843 MB，优先 GitHub Release，回退 Bohrium OSS
export PEROVSKITE_WEIGHT_DIR="$PWD/weights/fixed_hyper_param"
```

sha256 校验值见 `download_weights.sh` 内 `SHA256` 常量。

## 从零部署（服务器）

### 方式一：Bohrium 官方镜像（推荐，与原线上环境一致）

```bash
docker run --gpus all -it \
  -v $(pwd):/unimol_weight_perovskite/unimol_mcp_matmaster \
  registry.dp.tech/dptech/dp/native/prod-21221/unimol:1 bash

cd /unimol_weight_perovskite/unimol_mcp_matmaster
./download_weights.sh /unimol_weight_perovskite/weights   # 或挂载已有权重
cp .env.example .env && vi .env                           # 填 BOHRIUM_USERNAME / PASSWORD / PROJECT_ID
./run_test.sh    # test 环境, 端口 50001
./run_prod.sh    # prod 环境, 端口 50002
```

### 方式二：pip 安装

```bash
pip install bohr-agent-sdk unimol-tools python-dotenv rdkit pandas joblib
```

### 关键环境变量

| 变量 | 说明 |
|---|---|
| `PEROVSKITE_WEIGHT_DIR` | 微调权重目录 `fixed_hyper_param`（server.py 读取） |
| `UNIMOL_WEIGHT_DIR` | `unimol_tools` 预训练底座权重目录（镜像内默认 `/opt/conda/lib/python3.8/site-packages/unimol_tools/weights`，pip 安装时随包自带）。注意：与微调权重是两个不同的东西，缺一不可 |
| `BOHRIUM_USERNAME` / `BOHRIUM_PASSWORD` / `BOHRIUM_PROJECT_ID` | Bohrium 认证与存储（`.env`） |
| `TIEFBLUE_BASE_URL` / `BOHRIUM_*` | Bohrium/Tiefblue 环境端点，见 run 脚本 |

服务以 streamable-http 启动，端点为 `http://<host>:<port>/mcp`。

## 端口约定

| 脚本 | 环境 | 端口 |
|---|---|---|
| `run_test.sh` | Bohrium test 环境 | 50001 |
| `run_prod.sh` | Bohrium prod 环境 | 50002 |

> 历史部署：2025-11 与 2025-12 曾以 Bohrium AppSpace 服务（`unimol` / `uni-mol-perovskite`）形式部署，平台以 `/sse` 网关包装本服务的 streamable-http 端点。

## 文件结构

```
mcp_server/
├── server.py            # MCP 服务器与工具定义（CalculationMCPServer）
├── utils.py             # SMILES 过滤 / CSV 落盘 / UniMol Train/Predict 包装
├── download_weights.sh  # 权重下载（Release 优先 + OSS 回退 + sha256 校验）
├── run_test.sh          # test 环境启动（50001）
├── run_prod.sh          # prod 环境启动（50002）
├── .env.example         # Bohrium 凭证模板（.env 不入库）
└── test/                # 运行时生成的临时 CSV（不入库）
```
