# PPI Interactor Design

[English](#english) | [中文](#中文)

---

## English

Design brand-new protein-protein interaction (PPI) pairs with generative models. The pipeline
has three stages:

1. **Step 1 — GNN interaction predictor**: turn protein sequences into node features with
   ESM-2, build a PPI graph, and train a GraphSAGE + MLP-decoder link-prediction model that
   judges whether any two proteins interact.
2. **Step 2 — Sequence generator**: train an autoregressive Transformer (mini-GPT) to learn to
   generate brand-new protein sequences.
3. **Step 3 — De novo interactor design**: use the Step 2 generator to produce a large batch of
   candidate sequences, filter/dedupe them, rank them with the Step 1 GNN, validate the
   foldability of the top candidates with ESMFold (pLDDT), and export a FASTA file plus PDB
   structure files.

Dataset source: [Kaggle - ppi-dataset](https://www.kaggle.com/datasets/spandansureja/ppi-dataset)
(the scripts download it automatically via `kagglehub` — no manual download or cloud-drive
mounting needed).

### Tested environment / hardware

This project was developed and verified inside a Docker container on the following setup:

| | |
|---|---|
| Docker base image | `nvcr.io/nvidia/pytorch:25.03-py3` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (~96 GB VRAM) |
| NVIDIA driver | 582.08 |
| CUDA | 12.8 |
| PyTorch | `2.7.0a0` (NVIDIA `nv25.03` build, bundled in the base image) |
| cuDNN | 9.8 |
| Container shared memory | 128 GB (`--shm-size 128gb`) |

A GPU is effectively required: Step 1 embeds every protein sequence with ESM-2, and Step 3's
ESMFold validation is GPU-only in practice. A single high-VRAM GPU (this project used ~96 GB)
gives the most headroom, but ESMFold has run successfully on cards with less VRAM (~16-24 GB)
in the wild — `esmfold_model.trunk.set_chunk_size(64)` in `src/fold_validation.py` already
trades speed for lower peak memory. torch_geometric / transformers versions are pinned in
`requirements.txt` to what was verified in this environment (see that file's comments for why).

### Setup

**1. Start the container** (from the host machine, once):

```bash
docker run --gpus all -itd --shm-size 128gb --name ppi_container \
  -v /path/to/ppi_final_project:/code \
  nvcr.io/nvidia/pytorch:25.03-py3
```

Replace `/path/to/ppi_final_project` with the folder that contains `ppi_project/` on your host
(on Windows this looks like `-v D:/docker/ppi_final_project:/code`). The base image already
ships a matching CUDA-enabled PyTorch build, so there's no separate GPU driver setup inside the
container.

**2. Enter the container and install the remaining Python dependencies:**

```bash
docker exec -it ppi_container bash
cd /code/ppi_project
pip install -r requirements.txt
```

`torch` itself is intentionally left unpinned in `requirements.txt` and is already satisfied by
the base image — `pip install` will not try to replace it. Every other package
(`torch-geometric`, `transformers`, `kagglehub`, `umap-learn`, ...) gets installed/verified at
this step.

**3. (Optional) Provide a Kaggle/HuggingFace token** if you hit rate limits: `kagglehub` can
prompt for Kaggle credentials on first download, and setting `HF_TOKEN` avoids HuggingFace Hub
rate limiting when downloading the ESM-2 / ESMFold model weights.

### Running the pipeline

Run the scripts **in this order** from `ppi_project/` — each step reads the checkpoint(s)
produced by the previous one:

| # | Command | What it does | Produces | Needs |
|---|---|---|---|---|
| 1 | `python scripts/run_step1_train_gnn.py` | Downloads the dataset, builds the PPI graph with ESM-2 node embeddings, trains the GNN (500 epochs), evaluates, plots | `checkpoints/best_model.pth`, `checkpoints/graph_data.pt` | — |
| 2 | `python scripts/run_step2_train_generator.py` | Trains the autoregressive sequence generator (50 epochs) | `checkpoints/best_generator.pth` | — |
| 3 | `python scripts/run_pipeline_demo.py --target-node-id 809` | One-shot demo: Generator makes a new sequence, GNN scores it against a chosen existing node | prints the sequence + interaction probability | Step 1 + Step 2 outputs |
| 4 | `python scripts/run_step3_design_and_fold.py` | Full design run: generate many candidates → filter → rank with the GNN → fold the top 5 with ESMFold → validate the pLDDT scale | `outputs/designed_binders.fasta`, `outputs/folded_pdb/*.pdb` | Step 1 + Step 2 outputs |

Notes:
- Step 1 and Step 2 can run in either order relative to each other, but both must finish before
  Step 3 or Step 4 (Step 3 needs `checkpoints/graph_data.pt` from Step 1 and
  `checkpoints/best_generator.pth` from Step 2 together).
- `--target-node-id 809` in Step 3's demo is just an example index into the full graph — pick any
  valid node ID once you've inspected the dataset (or omit the flag to use the default).
- Step 1 (full ESM-2 embedding + 500 GNN epochs) and Step 4 (ESMFold, ~2.8 GB model download on
  first run) are the two slow/heavy steps; everything else finishes quickly.
- All four scripts assume the current working directory is `ppi_project/` (they add the project
  root to `sys.path` relative to their own file location either way, so `python
  scripts/run_step1_train_gnn.py` works from `ppi_project/`).

### Project layout

```
src/
  config.py            # global settings: paths, device, hyperparameters
  data.py              # dataset download, ESM embedding, PPI graph construction, train/val/test split
  models.py            # model definitions: GNNEncoder / EdgeDecoder / LinkPredictor / ProteinMiniGPT
  train_gnn.py         # Step 1: GNN training and evaluation
  train_generator.py   # Step 2: sequence generator training and generation
  pipeline.py          # end of Step 2: end-to-end demo (one new sequence -> GNN score)
  design_binders.py    # Step 3: target selection, generate -> filter -> rank
  fold_validation.py   # Step 3: ESMFold folding validation (pLDDT)
  visualize.py          # all plotting and candidate-sequence sanity-check functions

scripts/
  run_step1_train_gnn.py        # run Step 1, saves checkpoints/graph_data.pt for later steps to reuse
  run_step2_train_generator.py  # run Step 2
  run_pipeline_demo.py          # Step 2 end-to-end single-sequence demo
  run_step3_design_and_fold.py  # run Step 3
```

### Outputs (not version-controlled, generated automatically after running)

- `checkpoints/` — `best_model.pth` (GNN), `best_generator.pth` (sequence generator),
  `graph_data.pt` (cached graph data)
- `outputs/designed_binders.fasta` — the top-ranked candidate sequences from Step 3
- `outputs/folded_pdb/` — 3D structures predicted by ESMFold (view with PyMOL/ChimeraX or
  https://molstar.org/viewer)

### Known limitations / suggested follow-up validation

The GNN score is only a "model hypothesis," not biological evidence. Recommended next steps for
promising candidates:

- AlphaFold-Multimer: check the complex interface with the target (ipTM)
- BLAST: confirm sequence novelty, cross-reference known interactors in STRING/BioGRID
- Wet-lab binding assay: the final validation

---

## 中文

用生成式模型設計全新的蛋白質-蛋白質交互作用（PPI）配對，整套流程分成三個階段：

1. **Step 1 — GNN 交互作用預測器**：把蛋白質序列用 ESM-2 轉成節點特徵，建立 PPI 圖，訓練一個 GraphSAGE + MLP decoder 的連結預測模型，判斷任兩個蛋白質是否會交互作用。
2. **Step 2 — 序列生成器**：訓練一個自迴歸 Transformer（mini-GPT），學習生成全新的蛋白質序列。
3. **Step 3 — De novo interactor 設計**：用 Step 2 的生成器大量產生候選序列，過濾/去重後交給 Step 1 的 GNN 評分排名，再用 ESMFold 驗證候選的可折疊性（pLDDT），最後輸出 FASTA 與 PDB 結構檔。

資料集來源：[Kaggle - ppi-dataset](https://www.kaggle.com/datasets/spandansureja/ppi-dataset)（腳本會用 `kagglehub` 自動下載，不需手動下載或掛載雲端硬碟）。

### 測試環境 / 硬體規格

本專案是在以下設定的 Docker 容器中開發與驗證的：

| 項目 | 內容 |
|---|---|
| Docker base image | `nvcr.io/nvidia/pytorch:25.03-py3` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（約 96 GB VRAM） |
| NVIDIA 驅動版本 | 582.08 |
| CUDA | 12.8 |
| PyTorch | `2.7.0a0`（NVIDIA `nv25.03` build，base image 內建） |
| cuDNN | 9.8 |
| 容器共享記憶體 | 128 GB（`--shm-size 128gb`） |

實務上這個專案幾乎一定要有 GPU：Step 1 要用 ESM-2 對每條蛋白質序列做 embedding，Step 3 的 ESMFold 驗證基本上只能在 GPU 上跑。這裡用的是高 VRAM 卡（約 96 GB）跑起來最寬裕，但 ESMFold 在較小 VRAM（約 16-24 GB）的卡上也有人成功跑過 —— `src/fold_validation.py` 裡的 `esmfold_model.trunk.set_chunk_size(64)` 已經是用犧牲一點速度換取降低顯存尖峰的設定。`requirements.txt` 裡 `torch_geometric`/`transformers` 等套件都鎖在這個環境驗證過的版本（原因寫在該檔案的註解裡）。

### 環境建置

**1. 啟動容器**（在 host 端執行一次即可）：

```bash
docker run --gpus all -itd --shm-size 128gb --name ppi_container \
  -v /path/to/ppi_final_project:/code \
  nvcr.io/nvidia/pytorch:25.03-py3
```

把 `/path/to/ppi_final_project` 換成你 host 上包含 `ppi_project/` 的那個資料夾（Windows 上會長得像 `-v D:/docker/ppi_final_project:/code`）。base image 已經內建對應的 CUDA 版 PyTorch，容器內不需要再另外裝 GPU 驅動。

**2. 進入容器並安裝剩下的 Python 套件：**

```bash
docker exec -it ppi_container bash
cd /code/ppi_project
pip install -r requirements.txt
```

`torch` 在 `requirements.txt` 裡刻意不鎖版本，且 base image 已經滿足這個需求，`pip install` 不會嘗試去把它換掉。其餘套件（`torch-geometric`、`transformers`、`kagglehub`、`umap-learn`……）會在這一步裝好/驗證版本。

**3.（選用）準備 Kaggle / HuggingFace token**：`kagglehub` 第一次下載資料集時可能需要你的 Kaggle 帳號憑證；設定 `HF_TOKEN` 環境變數則可以避免下載 ESM-2 / ESMFold 模型權重時被 HuggingFace Hub 限速。

### 執行順序

在 `ppi_project/` 底下**依序**執行以下指令 —— 每一步都會讀取前一步存下的checkpoint：

| # | 指令 | 做什麼 | 產出 | 前置需求 |
|---|---|---|---|---|
| 1 | `python scripts/run_step1_train_gnn.py` | 下載資料集、用 ESM-2 建 PPI graph 節點特徵、訓練 GNN（500 epoch）、評估、畫圖 | `checkpoints/best_model.pth`、`checkpoints/graph_data.pt` | 無 |
| 2 | `python scripts/run_step2_train_generator.py` | 訓練自迴歸序列生成器（50 epoch） | `checkpoints/best_generator.pth` | 無 |
| 3 | `python scripts/run_pipeline_demo.py --target-node-id 809` | 單次示範：Generator 造一條新序列，GNN 對指定的既有節點打分 | 印出序列與交互作用機率 | 需要 Step 1 + Step 2 的產出 |
| 4 | `python scripts/run_step3_design_and_fold.py` | 完整設計流程：生成大量候選 → 過濾 → GNN 排名 → 用 ESMFold 折疊前 5 名 → 驗證 pLDDT 尺度 | `outputs/designed_binders.fasta`、`outputs/folded_pdb/*.pdb` | 需要 Step 1 + Step 2 的產出 |

補充：
- Step 1 和 Step 2 彼此執行順序可以互換，但兩者都要先跑完才能跑 Step 3 / Step 4（Step 3 同時需要 Step 1 的 `checkpoints/graph_data.pt` 與 Step 2 的 `checkpoints/best_generator.pth`）。
- Step 3 demo 裡的 `--target-node-id 809` 只是示範用的節點編號，實際使用時可以檢視資料集後換成任意合法的 node ID（不加這個參數就用預設值）。
- 最耗時的兩步是 Step 1（完整 ESM-2 embedding + 500 個 GNN epoch）與 Step 4（ESMFold，第一次執行約會下載 2.8 GB 模型），其餘步驟都很快。
- 四個腳本都假設是在 `ppi_project/` 底下執行（它們會依自己檔案位置把專案根目錄加進 `sys.path`，所以在 `ppi_project/` 下跑 `python scripts/run_step1_train_gnn.py` 就可以）。

### 專案結構

```
src/
  config.py            # 路徑、裝置、超參數等全域設定
  data.py              # 資料下載、ESM embedding、PPI graph 建構、train/val/test 切分
  models.py            # GNNEncoder / EdgeDecoder / LinkPredictor / ProteinMiniGPT 等模型定義
  train_gnn.py         # Step 1：GNN 訓練與評估
  train_generator.py   # Step 2：序列生成器訓練與生成
  pipeline.py          # Step 2 尾端：單條新序列 -> GNN 打分 的端到端示範
  design_binders.py    # Step 3：target 挑選、生成 -> 篩選 -> 排名
  fold_validation.py   # Step 3：ESMFold 折疊驗證 (pLDDT)
  visualize.py          # 所有繪圖與序列品質快篩函式

scripts/
  run_step1_train_gnn.py        # 執行 Step 1，並存下 checkpoints/graph_data.pt 供後續步驟重用
  run_step2_train_generator.py  # 執行 Step 2
  run_pipeline_demo.py          # Step 2 端到端單條序列示範
  run_step3_design_and_fold.py  # 執行 Step 3
```

### 輸出（不進版控，執行後自動產生）

- `checkpoints/` — `best_model.pth`（GNN）、`best_generator.pth`（序列生成器）、`graph_data.pt`（快取的圖資料）
- `outputs/designed_binders.fasta` — Step 3 排名前幾名的候選序列
- `outputs/folded_pdb/` — ESMFold 預測的 3D 結構（可用 PyMOL/ChimeraX 或 https://molstar.org/viewer 檢視）

### 已知限制 / 後續驗證方向

GNN 分數只是「模型假設」，並非生物學證據。建議候選再依序做：

- AlphaFold-Multimer：檢查與 target 的複合物介面 (ipTM)
- BLAST：確認序列新穎度，比對 STRING/BioGRID 已知 interactor
- 濕實驗 binding assay：最終驗證
