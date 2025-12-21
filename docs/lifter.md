# lifter - Slurm ジョブ投下ツール

SSH経由でSlurmクラスターにジョブを投下するツール。W&B Sweep統合によるハイパーパラメータ探索をサポート。

## 目次

- [クイックスタート](#クイックスタート)
- [インストール](#インストール)
- [環境設定](#環境設定)
- [コマンド一覧](#コマンド一覧)
- [W&B Sweep連携](#wb-sweep連携)
- [ジョブスクリプト例](#ジョブスクリプト例)
- [トラブルシューティング](#トラブルシューティング)

## クイックスタート

```bash
cd lifter
pip install -e .

# 環境設定
cp .env.template .env
# .envを編集

# ジョブ投下
lifter submit jobs/train.sh

# ジョブ状態確認
lifter status

# ジョブ完了待機（ログ表示あり）
lifter wait <job_id>
```

## インストール

```bash
cd lifter
pip install -e .
```

### 必要環境

- Python 3.10+
- SSH接続可能なSlurmクラスター
- W&B（Sweep使用時）

## 環境設定

### .env ファイル

```bash
# .env.templateからコピー
cp .env.template .env
```

### SSH接続設定

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `SLURM_SSH_HOST` | SSHホスト名/IPアドレス | **必須** |
| `SLURM_SSH_USER` | SSHユーザー名 | **必須** |
| `SLURM_SSH_PORT` | SSHポート | `22` |
| `SLURM_SSH_AUTH` | 認証方式（`password`/`key`） | `password` |
| `SLURM_SSH_KEY` | SSH秘密鍵パス（key認証時） | `~/.ssh/id_rsa` |

### Slurm設定

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `SLURM_REMOTE_WORKDIR` | リモート作業ディレクトリ | `~/crane_x7_vla` |
| `SLURM_PARTITION` | パーティション名 | `gpu` |
| `SLURM_GPUS` | GPU数 | `1` |
| `SLURM_GPU_TYPE` | GPUタイプ（a100, v100等） | - |
| `SLURM_TIME` | 実行時間（HH:MM:SS） | `24:00:00` |
| `SLURM_MEM` | メモリ | `32G` |
| `SLURM_CPUS` | CPU数 | `8` |
| `SLURM_JOB_PREFIX` | ジョブ名プレフィックス | `crane_x7` |
| `SLURM_CONTAINER` | コンテナイメージ（Pyxis用） | - |

### ジョブ待機設定

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `SLURM_POLL_INTERVAL` | 状態確認間隔（秒） | `60` |
| `SLURM_LOG_POLL_INTERVAL` | ログ取得間隔（秒） | `5` |
| `SLURM_MAX_CONCURRENT_JOBS` | 同時実行ジョブ上限（Sweep用） | `1` |

### W&B設定

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `WANDB_API_KEY` | W&B APIキー | - |
| `WANDB_ENTITY` | W&Bエンティティ | 自動取得 |
| `WANDB_PROJECT` | W&Bプロジェクト名 | `crane_x7_sweep` |

## コマンド一覧

### submit - ジョブ投下

```bash
lifter submit <script> [OPTIONS]
```

| オプション | 説明 |
|-----------|------|
| `--env`, `-e` | 環境設定ファイル（.env） |
| `--dry-run`, `-n` | 実際には投下せず、内容を表示 |
| `--password`, `-p` | SSHパスワード（省略時は対話入力） |

**例:**

```bash
# ジョブ投下
lifter submit jobs/train.sh

# ドライラン
lifter submit jobs/train.sh --dry-run

# 別の環境設定を使用
lifter submit jobs/train.sh -e .env.production
```

### status - 状態確認

```bash
lifter status [JOB_ID] [OPTIONS]
```

| オプション | 説明 |
|-----------|------|
| `--env`, `-e` | 環境設定ファイル |
| `--all`, `-a` | 全ユーザーのジョブを表示 |
| `--password`, `-p` | SSHパスワード |

**例:**

```bash
# 自分のジョブ一覧
lifter status

# 特定ジョブの状態
lifter status 12345

# 全ユーザーのジョブ
lifter status --all
```

### cancel - ジョブキャンセル

```bash
lifter cancel <job_id> [OPTIONS]
```

| オプション | 説明 |
|-----------|------|
| `--env`, `-e` | 環境設定ファイル |
| `--password`, `-p` | SSHパスワード |

**例:**

```bash
lifter cancel 12345
```

### wait - 完了待機

```bash
lifter wait <job_id> [OPTIONS]
```

| オプション | 説明 |
|-----------|------|
| `--env`, `-e` | 環境設定ファイル |
| `--interval`, `-i` | 状態ポーリング間隔（秒） |
| `--timeout`, `-t` | タイムアウト（秒） |
| `--no-log` | ログ表示を無効化 |
| `--log-interval`, `-l` | ログポーリング間隔（秒） |
| `--password`, `-p` | SSHパスワード |

**例:**

```bash
# ログ表示しながら待機
lifter wait 12345

# ログなしで待機
lifter wait 12345 --no-log

# タイムアウト設定
lifter wait 12345 --timeout 3600
```

## W&B Sweep連携

### sweep start - 新規Sweep開始

```bash
lifter sweep start <config.yaml> [OPTIONS]
```

| オプション | 説明 |
|-----------|------|
| `--max-runs`, `-n` | 最大実行数 |
| `--max-concurrent`, `-c` | 同時実行ジョブ上限 |
| `--poll-interval`, `-i` | 状態ポーリング間隔（秒） |
| `--log-interval`, `-l` | ログポーリング間隔（秒） |
| `--template`, `-t` | ジョブテンプレートファイル |
| `--dry-run` | ドライラン |
| `--local` | ローカル実行（SSH/Slurm不要） |
| `--password`, `-p` | SSHパスワード |

**例:**

```bash
# Sweep開始（10回実行）
lifter sweep start sweeps/openvla.yaml --max-runs 10

# 並列実行（3ジョブ同時）
lifter sweep start sweeps/openvla.yaml --max-runs 20 --max-concurrent 3
```

### sweep resume - Sweep再開

```bash
lifter sweep resume <sweep_id> [OPTIONS]
```

**例:**

```bash
# 既存Sweepを再開
lifter sweep resume abc123def --max-runs 10
```

### ローカルSweep実行

SSH/Slurmを使わずにローカル環境でSweepを実行できます。

```bash
# ローカルでSweep開始（テンプレート必須）
lifter sweep start examples/sweeps/sweep_openvla.yaml \
  --local \
  --template examples/templates_local/openvla_sweep.sh \
  --max-runs 5

# ローカルで並列実行
lifter sweep start examples/sweeps/sweep_pi0.yaml \
  --local \
  --template examples/templates_local/pi0_sweep.sh \
  --max-runs 10 \
  --max-concurrent 2

# ローカルSweep再開
lifter sweep resume <SWEEP_ID> \
  --local \
  --template examples/templates_local/openvla_sweep.sh
```

**注意点:**
- `--local`使用時は`--template`が必須
- ローカル用テンプレートは`examples/templates_local/`にあります
- SSH/Slurm設定（`.env`のSSH/Slurm関連変数）は不要
- W&B設定（`WANDB_API_KEY`等）は必要

**ローカル用テンプレート:**
- `openvla_sweep.sh` - OpenVLA用
- `pi0_sweep.sh` - Pi0用
- `pi05_sweep.sh` - Pi0.5用

### sweep status - Sweep状態確認

```bash
lifter sweep status <sweep_id> [OPTIONS]
```

**例:**

```bash
lifter sweep status abc123def
```

### VLA CLIのagentコマンド

VLAトレーニングCLIには組み込みのW&B Sweepエージェント機能があります。lifterのsweepと組み合わせて使用：

```bash
# VLA CLIのagentコマンドを直接使用
python -m crane_x7_vla.training.cli agent pi0.5 \
  --sweep-id <SWEEP_ID> \
  --entity <WANDB_ENTITY> \
  --project <WANDB_PROJECT> \
  --data-root /workspace/data/tfrecord_logs

# lifterでSweepを開始し、VLA CLIのagentが実行される
lifter sweep start examples/sweeps/sweep_pi0.yaml \
  --local \
  --template examples/templates_local/pi0_sweep.sh
```

### Sweep設定ファイル例

```yaml
# sweeps/openvla.yaml
program: train.py
method: bayes
metric:
  name: eval/loss
  goal: minimize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-4
  batch_size:
    values: [8, 16, 32]
  lora_rank:
    values: [16, 32, 64]
```

## テンプレート機能

lifterはJinja2ベースのテンプレートエンジンを使用しており、ジョブスクリプト内で環境変数の展開、条件分岐、デフォルト値の設定が可能です。

### 基本的な変数展開

`.env`ファイルで定義した変数をスクリプト内で使用できます。

```bash
# .env
GPU_TYPE=a100
NUM_GPUS=2
EXPERIMENT_NAME=crane_x7_v1

# job.sh
#!/bin/bash
#SBATCH --gres=gpu:{{ GPU_TYPE }}:{{ NUM_GPUS }}

echo "Starting experiment: {{ EXPERIMENT_NAME }}"
```

### 条件分岐

```bash
{% if GPU_TYPE %}
#SBATCH --gres=gpu:{{ GPU_TYPE }}:{{ NUM_GPUS | default(1) }}
{% else %}
# CPUモードで実行
{% endif %}
```

### デフォルト値

変数が未定義の場合のデフォルト値を指定できます。

```bash
#SBATCH --mem={{ MEM | default('32G') }}
#SBATCH --cpus-per-task={{ CPUS | default(8) }}
```

### フィルタ

Jinja2の組み込みフィルタが使用できます。

```bash
# 大文字/小文字変換
echo "GPU: {{ GPU_TYPE | upper }}"
echo "user: {{ USER | lower }}"
```

### Sweep用テンプレート

W&B Sweep実行時には、自動的に以下の変数が利用可能です：

- `SWEEP_ID`: W&B Sweep ID
- `RUN_NUMBER`: Sweep内での実行番号

```bash
#!/bin/bash
#SBATCH --job-name=sweep_{{ SWEEP_ID[:8] }}_{{ RUN_NUMBER }}

python -m crane_x7_vla.training.cli agent openvla \
  --sweep-id {{ SWEEP_ID }} \
  --count 1
```

## ジョブスクリプト例

### 基本的なトレーニングジョブ

```bash
#!/bin/bash
#SBATCH --job-name=crane_x7_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# 環境設定
source ~/.bashrc
conda activate vla

# トレーニング実行
cd ~/crane_x7_vla/vla
python -m crane_x7_vla.training.cli train openvla \
  --data-root /data/tfrecord_logs \
  --experiment-name crane_x7_openvla \
  --training-batch-size 16
```

### コンテナ使用（Pyxis/Enroot）

```bash
#!/bin/bash
#SBATCH --job-name=crane_x7_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --container-image=noppdev/vla:latest
#SBATCH --container-mounts=/data:/data:ro

cd /workspace
python -m crane_x7_vla.training.cli train openvla \
  --data-root /data/tfrecord_logs
```

## トラブルシューティング

### SSH接続エラー

```
ConnectionRefusedError: [Errno 111] Connection refused
```

- ホスト名/IPアドレスを確認
- SSHポートを確認
- ファイアウォール設定を確認

### 認証エラー

```
AuthenticationException: Authentication failed
```

- ユーザー名を確認
- パスワードまたは秘密鍵を確認
- `SLURM_SSH_AUTH`の設定を確認

### ジョブが開始されない

```
squeue: Job 12345 is PENDING
```

- パーティションのリソース状況を確認: `sinfo`
- 要求リソースが利用可能か確認
- 優先度を確認: `sprio`

### W&B Sweepエラー

```
WandbSweepError: Failed to create sweep
```

- `WANDB_API_KEY`を確認
- ネットワーク接続を確認
- W&Bにログイン: `wandb login`

### ログが表示されない

- `--no-log`オプションが指定されていないか確認
- `SLURM_LOG_POLL_INTERVAL`が適切か確認
- 出力ファイルのパスを確認

### タイムアウト

- `--timeout`オプションで延長
- `SLURM_POLL_INTERVAL`を調整

## ライセンス

- **オリジナルコード**: MIT License（Copyright 2025 nop）
