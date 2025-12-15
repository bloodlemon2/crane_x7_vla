# lifter

SSH経由でSlurmクラスターにジョブを投下するツール。W&B Sweep統合によるハイパーパラメータ探索をサポート。

## インストール

```bash
cd lifter
pip install -e .
```

## クイックスタート

```bash
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

## コマンド

| コマンド | 説明 |
|---------|------|
| `lifter submit <script>` | ジョブ投下 |
| `lifter status [job_id]` | 状態確認 |
| `lifter cancel <job_id>` | ジョブキャンセル |
| `lifter wait <job_id>` | 完了待機 |
| `lifter sweep start <config.yaml>` | W&B Sweep開始 |
| `lifter sweep resume <sweep_id>` | Sweep再開 |
| `lifter sweep status <sweep_id>` | Sweep状態確認 |

## W&B Sweep

```bash
# リモートSlurmクラスターでSweep
lifter sweep start sweeps/openvla.yaml --max-runs 10

# ローカル実行（SSH/Slurm不要）
lifter sweep start sweeps/openvla.yaml \
  --local \
  --template templates_local/openvla_sweep.sh \
  --max-runs 5
```

## 詳細ドキュメント

[docs/lifter.md](../docs/lifter.md)を参照してください。

## ライセンス

MIT License
