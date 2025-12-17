# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweep実行エンジン.

W&B Sweepを作成し、Slurmジョブとして実行するエンジン。

新しいアーキテクチャでは、パラメータの取得とRunの作成は
Slurmジョブ内のwandb.agent()が行います。これにより、
RunがSweepに正しく関連付けられます。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lifter.clients import SlurmClient
from lifter.config import Settings
from lifter.job_script import JobScriptBuilder, SlurmDirectives
from lifter.sweep.base import BaseSweepEngine
from lifter.sweep.template import JobGenerator
from lifter.sweep.wandb_client import WandbSweepClient


class SweepEngine(BaseSweepEngine):
    """Slurm Sweep実行エンジン.

    新しいアーキテクチャでは、パラメータの取得とRunの作成は
    Slurmジョブ内のwandb.agent()が行います。
    エンジンはSweepの作成とジョブの投下・監視のみを担当します。
    """

    def __init__(
        self,
        slurm: SlurmClient,
        wandb: WandbSweepClient,
        settings: Settings,
        job_generator: JobGenerator | None = None,
    ):
        """エンジンを初期化.

        Args:
            slurm: Slurmクライアント
            wandb: W&B Sweepクライアント
            settings: 設定
            job_generator: ジョブスクリプト生成関数 (省略時はデフォルト生成)
        """
        super().__init__(wandb, job_generator)
        self.slurm = slurm
        self.settings = settings

    def _get_mode_name(self) -> str:
        """実行モード名を取得."""
        return "Slurm"

    def _default_job_generator(self, sweep_id: str, run_number: int) -> str:
        """デフォルトのジョブスクリプト生成.

        Note: 実際の使用ではテンプレートベースのジェネレータを使用することを推奨。
        このデフォルト実装は crane_x7_vla.training.cli の agent コマンドを呼び出します。

        Args:
            sweep_id: W&B Sweep ID
            run_number: このSweep内での実行番号

        Returns:
            ジョブスクリプトの内容
        """
        slurm_config = self.settings.slurm
        wandb_config = self.settings.wandb

        directives = SlurmDirectives(
            job_name=f"{slurm_config.job_prefix}_sweep_{sweep_id[:8]}_{run_number:03d}",
            partition=slurm_config.partition,
            cpus_per_task=slurm_config.cpus,
            mem=slurm_config.mem,
            gpus=slurm_config.gpus,
            gpu_type=slurm_config.gpu_type,
            time=slurm_config.time,
            container=slurm_config.container,
        )

        builder = JobScriptBuilder(directives)

        # 環境変数
        builder.add_env("PYTHONUNBUFFERED", "1")
        if wandb_config.api_key:
            builder.add_env("WANDB_API_KEY", wandb_config.api_key)
        if wandb_config.entity:
            builder.add_env("WANDB_ENTITY", wandb_config.entity)
        if wandb_config.project:
            builder.add_env("WANDB_PROJECT", wandb_config.project)

        # セットアップ
        builder.add_setup(f"cd {slurm_config.remote_workdir}")
        builder.add_setup("echo '=== Starting Sweep Agent Job ==='")
        builder.add_setup(f"echo 'Sweep ID: {sweep_id}'")
        builder.add_setup(f"echo 'Run Number: {run_number}'")

        # agent コマンドを実行
        # wandb.agent()が自動的にパラメータを取得し、RunをSweepに関連付ける
        builder.add_comment("crane_x7_vla agent コマンドでSweepからパラメータを取得")
        builder.add_command(
            f"python -m crane_x7_vla.training.cli agent openvla "
            f"--sweep-id {sweep_id} "
            f"--entity {wandb_config.entity or ''} "
            f"--project {wandb_config.project or 'crane_x7'} "
            f"--data-root /data "
            f"--output-dir /output "
            f"--count 1"
        )

        return builder.build()

    def _submit_job(self, script_content: str, script_name: str) -> str:
        """Slurmジョブを投下.

        Args:
            script_content: ジョブスクリプトの内容
            script_name: スクリプト名

        Returns:
            ジョブID

        Raises:
            SlurmError: ジョブ投下に失敗した場合
        """
        return self.slurm.submit_script_content(script_content, script_name)

    def _wait_for_job(
        self,
        job_id: str,
        poll_interval: int,
        log_poll_interval: int,
    ) -> str:
        """Slurmジョブ完了を待機.

        Args:
            job_id: 待機するジョブID
            poll_interval: 状態ポーリング間隔 (秒)
            log_poll_interval: ログポーリング間隔 (秒)

        Returns:
            最終状態 (COMPLETED, FAILED, etc.)
        """
        return self.slurm.wait_for_completion(
            job_id,
            poll_interval=poll_interval,
            log_poll_interval=log_poll_interval,
        )

    def _get_job_state(self, job_id: str) -> str | None:
        """Slurmジョブ状態を取得.

        Args:
            job_id: ジョブID

        Returns:
            ジョブの状態、見つからない場合はNone
        """
        return self.slurm.get_job_state(job_id)

    def _is_job_running(self, state: str | None) -> bool:
        """ジョブが実行中かどうかを判定.

        Args:
            state: ジョブの状態

        Returns:
            実行中の場合True
        """
        return state in ("RUNNING", "PENDING", "R", "PD")


# 後方互換性のためのエイリアス
from lifter.sweep.template import create_custom_job_generator  # noqa: E402, F401

if TYPE_CHECKING:
    from lifter.config import LocalSettings
    from lifter.sweep.backends.local import LocalExecutionBackend


class LocalSweepEngine(BaseSweepEngine):
    """ローカルSweep実行エンジン.

    SSH/Slurmを使わずにローカルでSweepを実行する。
    並列実行（max_concurrent_jobs > 1）もサポート。

    Note: この実装は後方互換性のため engine.py に残しています。
    実際のLocalExecutionBackendは sweep/backends/local.py にあります。
    """

    def __init__(
        self,
        backend: LocalExecutionBackend,
        wandb: WandbSweepClient,
        settings: LocalSettings,
        job_generator: JobGenerator | None = None,
    ):
        """エンジンを初期化.

        Args:
            backend: ローカル実行バックエンド
            wandb: W&B Sweepクライアント
            settings: ローカル設定
            job_generator: ジョブスクリプト生成関数 (必須)
        """
        super().__init__(wandb, job_generator)
        self.backend = backend
        self.settings = settings

    def _get_mode_name(self) -> str:
        """実行モード名を取得."""
        return "ローカル"

    def _default_job_generator(self, sweep_id: str, run_number: int) -> str:
        """デフォルトのジョブスクリプト生成.

        Note: ローカル実行ではテンプレートベースのジェネレータを使用することを推奨。
        """
        wandb_config = self.settings.wandb

        script = """#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export WANDB_MODE=online
"""
        if wandb_config.api_key:
            script += f"export WANDB_API_KEY={wandb_config.api_key}\n"
        if wandb_config.entity:
            script += f"export WANDB_ENTITY={wandb_config.entity}\n"
        if wandb_config.project:
            script += f"export WANDB_PROJECT={wandb_config.project}\n"

        script += f"""
echo '=== Starting Local Sweep Agent Job ==='
echo 'Sweep ID: {sweep_id}'
echo 'Run Number: {run_number}'

python -m crane_x7_vla.training.cli agent openvla \\
    --sweep-id {sweep_id} \\
    --entity {wandb_config.entity or ''} \\
    --project {wandb_config.project or 'crane_x7'} \\
    --data-root {self.settings.data_root} \\
    --output-dir {self.settings.output_dir} \\
    --count 1
"""
        return script

    def _submit_job(self, script_content: str, script_name: str) -> str:
        """ローカルジョブを投下.

        Args:
            script_content: ジョブスクリプトの内容
            script_name: スクリプト名

        Returns:
            ジョブID
        """
        return self.backend.submit_job(script_content, script_name)

    def _wait_for_job(
        self,
        job_id: str,
        poll_interval: int,
        log_poll_interval: int,
    ) -> str:
        """ローカルジョブ完了を待機.

        Args:
            job_id: 待機するジョブID
            poll_interval: 状態ポーリング間隔 (秒)
            log_poll_interval: ログポーリング間隔 (秒)

        Returns:
            最終状態 (COMPLETED, FAILED, etc.)
        """
        return self.backend.wait_for_completion(
            job_id,
            poll_interval=poll_interval,
            log_poll_interval=log_poll_interval,
        )

    def _get_job_state(self, job_id: str) -> str | None:
        """ローカルジョブ状態を取得.

        Args:
            job_id: ジョブID

        Returns:
            ジョブの状態、見つからない場合はNone
        """
        return self.backend.get_job_state(job_id)

    def _is_job_running(self, state: str | None) -> bool:
        """ジョブが実行中かどうかを判定.

        Args:
            state: ジョブの状態

        Returns:
            実行中の場合True
        """
        return state in ("RUNNING", None)
