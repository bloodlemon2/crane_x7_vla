# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweep実行エンジンの基底クラス.

SweepEngineとLocalSweepEngineの共通機能を提供する。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.panel import Panel

from lifter.core.console import console
from lifter.sweep.template import JobGenerator
from lifter.utils import generate_timestamp

if TYPE_CHECKING:
    from lifter.sweep.wandb_client import WandbSweepClient


class BaseSweepEngine(ABC):
    """Sweep実行エンジンの基底クラス.

    SweepEngineとLocalSweepEngineの共通機能を提供する。
    サブクラスは_submit_job, _wait_for_job, _get_job_stateを実装する必要がある。
    """

    def __init__(
        self,
        wandb: WandbSweepClient,
        job_generator: JobGenerator | None = None,
    ):
        """エンジンを初期化.

        Args:
            wandb: W&B Sweepクライアント
            job_generator: ジョブスクリプト生成関数
        """
        self.wandb = wandb
        self.job_generator = job_generator or self._default_job_generator

        # 状態ディレクトリ
        self._state_dir = Path(".sweep_state")

    @abstractmethod
    def _default_job_generator(self, sweep_id: str, run_number: int) -> str:
        """デフォルトのジョブスクリプト生成.

        Args:
            sweep_id: W&B Sweep ID
            run_number: このSweep内での実行番号

        Returns:
            ジョブスクリプトの内容
        """
        ...

    @abstractmethod
    def _submit_job(self, script_content: str, script_name: str) -> str:
        """ジョブを投下.

        Args:
            script_content: ジョブスクリプトの内容
            script_name: スクリプト名

        Returns:
            ジョブID
        """
        ...

    @abstractmethod
    def _wait_for_job(
        self,
        job_id: str,
        poll_interval: int,
        log_poll_interval: int,
    ) -> str:
        """ジョブ完了を待機.

        Args:
            job_id: 待機するジョブID
            poll_interval: 状態ポーリング間隔 (秒)
            log_poll_interval: ログポーリング間隔 (秒)

        Returns:
            最終状態 (COMPLETED, FAILED, etc.)
        """
        ...

    @abstractmethod
    def _get_job_state(self, job_id: str) -> str | None:
        """ジョブ状態を取得.

        Args:
            job_id: ジョブID

        Returns:
            ジョブの状態、見つからない場合はNone
        """
        ...

    @abstractmethod
    def _is_job_running(self, state: str | None) -> bool:
        """ジョブが実行中かどうかを判定.

        Args:
            state: ジョブの状態

        Returns:
            実行中の場合True
        """
        ...

    @abstractmethod
    def _get_mode_name(self) -> str:
        """実行モード名を取得.

        Returns:
            "Slurm" または "ローカル" など
        """
        ...

    def _save_state(self, sweep_id: str, run_number: int, job_id: str, mode: str = "slurm") -> None:
        """Sweep状態を保存.

        Args:
            sweep_id: Sweep ID
            run_number: 実行番号
            job_id: Job ID
            mode: 実行モード ("slurm" または "local")
        """
        self._state_dir.mkdir(parents=True, exist_ok=True)
        state_file = self._state_dir / f"{sweep_id}.json"

        # 既存の状態を読み込み
        state: dict[str, Any] = {}
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)

        # 新しい実行を追加
        if "jobs" not in state:
            state["jobs"] = []

        state["jobs"].append(
            {
                "run_number": run_number,
                "job_id": job_id,
                "timestamp": generate_timestamp(),
                "mode": mode,
            }
        )

        # 保存
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def run(
        self,
        sweep_id: str,
        max_runs: int = 10,
        max_concurrent_jobs: int = 1,
        poll_interval: int = 60,
        log_poll_interval: int = 5,
        dry_run: bool = False,
    ) -> None:
        """Sweepを実行.

        Args:
            sweep_id: SweepのID
            max_runs: 最大実行数
            max_concurrent_jobs: 同時実行ジョブ数の上限 (1の場合は逐次実行)
            poll_interval: 状態ポーリング間隔 (秒)
            log_poll_interval: ログポーリング間隔 (秒)
            dry_run: ドライランモード
        """
        mode_name = self._get_mode_name()
        console.print(
            Panel(
                f"Sweep: {sweep_id}\n"
                f"最大実行数: {max_runs}\n"
                f"同時実行数: {max_concurrent_jobs}\n"
                f"ポーリング間隔: {poll_interval}秒\n"
                f"モード: [cyan]{mode_name}[/cyan]\n"
                f"URL: {self.wandb.get_sweep_url(sweep_id)}",
                title=f"Sweep実行開始 ({mode_name})",
                border_style="green",
            )
        )

        if dry_run:
            console.print("[yellow]ドライランモード: 実際にはジョブを投下しません[/yellow]")

        if max_concurrent_jobs == 1:
            self._run_sequential(
                sweep_id=sweep_id,
                max_runs=max_runs,
                poll_interval=poll_interval,
                log_poll_interval=log_poll_interval,
                dry_run=dry_run,
            )
        else:
            self._run_parallel(
                sweep_id=sweep_id,
                max_runs=max_runs,
                max_concurrent_jobs=max_concurrent_jobs,
                poll_interval=poll_interval,
                dry_run=dry_run,
            )

    def _run_sequential(
        self,
        sweep_id: str,
        max_runs: int,
        poll_interval: int,
        log_poll_interval: int,
        dry_run: bool,
    ) -> None:
        """逐次実行モード（リアルタイムログ表示あり）."""
        completed_runs = 0
        mode = "local" if self._get_mode_name() == "ローカル" else "slurm"

        while completed_runs < max_runs:
            run_number = completed_runs + 1
            console.print(f"\n[bold]Run {run_number}/{max_runs}[/bold]")

            # Sweepの状態を確認
            sweep_state = self.wandb.get_sweep_state(sweep_id)
            if sweep_state == "FINISHED":
                console.print("[green]Sweepが終了しました[/green]")
                break

            # ジョブスクリプトを生成
            console.print("[dim]ジョブスクリプトを生成中...[/dim]")
            script_content = self.job_generator(sweep_id, run_number)

            if dry_run:
                console.print("\n[yellow]生成されるジョブスクリプト:[/yellow]")
                console.print("-" * 40)
                console.print(script_content)
                console.print("-" * 40)
                completed_runs += 1
                continue

            # ジョブを投下
            try:
                script_name = f"sweep_{sweep_id[:8]}_{run_number:03d}_{generate_timestamp()}.sh"
                job_id = self._submit_job(script_content, script_name)

                # 状態を保存
                self._save_state(sweep_id, run_number, job_id, mode=mode)

            except Exception as e:
                console.print(f"[red]ジョブ投下に失敗しました: {e}[/red]")
                continue

            # ジョブ完了を待機
            try:
                final_state = self._wait_for_job(
                    job_id,
                    poll_interval=poll_interval,
                    log_poll_interval=log_poll_interval,
                )

                # 結果をログ
                if final_state == "COMPLETED":
                    console.print(f"[green]✓ ジョブ {job_id} が完了しました[/green]")
                else:
                    console.print(f"[red]✗ ジョブ {job_id} が失敗しました: {final_state}[/red]")

            except Exception as e:
                console.print(f"[red]ジョブ待機に失敗しました: {e}[/red]")

            completed_runs += 1

        console.print(f"\n[bold green]Sweep完了: {completed_runs}回実行しました[/bold green]")

    def _run_parallel(
        self,
        sweep_id: str,
        max_runs: int,
        max_concurrent_jobs: int,
        poll_interval: int,
        dry_run: bool,
    ) -> None:
        """並列実行モード（リアルタイムログ表示なし）."""
        running_jobs: dict[str, int] = {}  # {job_id: run_number}
        completed_runs = 0
        submitted_runs = 0
        mode = "local" if self._get_mode_name() == "ローカル" else "slurm"

        console.print(f"[cyan]並列実行モード: 最大{max_concurrent_jobs}ジョブ同時実行[/cyan]")

        while completed_runs < max_runs:
            # Sweepの状態を確認
            sweep_state = self.wandb.get_sweep_state(sweep_id)
            if sweep_state == "FINISHED":
                console.print("[green]Sweepが終了しました[/green]")
                break

            # 完了したジョブを確認
            for job_id in list(running_jobs.keys()):
                state = self._get_job_state(job_id)
                if not self._is_job_running(state):
                    run_number = running_jobs[job_id]
                    del running_jobs[job_id]
                    completed_runs += 1
                    # 結果をログ
                    if state == "COMPLETED":
                        console.print(f"[green]✓ Run {run_number} (Job {job_id}) 完了[/green]")
                    else:
                        console.print(f"[red]✗ Run {run_number} (Job {job_id}) 失敗: {state}[/red]")

            # 新しいジョブを投下（上限まで）
            while len(running_jobs) < max_concurrent_jobs and submitted_runs < max_runs:
                submitted_runs += 1
                run_number = submitted_runs

                console.print(f"\n[bold]Run {run_number}/{max_runs} 投下中...[/bold]")

                script_content = self.job_generator(sweep_id, run_number)

                if dry_run:
                    console.print(f"[yellow]ドライラン: Run {run_number}[/yellow]")
                    completed_runs += 1
                    continue

                try:
                    script_name = f"sweep_{sweep_id[:8]}_{run_number:03d}_{generate_timestamp()}.sh"
                    job_id = self._submit_job(script_content, script_name)
                    running_jobs[job_id] = run_number
                    self._save_state(sweep_id, run_number, job_id, mode=mode)
                    console.print(f"[dim]Run {run_number}: Job {job_id} を投下しました[/dim]")
                except Exception as e:
                    console.print(f"[red]Run {run_number} の投下に失敗: {e}[/red]")

            # 実行中ジョブの状態を表示
            if running_jobs and not dry_run:
                console.print(
                    f"[dim]実行中: {len(running_jobs)}件, "
                    f"完了: {completed_runs}/{max_runs}件[/dim]"
                )
                time.sleep(poll_interval)

        # 残りのジョブを待機
        if running_jobs:
            console.print(f"\n[cyan]残り{len(running_jobs)}件のジョブの完了を待機中...[/cyan]")

        while running_jobs:
            for job_id in list(running_jobs.keys()):
                state = self._get_job_state(job_id)
                if not self._is_job_running(state):
                    run_number = running_jobs[job_id]
                    del running_jobs[job_id]
                    completed_runs += 1
                    if state == "COMPLETED":
                        console.print(f"[green]✓ Run {run_number} (Job {job_id}) 完了[/green]")
                    else:
                        console.print(f"[red]✗ Run {run_number} (Job {job_id}) 失敗: {state}[/red]")
            if running_jobs:
                console.print(f"[dim]残り: {len(running_jobs)}件[/dim]")
                time.sleep(poll_interval)

        console.print(f"\n[bold green]Sweep完了: {completed_runs}回実行しました[/bold green]")
