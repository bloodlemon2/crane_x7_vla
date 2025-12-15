# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""ローカル実行バックエンド.

SSH/Slurmを使わずにローカルでジョブを実行する。
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from slurm_submit.core.console import console

if TYPE_CHECKING:
    pass


def _format_duration(seconds: float) -> str:
    """秒数を読みやすい形式に変換."""
    duration = timedelta(seconds=int(seconds))
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


@dataclass
class LocalJob:
    """ローカルジョブ情報."""

    job_id: str
    process: subprocess.Popen[bytes]
    log_file: Path
    script_path: Path
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> str:
        """経過時間を取得."""
        return _format_duration(time.time() - self.start_time)


@dataclass
class LocalJobMonitor:
    """ローカルジョブ監視状態."""

    job_id: str
    start_time: float = field(default_factory=time.time)
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    last_log_offset: int = 0
    state: str = "UNKNOWN"

    @property
    def elapsed_time(self) -> str:
        """経過時間を取得."""
        return _format_duration(time.time() - self.start_time)

    def add_log_lines(self, lines: list[str]) -> None:
        """ログ行を追加."""
        for line in lines:
            stripped = line.rstrip()
            if stripped:
                self.log_lines.append(stripped)


class LocalExecutionBackend:
    """ローカル実行バックエンド.

    SSH/Slurmを使わずに、subprocessでローカル実行する。
    """

    def __init__(self, workdir: Path | None = None):
        """バックエンドを初期化.

        Args:
            workdir: 作業ディレクトリ（省略時はカレントディレクトリ）
        """
        self.workdir = workdir or Path.cwd()
        self._jobs: dict[str, LocalJob] = {}
        self._logs_dir = self.workdir / "logs" / "local_sweep"

        # クリーンアップ登録
        atexit.register(self._cleanup_all_jobs)

    def _cleanup_all_jobs(self) -> None:
        """全ジョブをクリーンアップ."""
        for job in list(self._jobs.values()):
            try:
                if job.process.poll() is None:
                    job.process.terminate()
                    try:
                        job.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        job.process.kill()
            except Exception:
                pass

    def submit_job(self, script_content: str, job_name: str) -> str:
        """シェルスクリプトをローカルで実行.

        Args:
            script_content: ジョブスクリプトの内容
            job_name: ジョブ名

        Returns:
            ジョブID
        """
        # ログディレクトリ作成
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        # ユニークなジョブID生成
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        log_file = self._logs_dir / f"{job_name}_{job_id}.log"
        script_path = self._logs_dir / f"{job_name}_{job_id}.sh"

        # スクリプトを一時ファイルに保存
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        console.print(f"[dim]ローカルジョブを開始: {job_id}[/dim]")
        console.print(f"[dim]ログファイル: {log_file}[/dim]")

        # サブプロセスで実行
        log_handle = open(log_file, "w")
        process = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=self.workdir,
            start_new_session=True,  # プロセスグループを分離
        )

        self._jobs[job_id] = LocalJob(
            job_id=job_id,
            process=process,
            log_file=log_file,
            script_path=script_path,
        )

        console.print(f"[green]ローカルジョブが開始されました: {job_id} (PID: {process.pid})[/green]")
        return job_id

    def get_job_state(self, job_id: str) -> str | None:
        """ジョブ状態を取得.

        Args:
            job_id: ジョブID

        Returns:
            ジョブ状態、見つからない場合はNone
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        return_code = job.process.poll()
        if return_code is None:
            return "RUNNING"
        elif return_code == 0:
            return "COMPLETED"
        else:
            return "FAILED"

    def cancel_job(self, job_id: str) -> None:
        """ジョブをキャンセル.

        Args:
            job_id: キャンセルするジョブID
        """
        job = self._jobs.get(job_id)
        if not job:
            console.print(f"[yellow]ジョブ {job_id} が見つかりません[/yellow]")
            return

        if job.process.poll() is None:
            # プロセスグループ全体を終了
            try:
                os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(job.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            console.print(f"[yellow]ジョブ {job_id} をキャンセルしました[/yellow]")
        else:
            console.print(f"[dim]ジョブ {job_id} は既に終了しています[/dim]")

    def get_log_tail(self, log_path: Path, offset: int = 0, max_lines: int = 10) -> tuple[list[str], int]:
        """ログファイルの末尾を取得.

        Args:
            log_path: ログファイルのパス
            offset: 読み取り開始位置（バイト）
            max_lines: 取得する最大行数

        Returns:
            (新しい行のリスト, 新しいオフセット) のタプル
        """
        if not log_path.exists():
            return [], offset

        try:
            file_size = log_path.stat().st_size
        except OSError:
            return [], offset

        if file_size <= offset:
            return [], offset

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                if offset > 0:
                    f.seek(offset)
                content = f.read()
                lines = content.split("\n")
                # 最後のmax_lines行を取得
                if len(lines) > max_lines:
                    lines = lines[-max_lines:]
                return lines, file_size
        except OSError:
            return [], offset

    def _build_monitor_display(self, monitor: LocalJobMonitor) -> Group:
        """監視表示を構築."""
        # ヘッダー情報
        header = Text()
        header.append("Job ID: ", style="bold")
        header.append(monitor.job_id, style="cyan")
        header.append("  State: ", style="bold")

        state_style = {
            "RUNNING": "green",
            "COMPLETED": "blue",
            "FAILED": "red",
            "UNKNOWN": "yellow",
        }.get(monitor.state, "white")
        header.append(monitor.state, style=state_style)

        header.append("  Elapsed: ", style="bold")
        header.append(monitor.elapsed_time, style="magenta")

        # ログ表示（最新5行）
        log_lines = list(monitor.log_lines)[-5:]
        log_text = "\n".join(log_lines) if log_lines else "[dim]ログ待機中...[/dim]"

        log_panel = Panel(
            log_text,
            title="Output",
            border_style="dim",
            padding=(0, 1),
        )

        return Group(header, log_panel)

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        log_poll_interval: int = 5,
        callback: Callable[[Any, str | None], None] | None = None,
        show_log: bool = True,
    ) -> str:
        """ジョブ完了を待機.

        Args:
            job_id: 待機するジョブID
            poll_interval: 状態ポーリング間隔 (秒)
            log_poll_interval: ログポーリング間隔 (秒)
            callback: 状態変化時のコールバック
            show_log: ログを表示するかどうか

        Returns:
            最終状態 (COMPLETED, FAILED, etc.)
        """
        job = self._jobs.get(job_id)
        if not job:
            console.print(f"[red]ジョブ {job_id} が見つかりません[/red]")
            return "UNKNOWN"

        monitor = LocalJobMonitor(job_id=job_id)
        last_state_check = 0.0
        last_log_check = 0.0

        def update_display(live: Live) -> str | None:
            """表示を更新し、終了状態があれば返す."""
            nonlocal last_state_check, last_log_check

            current_time = time.time()
            final_state: str | None = None

            # 状態チェック
            if current_time - last_state_check >= poll_interval or last_state_check == 0:
                last_state_check = current_time

                state = self.get_job_state(job_id)

                if state != monitor.state:
                    monitor.state = state or "UNKNOWN"
                    if callback:
                        callback(None, state)

                # 完了判定
                if state == "COMPLETED":
                    final_state = "COMPLETED"
                elif state == "FAILED":
                    final_state = "FAILED"
                elif state is None:
                    final_state = "UNKNOWN"

            # ログチェック（より頻繁に）
            if show_log and current_time - last_log_check >= log_poll_interval:
                last_log_check = current_time
                new_lines, new_offset = self.get_log_tail(
                    job.log_file, monitor.last_log_offset, max_lines=100
                )
                if new_lines:
                    monitor.add_log_lines(new_lines)
                    monitor.last_log_offset = new_offset

            # 表示更新
            live.update(self._build_monitor_display(monitor))

            return final_state

        # 初期状態を取得
        initial_state = self.get_job_state(job_id)
        monitor.state = initial_state or "UNKNOWN"

        with Live(
            self._build_monitor_display(monitor),
            console=console,
            refresh_per_second=1,
            transient=False,
        ) as live:
            while True:
                final_state = update_display(live)
                if final_state:
                    # 最終ログを取得
                    if show_log:
                        new_lines, _ = self.get_log_tail(job.log_file, monitor.last_log_offset, max_lines=100)
                        if new_lines:
                            monitor.add_log_lines(new_lines)
                        live.update(self._build_monitor_display(monitor))

                    # 最終状態を表示
                    if final_state == "COMPLETED":
                        console.print(f"\n[bold green]✓ ジョブ {job_id} が完了しました[/bold green]")
                    else:
                        console.print(f"\n[bold red]✗ ジョブ {job_id} が失敗しました ({final_state})[/bold red]")
                        self._print_error_details(job)
                    return final_state
                time.sleep(1)

    def _print_error_details(self, job: LocalJob) -> None:
        """ジョブ失敗時のエラー詳細を表示.

        Args:
            job: 失敗したジョブ
        """
        return_code = job.process.poll()
        console.print(f"[red]  ExitCode: {return_code}[/red]")

        # ログファイルの末尾を表示
        if job.log_file.exists():
            try:
                with open(job.log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    last_lines = lines[-20:] if len(lines) > 20 else lines

                    console.print("\n[bold yellow]--- Log (last 20 lines) ---[/bold yellow]")
                    for line in last_lines:
                        console.print(f"[dim]{line.rstrip()}[/dim]")
            except OSError:
                pass
