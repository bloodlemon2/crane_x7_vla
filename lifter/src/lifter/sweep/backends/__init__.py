# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweep実行バックエンド.

SlurmやローカルなどのジョブExecutionバックエンドを提供。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lifter.clients.slurm import JobInfo


@runtime_checkable
class ExecutionBackend(Protocol):
    """ジョブ実行バックエンドのプロトコル.

    SlurmClientやLocalExecutionBackendが実装するインターフェース。
    """

    def submit_job(self, script_content: str, job_name: str) -> str:
        """ジョブを投下してIDを返す.

        Args:
            script_content: ジョブスクリプトの内容
            job_name: ジョブ名

        Returns:
            ジョブID
        """
        ...

    def get_job_state(self, job_id: str) -> str | None:
        """ジョブ状態を取得.

        Args:
            job_id: ジョブID

        Returns:
            ジョブ状態 (RUNNING, COMPLETED, FAILED, PENDING, etc.)
            ジョブが見つからない場合はNone
        """
        ...

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        log_poll_interval: int = 5,
        callback: Callable[["JobInfo | None", str | None], None] | None = None,
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
            最終状態 (COMPLETED, FAILED, TIMEOUT, CANCELLED, etc.)
        """
        ...


from lifter.sweep.backends.local import LocalExecutionBackend

__all__ = ["ExecutionBackend", "LocalExecutionBackend"]
