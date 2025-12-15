# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""
prompters.py

プロンプトビルダークラス群.
prismatic/models/backbones/llm/prompting/base_prompter.pyおよびvicuna_v15_prompter.pyからの移植。
"""

from abc import ABC, abstractmethod


class PromptBuilder(ABC):
    """マルチターンプロンプトビルダーの抽象クラス."""

    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...


class PurePromptBuilder(PromptBuilder):
    """シンプルなプロンプトビルダー(In/Out形式)."""

    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        super().__init__(model_family, system_prompt)

        # TODO (siddk) =>> Can't always assume LlamaTokenizer --> FIX ME!
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()


# Default System Prompt for LLaVa Models
SYS_PROMPTS = {
    "prismatic": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    "openvla": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
}


class VicunaV15ChatPromptBuilder(PromptBuilder):
    """Vicuna-v1.5チャットプロンプトビルダー.

    Reference: https://huggingface.co/lmsys/vicuna-13b-v1.5
    """

    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = (SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt).strip() + " "

        # LLaMa-2 Specific
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"USER: {msg} ASSISTANT: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.system_prompt + self.wrap_human(message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.system_prompt + self.wrap_human(message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()


__all__ = [
    "SYS_PROMPTS",
    "PromptBuilder",
    "PurePromptBuilder",
    "VicunaV15ChatPromptBuilder",
]
