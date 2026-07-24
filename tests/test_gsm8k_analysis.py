import torch

from analyze_gsm8k_hard_with_lens import (
    THINKING_PROMPT_VERSION,
    _extract_generated_answer,
    _PresencePenaltyLogitsProcessor,
    _render_prompt,
    _split_generation_ids,
)


class FakeTokenizer:
    eos_token_id = 99
    unk_token_id = 0
    chat_template = "present"

    def __init__(self):
        self.template_kwargs = None

    def convert_tokens_to_ids(self, token):
        return {
            "</think>": 42,
            "<|endoftext|>": 98,
            "<|im_end|>": 99,
        }.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "rendered prompt"


def test_hash_answer_must_be_on_its_own_line():
    text = "I was told to finish with `#### 18`, so I will now answer."
    answer, method = _extract_generated_answer(text)

    assert method != "hash_delimiter"
    assert answer == "18"


def test_hash_answer_is_accepted_from_separate_line():
    answer, method = _extract_generated_answer("Finished.\n#### 18\n")

    assert answer == "18"
    assert method == "hash_delimiter"


def test_hash_answer_line_must_contain_only_a_number():
    answer, method = _extract_generated_answer("#### 18 is the answer\n")

    assert method != "hash_delimiter"
    assert answer == "18"


def test_thinking_generation_is_split_at_close_token_and_eos():
    tokenizer = FakeTokenizer()
    result = _split_generation_ids(
        tokenizer,
        [10, 11, 42, 20, 21, 99],
        thinking_enabled=True,
    )

    assert result == {
        "thinking_ids": [10, 11],
        "final_ids": [20, 21],
        "thinking_closed": True,
        "thinking_end_step": 2,
        "ended_with_eos": True,
    }


def test_qwen_endoftext_is_also_recognized_as_eos():
    tokenizer = FakeTokenizer()
    result = _split_generation_ids(
        tokenizer,
        [10, 42, 20, 98],
        thinking_enabled=True,
    )

    assert result["ended_with_eos"] is True
    assert result["thinking_ids"] == [10]
    assert result["final_ids"] == [20]


def test_unclosed_thinking_has_no_final_response():
    tokenizer = FakeTokenizer()
    result = _split_generation_ids(
        tokenizer,
        [10, 11],
        thinking_enabled=True,
    )

    assert result["thinking_ids"] == [10, 11]
    assert result["final_ids"] == []
    assert result["thinking_closed"] is False
    assert result["ended_with_eos"] is False


def test_nonthinking_generation_is_entirely_final_response():
    tokenizer = FakeTokenizer()
    result = _split_generation_ids(
        tokenizer,
        [20, 21, 99],
        thinking_enabled=False,
    )

    assert result["thinking_ids"] == []
    assert result["final_ids"] == [20, 21]
    assert result["thinking_closed"] is True
    assert result["ended_with_eos"] is True


def test_render_prompt_explicitly_controls_thinking_mode():
    tokenizer = FakeTokenizer()
    _, prompt = _render_prompt(
        tokenizer,
        "How many?",
        enable_thinking=True,
    )

    assert prompt == "rendered prompt"
    assert tokenizer.template_kwargs["enable_thinking"] is True
    assert THINKING_PROMPT_VERSION == "thinking-native-user-only-v5"


def test_thinking_prompt_does_not_discuss_thinking_controls():
    tokenizer = FakeTokenizer()
    messages, _ = _render_prompt(
        tokenizer,
        "How many?",
        enable_thinking=True,
    )

    combined = "\n".join(message["content"] for message in messages).lower()
    assert "thinking section" not in combined
    assert "</think>" not in combined
    assert "close" not in combined
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("How many?")


def test_presence_penalty_only_penalizes_unique_generated_tokens():
    processor = _PresencePenaltyLogitsProcessor(1.5, prompt_length=2)
    input_ids = torch.tensor([[8, 9, 3, 3, 5]])
    scores = torch.zeros((1, 10))

    result = processor(input_ids, scores)

    assert result[0, 3].item() == -1.5
    assert result[0, 5].item() == -1.5
    assert result[0, 8].item() == 0
    assert torch.equal(scores, torch.zeros_like(scores))
