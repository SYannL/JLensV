from upgrade_gsm8k_generation_artifacts import _upgrade_artifact


def test_upgrade_moves_model_output_and_recognizes_qwen_eos():
    artifact = {
        "status": "generated",
        "timestamp_utc": "now",
        "source": {"canonical_gsm8k": {}},
        "generation": {
            "max_new_tokens": 2048,
            "n_tokens": 4,
            "token_ids": [1, 2, 3, 4],
            "tokens": ["work", "</think>", "#### 2", "<|endoftext|>"],
            "text": "work#### 2",
            "thinking": {"text": "work", "closed": True},
            "final_response": {"text": "#### 2", "n_tokens": 1},
            "ended_with_eos": False,
        },
        "quality": {
            "valid_final_hash_answer": True,
            "complete": False,
        },
    }

    upgraded = _upgrade_artifact(artifact)

    assert list(upgraded)[:3] == ["status", "timestamp_utc", "model_output"]
    assert upgraded["model_output"]["raw_text"].endswith("<|endoftext|>")
    assert "thinking" not in upgraded["generation"]
    assert "final_response" not in upgraded["generation"]
    assert upgraded["quality"]["complete"] is True
