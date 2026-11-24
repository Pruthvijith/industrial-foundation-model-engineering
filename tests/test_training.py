import json
from pathlib import Path
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from posttrain.data import cases, validate, digest
from posttrain.experiment import run, choice_tokens
from posttrain.alignment import advantages, clipped_objective, run as align


def mini(path):
    vocab = {
        s: i
        for i, s in enumerate(
            [
                "<unk>",
                "<eos>",
                "A",
                "B",
                "C",
                "D",
                "power",
                "pressure",
                "energy",
                "Lab",
                "Options",
                "unit",
                ":",
            ]
        )
    }
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    hf.save_pretrained(path)
    torch.manual_seed(10)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            max_position_embeddings=256,
        )
    ).save_pretrained(path)


def test_split_reproducible_and_group_disjoint():
    train, test = cases("train", 4), cases("test", 4)
    assert digest(train) == digest(cases("train", 4))
    assert {r["group"] for r in train}.isdisjoint(r["group"] for r in test)
    for r in train + test:
        assert f"{r['target']}={r['kind']}" in r["prompt"]


@pytest.mark.parametrize("split,count", [("validation", 1), ("train", 0)])
def test_bad_split(split, count):
    with pytest.raises(ValueError):
        cases(split, count)


def test_duplicate_and_invalid_preferences():
    rows = cases("test", 1)
    for bad in [[], rows + rows, [{**rows[0], "rejected": rows[0]["target"]}]]:
        with pytest.raises(ValueError):
            validate(bad)


def test_adapter_and_preference_downstream(tmp_path):
    base = tmp_path / "base"
    mini(base)
    cfg = json.loads(Path("configs/cpu.json").read_text())
    cfg.update(train_groups=2, test_groups=2, sft_steps=2, dpo_steps=2, rank=2)
    out = tmp_path / "run"
    result = run(cfg, out, str(base))
    assert set(result["choices_accuracy"]) == {"base", "sft", "dpo"}
    assert len(set(result["adapters"].values())) == 2
    assert len(json.loads((out / "dpo.json").read_text())) == 8
    align(str(base), "main", out / "sft", tmp_path / "alignment")
    result = json.loads((tmp_path / "alignment/metrics.json").read_text())
    assert result["groups_sampled"] == 40 and result["student_parameters"] > 0


def test_qlora_profile_fails_without_cuda(tmp_path):
    src = tmp_path / "base"
    mini(src)
    cfg = json.loads(Path("configs/cpu.json").read_text())
    cfg["qlora"] = True
    with pytest.raises(ValueError, match="CUDA"):
        run(cfg, tmp_path / "r", str(src))


def test_token_contract_and_zero_advantage():
    class Ambiguous:
        def encode(self, *args, **kwargs):
            return [1]

    with pytest.raises(ValueError):
        choice_tokens(Ambiguous())
    assert torch.equal(advantages(torch.ones(2, 4)), torch.zeros(2, 4))
    policy = torch.tensor([[0.25, 0.75]]).log()
    actions = torch.tensor([[0, 1]])
    loss = clipped_objective(policy, policy, actions, torch.zeros(1, 2), policy)
    assert abs(loss.item()) < 1e-7


def test_trl_completion_pipeline(tmp_path, monkeypatch):
    import sys
    from posttrain.train_trl import main

    src = tmp_path / "weights"
    mini(src)
    out = tmp_path / "checkpoint"
    monkeypatch.setenv("LOCAL_MODEL", str(src))
    monkeypatch.setattr(sys, "argv", ["train_trl", "--out", str(out), "--steps", "1"])
    main()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["world_size"] == 1 and metrics["optimizer_steps"] == 1
    assert (out / "checkpoint-1" / "trainer_state.json").is_file()
