from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evidence_codec.core.config import RerankerConfig
from evidence_codec.core.types import Candidate

CLASSIFIER_FILE = "classifier.pt"
RERANKER_CONFIG_FILE = "evidence_codec_reranker_config.json"


def encode_query_chunk_pair(
    tokenizer,
    query: str,
    chunk_text: str,
    sequence_length: int = 512,
    query_max_tokens: int = 96,
) -> dict[str, list[int]]:
    """Pack `[CLS] query [SEP] chunk [SEP]` with explicit query truncation."""

    query_ids = tokenizer.encode(
        query,
        add_special_tokens=False,
        truncation=True,
        max_length=query_max_tokens,
    )
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    chunk_budget = max(1, sequence_length - len(query_ids) - special_tokens)
    chunk_ids = tokenizer.encode(
        chunk_text,
        add_special_tokens=False,
        truncation=True,
        max_length=chunk_budget,
    )
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        input_ids = tokenizer.build_inputs_with_special_tokens(query_ids, chunk_ids)
    else:
        query_text = tokenizer.decode(
            query_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        encoded = tokenizer(
            query_text,
            chunk_text,
            add_special_tokens=True,
            truncation="only_second",
            max_length=sequence_length,
        )
        input_ids = encoded["input_ids"]
    if len(input_ids) > sequence_length:
        input_ids = input_ids[:sequence_length]
        sep_token_id = getattr(tokenizer, "sep_token_id", None)
        if sep_token_id is not None:
            input_ids[-1] = sep_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }


def collate_query_chunk_pairs(
    tokenizer,
    examples: list[dict[str, Any]],
    config: RerankerConfig | None = None,
) -> dict[str, Any]:
    """Tokenize and pad query-chunk training or inference examples."""

    config = config or RerankerConfig()
    encodings = [
        encode_query_chunk_pair(
            tokenizer,
            example["query"],
            example["chunk_text"],
            sequence_length=config.sequence_length,
            query_max_tokens=config.query_max_tokens,
        )
        for example in examples
    ]
    batch = tokenizer.pad(
        encodings,
        padding=True,
        return_tensors="pt",
    )
    if "label" in examples[0]:
        import torch

        batch["labels"] = torch.tensor([float(example["label"]) for example in examples])
    if "group_id" in examples[0]:
        import torch

        batch["group_ids"] = torch.tensor([int(example["group_id"]) for example in examples])
    return batch


class ModernBertCrossEncoder:
    """ModernBERT encoder with a scalar head on final-layer CLS."""

    def __init__(self, model_name_or_path: str, config: RerankerConfig | None = None) -> None:
        import torch
        from transformers import AutoConfig, AutoModel

        self.reranker_config = config or RerankerConfig(model_name=model_name_or_path)
        dtype = torch.bfloat16 if self.reranker_config.precision == "bf16" else None
        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=hf_config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.dropout = torch.nn.Dropout(self.reranker_config.dropout)
        self.classifier = torch.nn.Linear(hf_config.hidden_size, 1)

    def to(self, device: str):
        self.encoder.to(device)
        self.classifier.to(device)
        return self

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.dropout.train(mode)
        self.classifier.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.classifier.parameters()

    def __call__(self, **batch):
        return self.forward(**batch)

    def forward(self, **batch):
        accepted = set(inspect.signature(self.encoder.forward).parameters)
        encoder_inputs = {
            key: value
            for key, value in batch.items()
            if key in accepted and key not in {"labels", "group_ids"}
        }
        outputs = self.encoder(**encoder_inputs)
        cls_state = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_state)).squeeze(-1)
        return logits

    def save_pretrained(self, output_dir: str | Path, tokenizer=None) -> None:
        import torch

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), output_dir / CLASSIFIER_FILE)
        (output_dir / RERANKER_CONFIG_FILE).write_text(
            json.dumps(asdict(self.reranker_config), indent=2),
            encoding="utf-8",
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path, config: RerankerConfig | None = None):
        import torch

        model_name_or_path = Path(model_name_or_path) if not isinstance(model_name_or_path, str) else model_name_or_path
        config_path = Path(model_name_or_path) / RERANKER_CONFIG_FILE if isinstance(model_name_or_path, Path) else None
        if config is None and config_path is not None and config_path.exists():
            config = RerankerConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        instance = cls(str(model_name_or_path), config=config)
        classifier_path = Path(model_name_or_path) / CLASSIFIER_FILE
        if classifier_path.exists():
            instance.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        return instance


class CrossEncoderReranker:
    """Lazy ModernBERT-large query-chunk scorer."""

    def __init__(
        self,
        model_name_or_path: str = "answerdotai/ModernBERT-large",
        device: str | None = None,
        config: RerankerConfig | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.config = config or RerankerConfig(model_name=model_name_or_path)
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self._model = ModernBertCrossEncoder.from_pretrained(self.model_name_or_path, config=self.config)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    def score(self, query: str, candidates: list[Candidate], batch_size: int = 32) -> list[float]:
        return self.score_texts(query, [candidate.chunk.text for candidate in candidates], batch_size=batch_size)

    def score_texts(self, query: str, chunk_texts: list[str], batch_size: int = 32) -> list[float]:
        import torch

        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None
        assert self._tokenizer is not None
        scores: list[float] = []
        autocast_enabled = self.device == "cuda" and self.config.precision == "bf16"
        with torch.inference_mode():
            for start in range(0, len(chunk_texts), batch_size):
                texts = chunk_texts[start : start + batch_size]
                examples = [{"query": query, "chunk_text": text} for text in texts]
                batch = collate_query_chunk_pairs(self._tokenizer, examples, config=self.config)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    logits = self._model(**batch)
                scores.extend(float(value) for value in logits.detach().float().cpu().tolist())
        return scores
