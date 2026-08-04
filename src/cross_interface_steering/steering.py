from __future__ import annotations

import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decomposition import contrast_directions, decompose_directions
from .metrics import js_divergence


@dataclass
class ChoiceSpec:
    negative_label: str = "morally wrong"
    positive_label: str = "morally right"
    negative_letter: str = "A"
    positive_letter: str = "B"

    def format_prompt(self, prompt: str) -> str:
        label_by_letter = {
            self.negative_letter: self.negative_label,
            self.positive_letter: self.positive_label,
        }
        ordered_letters = sorted(label_by_letter)
        return (
            f"{prompt}\n"
            f"{ordered_letters[0]}. {label_by_letter[ordered_letters[0]]}\n"
            f"{ordered_letters[1]}. {label_by_letter[ordered_letters[1]]}\n"
            "Answer:"
        )

    @property
    def negative_choice_index(self) -> int:
        return 0 if self.negative_letter == "A" else 1

    @property
    def positive_choice_index(self) -> int:
        return 0 if self.positive_letter == "A" else 1


def model_alias(model_name: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name).strip())
    return alias.strip("_") or "model"


def load_tokenizer_and_model(model_name: str, *, device_map: str = "auto", torch_dtype: str = "auto") -> tuple[Any, Any]:
    # This package is PyTorch-only. Prevent Transformers from importing an
    # unrelated TensorFlow installation, which may be ABI-incompatible with
    # the NumPy version used by the experiment environment.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch_dtype
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def decoder_layers(model: Any) -> list[Any]:
    candidates = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise AttributeError("Could not locate decoder layers on model")


def resolve_layer_index(layer_index: int, n_layers: int) -> int:
    idx = int(layer_index)
    if idx < 0:
        idx = n_layers + idx
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"Layer index {layer_index} resolves to {idx}, outside 0..{n_layers - 1}")
    return idx


def choice_token_ids(tokenizer: Any, choices: list[str]) -> list[int]:
    ids = []
    for choice in choices:
        candidates = [f" {choice}", choice, f"\n{choice}"]
        token_id = None
        for text in candidates:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                token_id = int(encoded[0])
                break
        if token_id is None:
            encoded = tokenizer.encode(choice, add_special_tokens=False)
            token_id = int(encoded[-1])
        ids.append(token_id)
    return ids


def _input_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _last_token_indices(attention_mask: Any) -> Any:
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    positions = positions.expand_as(attention_mask)
    return positions.masked_fill(attention_mask.eq(0), -1).max(dim=1).values


def _gather_last_token(tensor: Any, attention_mask: Any) -> Any:
    import torch

    positions = _last_token_indices(attention_mask).to(tensor.device)
    batch_indices = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[batch_indices, positions, :]


def token_indices_from_character_offsets(
    offsets: np.ndarray,
    attention_mask: np.ndarray,
    character_positions: list[int | None],
) -> np.ndarray:
    """Resolve character boundaries to token positions in full prompts."""
    offsets = np.asarray(offsets)
    attention_mask = np.asarray(attention_mask)
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError("offsets must have shape [batch, sequence, 2]")
    if attention_mask.shape != offsets.shape[:2]:
        raise ValueError("attention_mask must match the first two offset dimensions")
    if len(character_positions) != offsets.shape[0]:
        raise ValueError("character_positions must contain one entry per prompt")

    resolved = []
    for row_offsets, row_mask, boundary in zip(offsets, attention_mask, character_positions):
        attended = np.flatnonzero(row_mask.astype(bool))
        if not len(attended):
            raise ValueError("Cannot resolve a position for an empty tokenized prompt")
        if boundary is None:
            resolved.append(int(attended[-1]))
            continue

        boundary = int(boundary)
        starts = row_offsets[:, 0]
        ends = row_offsets[:, 1]
        fully_before = attended[(ends[attended] > 0) & (ends[attended] <= boundary)]
        if len(fully_before):
            resolved.append(int(fully_before[-1]))
            continue
        overlaps = attended[(starts[attended] < boundary) & (ends[attended] > 0)]
        if not len(overlaps):
            raise ValueError(f"No token overlaps character boundary {boundary}")
        resolved.append(int(overlaps[-1]))
    return np.asarray(resolved, dtype=np.int64)


def token_ranges_from_character_offsets(
    offsets: np.ndarray,
    attention_mask: np.ndarray,
    character_ranges: list[tuple[int | None, int | None]],
    *,
    exclude_final_token: bool = False,
) -> list[np.ndarray]:
    """Resolve inclusive character ranges to attended token-index ranges.

    ``None`` has the same meaning as in :func:`token_indices_from_character_offsets`:
    it resolves to the final attended token.  The helper is deliberately kept
    separate from point-position resolution because persistence interventions
    operate over a variable number of prompt tokens.
    """
    if len(character_ranges) != len(offsets):
        raise ValueError("character_ranges must contain one range per prompt")
    starts = token_indices_from_character_offsets(
        offsets,
        attention_mask,
        [start for start, _ in character_ranges],
    )
    ends = token_indices_from_character_offsets(
        offsets,
        attention_mask,
        [end for _, end in character_ranges],
    )
    resolved: list[np.ndarray] = []
    for start, end in zip(starts, ends):
        low, high = sorted((int(start), int(end)))
        if exclude_final_token:
            high -= 1
        if high < low:
            raise ValueError("Injection range is empty after excluding the final token")
        resolved.append(np.arange(low, high + 1, dtype=np.int64))
    return resolved


def collect_choice_probs_and_position_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    position_character_offsets: dict[str, list[int | None]],
    layer_indices: list[int],
    batch_size: int = 4,
    max_length: int = 1024,
    choice_letters: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, dict[int, np.ndarray]]]:
    """Collect hidden states at named positions from the same full prompt."""
    import torch

    if not position_character_offsets:
        raise ValueError("At least one extraction position is required")
    for name, values in position_character_offsets.items():
        if len(values) != len(prompts):
            raise ValueError(f"Position {name!r} has {len(values)} offsets for {len(prompts)} prompts")

    choice_letters = choice_letters or ["A", "B"]
    layers = decoder_layers(model)
    resolved_layers = [resolve_layer_index(idx, len(layers)) for idx in layer_indices]
    hidden_state_indices = [idx + 1 for idx in resolved_layers]
    choice_ids = choice_token_ids(tokenizer, choice_letters)
    probs_batches = []
    activation_batches = {
        name: {idx: [] for idx in resolved_layers} for name in position_character_offsets
    }
    device = _input_device(model)

    for start in range(0, len(prompts), batch_size):
        stop = min(start + batch_size, len(prompts))
        batch = prompts[start:stop]
        try:
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, ValueError) as exc:
            raise ValueError(
                "Position extraction requires a fast tokenizer with offset mappings"
            ) from exc
        offsets = encoded.pop("offset_mapping")
        offset_array = offsets.detach().cpu().numpy() if hasattr(offsets, "detach") else np.asarray(offsets)
        mask_array = encoded["attention_mask"].detach().cpu().numpy()
        token_positions = {
            name: token_indices_from_character_offsets(
                offset_array,
                mask_array,
                values[start:stop],
            )
            for name, values in position_character_offsets.items()
        }
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        logits = _gather_last_token(outputs.logits, encoded["attention_mask"])
        choice_logits = logits[:, choice_ids]
        probs_batches.append(torch.softmax(choice_logits, dim=-1).detach().float().cpu().numpy())

        for resolved, hidden_idx in zip(resolved_layers, hidden_state_indices):
            hidden = outputs.hidden_states[hidden_idx]
            for name, positions in token_positions.items():
                position_tensor = torch.as_tensor(positions, device=hidden.device)
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                selected = hidden[batch_indices, position_tensor, :]
                activation_batches[name][resolved].append(
                    selected.detach().float().cpu().numpy()
                )
        del encoded, outputs, logits, choice_logits

    probs_all = np.concatenate(probs_batches, axis=0) if probs_batches else np.zeros((0, len(choice_ids)))
    activations = {
        name: {
            idx: np.concatenate(chunks, axis=0).astype(np.float32)
            for idx, chunks in layer_batches.items()
        }
        for name, layer_batches in activation_batches.items()
    }
    return probs_all, activations


def collect_choice_probs_and_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    layer_indices: list[int],
    batch_size: int = 4,
    max_length: int = 1024,
    choice_letters: list[str] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    import torch

    choice_letters = choice_letters or ["A", "B"]
    layers = decoder_layers(model)
    resolved_layers = [resolve_layer_index(idx, len(layers)) for idx in layer_indices]
    hidden_state_indices = [idx + 1 for idx in resolved_layers]
    choice_ids = choice_token_ids(tokenizer, choice_letters)

    probs_batches = []
    activation_batches: dict[int, list[np.ndarray]] = {idx: [] for idx in resolved_layers}
    device = _input_device(model)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        logits = _gather_last_token(outputs.logits, encoded["attention_mask"])
        choice_logits = logits[:, choice_ids]
        probs = torch.softmax(choice_logits, dim=-1).detach().float().cpu().numpy()
        probs_batches.append(probs)
        for resolved, hidden_idx in zip(resolved_layers, hidden_state_indices):
            hidden = _gather_last_token(outputs.hidden_states[hidden_idx], encoded["attention_mask"]).detach().float().cpu().numpy()
            activation_batches[resolved].append(hidden)
        del encoded, outputs, logits, choice_logits
    probs_all = np.concatenate(probs_batches, axis=0) if probs_batches else np.zeros((0, len(choice_ids)))
    activations = {idx: np.concatenate(chunks, axis=0).astype(np.float32) for idx, chunks in activation_batches.items()}
    return probs_all, activations


def collect_choice_probs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    batch_size: int = 4,
    max_length: int = 1024,
    choice_letters: list[str] | None = None,
) -> np.ndarray:
    import torch

    choice_letters = choice_letters or ["A", "B"]
    choice_ids = choice_token_ids(tokenizer, choice_letters)
    device = _input_device(model)
    batches = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = _gather_last_token(outputs.logits, encoded["attention_mask"])
        probs = torch.softmax(logits[:, choice_ids], dim=-1).detach().float().cpu().numpy()
        batches.append(probs)
        del encoded, outputs, logits
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, len(choice_ids)))


def completion_sequence_logprobs(
    model: Any,
    tokenizer: Any,
    prefixes: list[str],
    completions: list[list[str]],
    *,
    batch_size: int = 4,
    max_length: int = 1024,
    layer_index: int | None = None,
    direction: np.ndarray | None = None,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score candidate completion sequences, optionally after a prefix-position intervention.

    Unlike :func:`collect_choice_probs`, this routine never approximates a
    multi-token verbalizer by its final token.  It returns both summed and
    token-normalized log likelihoods, plus candidate token counts.  When a
    direction is supplied, it is written at the final token of the prefix,
    before the candidate completion begins.
    """
    import torch

    if len(prefixes) != len(completions):
        raise ValueError("prefixes and completions must have the same length")
    if not prefixes:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.int64),
        )
    n_candidates = len(completions[0])
    if n_candidates < 2 or any(len(values) != n_candidates for values in completions):
        raise ValueError("Every prefix must provide the same number of at least two completions")
    if (layer_index is None) != (direction is None):
        raise ValueError("layer_index and direction must either both be set or both be omitted")

    records: list[dict[str, Any]] = []
    for item_index, (prefix, values) in enumerate(zip(prefixes, completions)):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        if not prefix_ids:
            raise ValueError("A completion prefix must tokenize to at least one token")
        for candidate_index, completion in enumerate(values):
            full_text = prefix + completion
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)
            if len(full_ids) <= len(prefix_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    "Completion tokenization does not preserve the prefix. "
                    "End prefixes with an explicit delimiter and begin completions with a separator."
                )
            records.append(
                {
                    "item_index": item_index,
                    "candidate_index": candidate_index,
                    "text": full_text,
                    "prefix_length": len(prefix_ids),
                    "candidate_length": len(full_ids) - len(prefix_ids),
                    "candidate_token_id": int(full_ids[-1]),
                }
            )

    # Most letter, ordinal, and common-label interfaces are one-token. Score
    # those candidates from one prefix forward pass rather than expanding each
    # prompt into K almost-identical full continuations.
    if all(record["candidate_length"] == 1 for record in records):
        device = _input_device(model)
        handle = None
        current_positions = None
        if direction is not None:
            layers = decoder_layers(model)
            resolved = resolve_layer_index(int(layer_index), len(layers))
            layer = layers[resolved]
            delta = torch.tensor(np.asarray(direction, dtype=np.float32) * float(alpha), device=device)

            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                if current_positions is None:
                    return output
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                hidden[batch_indices, current_positions.to(hidden.device), :] = hidden[batch_indices, current_positions.to(hidden.device), :] + delta.to(device=hidden.device, dtype=hidden.dtype)
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            handle = layer.register_forward_hook(hook)
        sums = np.zeros((len(prefixes), n_candidates), dtype=np.float32)
        counts = np.ones((len(prefixes), n_candidates), dtype=np.int64)
        try:
            for start in range(0, len(prefixes), batch_size):
                stop = min(start + batch_size, len(prefixes))
                encoded = tokenizer(prefixes[start:stop], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                if direction is not None:
                    current_positions = _last_token_indices(encoded["attention_mask"])
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    logits = _gather_last_token(model(**encoded).logits, encoded["attention_mask"])
                    log_probs = torch.log_softmax(logits, dim=-1).detach().float().cpu().numpy()
                for item_index in range(start, stop):
                    for candidate_index in range(n_candidates):
                        token_id = records[item_index * n_candidates + candidate_index]["candidate_token_id"]
                        sums[item_index, candidate_index] = log_probs[item_index - start, token_id]
                current_positions = None
        finally:
            if handle is not None:
                handle.remove()
        return sums, sums.copy(), counts

    device = _input_device(model)
    handle = None
    current_positions = None
    if direction is not None:
        layers = decoder_layers(model)
        resolved = resolve_layer_index(int(layer_index), len(layers))
        layer = layers[resolved]
        delta = torch.tensor(np.asarray(direction, dtype=np.float32) * float(alpha), device=device)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if current_positions is None:
                return output
            hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            positions = current_positions.to(hidden.device)
            hidden[batch_indices, positions, :] = hidden[batch_indices, positions, :] + delta.to(
                device=hidden.device, dtype=hidden.dtype
            )
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = layer.register_forward_hook(hook)

    sum_scores = np.full((len(prefixes), n_candidates), np.nan, dtype=np.float32)
    mean_scores = np.full_like(sum_scores, np.nan)
    token_counts = np.zeros((len(prefixes), n_candidates), dtype=np.int64)
    try:
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            encoded = tokenizer(
                [record["text"] for record in batch_records],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            attention_mask = encoded["attention_mask"]
            active_lengths = attention_mask.sum(dim=1).detach().cpu().numpy().astype(int)
            sequence_length = int(attention_mask.shape[1])
            if direction is not None:
                offsets = np.asarray(
                    [
                        sequence_length - int(length)
                        if getattr(tokenizer, "padding_side", "right") == "left"
                        else 0
                        for length in active_lengths
                    ],
                    dtype=np.int64,
                )
                current_positions = torch.as_tensor(
                    offsets + np.asarray([record["prefix_length"] - 1 for record in batch_records]),
                    dtype=torch.long,
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)
            input_ids = encoded["input_ids"]
            for row_index, record in enumerate(batch_records):
                offset = sequence_length - active_lengths[row_index] if getattr(tokenizer, "padding_side", "right") == "left" else 0
                first = int(offset + record["prefix_length"])
                count = int(record["candidate_length"])
                if first + count > offset + active_lengths[row_index]:
                    raise ValueError("Completion was truncated; increase max_length")
                prediction_positions = torch.arange(first - 1, first + count - 1, device=device)
                target_ids = input_ids[row_index, first : first + count]
                score = log_probs[row_index, prediction_positions, target_ids].sum().item()
                item_index = int(record["item_index"])
                candidate_index = int(record["candidate_index"])
                sum_scores[item_index, candidate_index] = float(score)
                mean_scores[item_index, candidate_index] = float(score / count)
                token_counts[item_index, candidate_index] = count
            del encoded, outputs, log_probs
            current_positions = None
    finally:
        if handle is not None:
            handle.remove()
    return sum_scores, mean_scores, token_counts


def patched_choice_probs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    layer_index: int,
    direction: np.ndarray,
    alpha: float,
    batch_size: int = 4,
    max_length: int = 1024,
    choice_letters: list[str] | None = None,
    injection_character_offsets: list[int | None] | None = None,
) -> np.ndarray:
    import torch

    choice_letters = choice_letters or ["A", "B"]
    layers = decoder_layers(model)
    resolved = resolve_layer_index(layer_index, len(layers))
    layer = layers[resolved]
    choice_ids = choice_token_ids(tokenizer, choice_letters)
    device = _input_device(model)
    delta = torch.tensor(np.asarray(direction, dtype=np.float32) * float(alpha), device=device)
    current_attention_mask = None
    current_positions = None

    if injection_character_offsets is not None and len(injection_character_offsets) != len(prompts):
        raise ValueError("injection_character_offsets must contain one entry per prompt")

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        nonlocal current_attention_mask, current_positions
        if current_attention_mask is None:
            return output
        if isinstance(output, tuple):
            hidden = output[0].clone()
            positions = (
                current_positions.to(hidden.device)
                if current_positions is not None
                else _last_token_indices(current_attention_mask).to(hidden.device)
            )
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            hidden[batch_indices, positions, :] = hidden[batch_indices, positions, :] + delta.to(
                device=hidden.device, dtype=hidden.dtype
            )
            return (hidden, *output[1:])
        hidden = output.clone()
        positions = (
            current_positions.to(hidden.device)
            if current_positions is not None
            else _last_token_indices(current_attention_mask).to(hidden.device)
        )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        hidden[batch_indices, positions, :] = hidden[batch_indices, positions, :] + delta.to(
            device=hidden.device, dtype=hidden.dtype
        )
        return hidden

    handle = layer.register_forward_hook(hook)
    batches = []
    try:
        for start in range(0, len(prompts), batch_size):
            stop = min(start + batch_size, len(prompts))
            batch = prompts[start:stop]
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "max_length": max_length,
            }
            if injection_character_offsets is not None:
                tokenizer_kwargs["return_offsets_mapping"] = True
            try:
                encoded = tokenizer(batch, **tokenizer_kwargs)
            except (NotImplementedError, ValueError) as exc:
                if injection_character_offsets is not None:
                    raise ValueError(
                        "Position injection requires a fast tokenizer with offset mappings"
                    ) from exc
                raise
            if injection_character_offsets is not None:
                offsets = encoded.pop("offset_mapping")
                offset_array = offsets.detach().cpu().numpy() if hasattr(offsets, "detach") else np.asarray(offsets)
                mask_array = encoded["attention_mask"].detach().cpu().numpy()
                resolved = token_indices_from_character_offsets(
                    offset_array,
                    mask_array,
                    injection_character_offsets[start:stop],
                )
                current_positions = torch.as_tensor(resolved, dtype=torch.long)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            current_attention_mask = encoded["attention_mask"]
            with torch.no_grad():
                outputs = model(**encoded)
                logits = _gather_last_token(outputs.logits, encoded["attention_mask"])
            probs = torch.softmax(logits[:, choice_ids], dim=-1).detach().float().cpu().numpy()
            batches.append(probs)
            del encoded, outputs, logits
            current_attention_mask = None
            current_positions = None
    finally:
        handle.remove()
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, len(choice_ids)))


def patched_choice_probs_with_position_traces(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    layer_index: int,
    direction: np.ndarray,
    alpha: float,
    trace_character_offsets: list[int | None],
    trace_layer_indices: list[int],
    batch_size: int = 4,
    max_length: int = 1024,
    choice_letters: list[str] | None = None,
    injection_character_offsets: list[int | None] | None = None,
    injection_character_ranges: list[tuple[int | None, int | None]] | None = None,
    exclude_final_token_from_ranges: bool = False,
    mass_match_ranges: bool = False,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Apply a point or span intervention and trace hidden states at one position.

    This is used by the intervention-persistence audit.  A span intervention
    writes the direction at every token in the resolved range.  With
    ``mass_match_ranges=True``, its per-token perturbation is divided by the
    number of injected tokens, matching the total additive dose of a single
    point injection on each prompt.
    """
    import torch

    if (injection_character_offsets is None) == (injection_character_ranges is None):
        raise ValueError("Provide exactly one of injection_character_offsets or injection_character_ranges")
    if len(trace_character_offsets) != len(prompts):
        raise ValueError("trace_character_offsets must contain one entry per prompt")
    if injection_character_offsets is not None and len(injection_character_offsets) != len(prompts):
        raise ValueError("injection_character_offsets must contain one entry per prompt")
    if injection_character_ranges is not None and len(injection_character_ranges) != len(prompts):
        raise ValueError("injection_character_ranges must contain one range per prompt")

    choice_letters = choice_letters or ["A", "B"]
    layers = decoder_layers(model)
    resolved_layer = resolve_layer_index(layer_index, len(layers))
    resolved_trace_layers = [resolve_layer_index(idx, len(layers)) for idx in trace_layer_indices]
    if len(set(resolved_trace_layers)) != len(resolved_trace_layers):
        raise ValueError("trace_layer_indices must resolve to unique decoder layers")
    choice_ids = choice_token_ids(tokenizer, choice_letters)
    device = _input_device(model)
    base_delta = torch.tensor(np.asarray(direction, dtype=np.float32) * float(alpha), device=device)
    current_position_sets: list[np.ndarray] | None = None
    current_attention_mask = None
    current_token_counts: np.ndarray | None = None
    current_trace_positions = None
    trace_batches: dict[int, list[np.ndarray]] = {idx: [] for idx in resolved_trace_layers}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if current_position_sets is None or current_attention_mask is None or current_token_counts is None:
            return output
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        for row_index, positions in enumerate(current_position_sets):
            delta = base_delta.to(device=hidden.device, dtype=hidden.dtype)
            if mass_match_ranges and injection_character_ranges is not None:
                delta = delta / float(current_token_counts[row_index])
            token_positions = torch.as_tensor(positions, device=hidden.device)
            hidden[row_index, token_positions, :] = hidden[row_index, token_positions, :] + delta
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def make_trace_hook(trace_layer: int) -> Any:
        def trace_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if current_trace_positions is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            positions = current_trace_positions.to(hidden.device)
            batch_indices = torch.arange(len(positions), device=hidden.device)
            trace_batches[trace_layer].append(
                hidden[batch_indices, positions, :].detach().float().cpu().numpy()
            )
            return output

        return trace_hook

    handle = layers[resolved_layer].register_forward_hook(hook)
    trace_handles = [
        layers[trace_layer].register_forward_hook(make_trace_hook(trace_layer))
        for trace_layer in resolved_trace_layers
    ]
    prob_batches: list[np.ndarray] = []
    count_batches: list[np.ndarray] = []
    try:
        for start in range(0, len(prompts), batch_size):
            stop = min(start + batch_size, len(prompts))
            try:
                encoded = tokenizer(
                    prompts[start:stop],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_offsets_mapping=True,
                )
            except (NotImplementedError, ValueError) as exc:
                raise ValueError(
                    "Position traces require a fast tokenizer with offset mappings"
                ) from exc
            offsets = encoded.pop("offset_mapping")
            offset_array = offsets.detach().cpu().numpy() if hasattr(offsets, "detach") else np.asarray(offsets)
            mask_array = encoded["attention_mask"].detach().cpu().numpy()
            if injection_character_offsets is not None:
                points = token_indices_from_character_offsets(
                    offset_array,
                    mask_array,
                    injection_character_offsets[start:stop],
                )
                current_position_sets = [np.asarray([point], dtype=np.int64) for point in points]
            else:
                current_position_sets = token_ranges_from_character_offsets(
                    offset_array,
                    mask_array,
                    injection_character_ranges[start:stop],
                    exclude_final_token=exclude_final_token_from_ranges,
                )
            current_token_counts = np.asarray([len(values) for values in current_position_sets], dtype=np.int64)
            trace_positions = token_indices_from_character_offsets(
                offset_array,
                mask_array,
                trace_character_offsets[start:stop],
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            current_attention_mask = encoded["attention_mask"]
            current_trace_positions = torch.as_tensor(trace_positions, dtype=torch.long)
            with torch.no_grad():
                outputs = model(**encoded)
            logits = _gather_last_token(outputs.logits, encoded["attention_mask"])
            prob_batches.append(torch.softmax(logits[:, choice_ids], dim=-1).detach().float().cpu().numpy())
            count_batches.append(current_token_counts.copy())
            del encoded, outputs, logits
            current_position_sets = None
            current_attention_mask = None
            current_token_counts = None
            current_trace_positions = None
    finally:
        handle.remove()
        for trace_handle in trace_handles:
            trace_handle.remove()

    probs = np.concatenate(prob_batches, axis=0) if prob_batches else np.zeros((0, len(choice_ids)))
    traces = {
        layer: np.concatenate(parts, axis=0).astype(np.float32)
        for layer, parts in trace_batches.items()
    }
    counts = np.concatenate(count_batches, axis=0) if count_batches else np.zeros(0, dtype=np.int64)
    return probs, traces, counts


def build_binary_items_from_pairs(
    pairs: pd.DataFrame,
    *,
    choice_spec: ChoiceSpec | None = None,
    prompt_variant: str = "canonical",
) -> pd.DataFrame:
    choice_spec = choice_spec or ChoiceSpec()
    rows = []
    for _, pair in pairs.iterrows():
        negative_source = str(pair["negative_prompt"])
        positive_source = str(pair["positive_prompt"])
        rows.append(
            {
                "item_id": str(pair["negative_item_id"]),
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "split": pair["split"],
                "side": "negative",
                "target_choice": choice_spec.negative_choice_index,
                "target_polarity": "negative",
                "prompt_variant": prompt_variant,
                "scenario_char_end": len(negative_source),
                "prompt": choice_spec.format_prompt(negative_source),
            }
        )
        rows.append(
            {
                "item_id": str(pair["positive_item_id"]),
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "split": pair["split"],
                "side": "positive",
                "target_choice": choice_spec.positive_choice_index,
                "target_polarity": "positive",
                "prompt_variant": prompt_variant,
                "scenario_char_end": len(positive_source),
                "prompt": choice_spec.format_prompt(positive_source),
            }
        )
    items = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    items["item_index"] = np.arange(len(items), dtype=int)
    return items


def indexed_pairs(pairs: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    item_index = dict(zip(items["item_id"].astype(str), items["item_index"].astype(int)))
    out = pairs.copy()
    out["negative_item_index"] = out["negative_item_id"].astype(str).map(item_index)
    out["positive_item_index"] = out["positive_item_id"].astype(str).map(item_index)
    out = out[out["negative_item_index"].notna() & out["positive_item_index"].notna()].copy()
    out["negative_item_index"] = out["negative_item_index"].astype(int)
    out["positive_item_index"] = out["positive_item_index"].astype(int)
    return out


def baseline_rows(items: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = items.copy()
    out["prob_wrong"] = probs[:, 0]
    out["prob_right"] = probs[:, 1]
    out["prediction"] = probs.argmax(axis=1)
    out["correct"] = out["prediction"].eq(out["target_choice"].astype(int))
    return out


def evaluate_direction_bank(
    model: Any,
    tokenizer: Any,
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    base_probs: np.ndarray,
    directions: dict[tuple[str, str, int, float], np.ndarray],
    *,
    layer_index: int,
    alpha_grid: list[float],
    eval_split: str = "test",
    batch_size: int = 4,
    max_length: int = 1024,
    eval_side: str = "negative",
    target_polarity: str = "positive",
    prompt_variant: str = "canonical",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_items = items[(items["split"].astype(str).eq(eval_split)) & (items["side"].eq(eval_side))].copy()
    eval_items = eval_items.sort_values("item_index").reset_index(drop=True)
    if eval_items.empty:
        raise ValueError(f"No {eval_side}-side items found for eval_split={eval_split!r}")
    # Activation collection uses a different prompt mixture and may therefore
    # have slightly different low-precision numerics. Recompute the baseline on
    # the exact evaluation prompts and batches used by each contrast.
    _ = base_probs
    _ = pairs
    target_column = "target_choice"
    rows = []
    baseline_cache: dict[str, tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]] = {}
    for pair_type in sorted({str(key[0]) for key in directions}):
        subset = eval_items[eval_items["pair_type"].astype(str).eq(pair_type)].copy()
        if subset.empty:
            continue
        subset_prompts = subset["prompt"].tolist()
        subset_base = collect_choice_probs(
            model,
            tokenizer,
            subset_prompts,
            batch_size=batch_size,
            max_length=max_length,
            choice_letters=["A", "B"],
        )
        if target_polarity == "opposite":
            subset_target = 1 - subset[target_column].to_numpy(dtype=int)
        elif target_polarity in {"positive", "negative"}:
            target_rows = items[items["target_polarity"].eq(target_polarity)]
            if target_rows.empty:
                raise ValueError(f"Could not infer target choice for polarity {target_polarity!r}")
            subset_target = np.full(len(subset), int(target_rows["target_choice"].iloc[0]), dtype=int)
        else:
            subset_target = subset[target_column].to_numpy(dtype=int)
        baseline_cache[pair_type] = (
            subset,
            subset_prompts,
            subset_base,
            subset_base.argmax(axis=1),
            subset_target,
        )

    for (pair_type, mode, subspace_dim, mix_lambda), direction in sorted(directions.items()):
        cached = baseline_cache.get(str(pair_type))
        if cached is None:
            continue
        subset, subset_prompts, subset_base, subset_base_pred, subset_target = cached
        for alpha in alpha_grid:
            if mode == "zero_direction_control":
                patched = subset_base.copy()
            else:
                patched = patched_choice_probs(
                    model,
                    tokenizer,
                    subset_prompts,
                    layer_index=layer_index,
                    direction=direction,
                    alpha=float(alpha),
                    batch_size=batch_size,
                    max_length=max_length,
                    choice_letters=["A", "B"],
                )
            pred = patched.argmax(axis=1)
            js = [js_divergence(subset_base[i], patched[i]) for i in range(len(patched))]
            for local_i, (_, item) in enumerate(subset.iterrows()):
                rows.append(
                    {
                        "dataset": item.get("dataset", ""),
                        "pair_type": pair_type,
                        "pair_id": item["pair_id"],
                        "item_id": item["item_id"],
                        "split": eval_split,
                        "prompt_variant": prompt_variant,
                        "eval_side": eval_side,
                        "target_polarity": target_polarity,
                        "mode": mode,
                        "layer_index": int(layer_index),
                        "alpha": float(alpha),
                        "subspace_dim": int(subspace_dim),
                        "mix_lambda": float(mix_lambda),
                        "base_target_prob": float(subset_base[local_i, subset_target[local_i]]),
                        "patched_target_prob": float(patched[local_i, subset_target[local_i]]),
                        "delta_target_prob": float(patched[local_i, subset_target[local_i]] - subset_base[local_i, subset_target[local_i]]),
                        "base_positive_prob": float(subset_base[local_i, 1]),
                        "patched_positive_prob": float(patched[local_i, 1]),
                        "delta_positive_prob": float(patched[local_i, 1] - subset_base[local_i, 1]),
                        "base_intended_correct": bool(subset_base_pred[local_i] == subset_target[local_i]),
                        "patched_intended_correct": bool(pred[local_i] == subset_target[local_i]),
                        "prediction_changed": bool(pred[local_i] != subset_base_pred[local_i]),
                        "js_shift": float(js[local_i]),
                    }
                )
    eval_rows = pd.DataFrame(rows)
    summary = summarize_eval_rows(eval_rows)
    return eval_rows, summary


def summarize_eval_rows(eval_rows: pd.DataFrame) -> pd.DataFrame:
    if eval_rows.empty:
        return pd.DataFrame()
    group_cols = [
        "dataset",
        "pair_type",
        "prompt_variant",
        "eval_side",
        "target_polarity",
        "mode",
        "layer_index",
        "alpha",
        "subspace_dim",
        "mix_lambda",
    ]
    summary = (
        eval_rows.groupby(group_cols, as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            base_intended_acc=("base_intended_correct", "mean"),
            patched_intended_acc=("patched_intended_correct", "mean"),
            mean_delta_target_prob=("delta_target_prob", "mean"),
            mean_delta_positive_prob=("delta_positive_prob", "mean"),
            mean_js_shift=("js_shift", "mean"),
            prediction_changed_rate=("prediction_changed", "mean"),
        )
        .sort_values(group_cols)
    )
    summary["delta_intended_acc"] = summary["patched_intended_acc"] - summary["base_intended_acc"]
    return summary


def run_binary_direction_validation(
    model: Any,
    tokenizer: Any,
    pairs: pd.DataFrame,
    *,
    layer_index: int,
    alpha_grid: list[float],
    subspace_dims: list[int],
    mixing_grid: list[float],
    train_split: str = "train",
    eval_split: str = "test",
    batch_size: int = 4,
    max_length: int = 1024,
    seed: int = 13,
    prompt_variants: list[str] | None = None,
    eval_directions: list[str] | None = None,
    include_decomposition_modes: bool = True,
    include_loco: bool = False,
    loco_subspace_dim: int = 1,
    extraction_position: str = "pre_answer",
    negative_label: str = "morally wrong",
    positive_label: str = "morally right",
) -> dict[str, pd.DataFrame]:
    resolved_layer = resolve_layer_index(layer_index, len(decoder_layers(model)))
    if extraction_position not in {"scenario_end", "pre_answer"}:
        raise ValueError("extraction_position must be scenario_end or pre_answer")
    prompt_variants = prompt_variants or ["canonical"]
    eval_directions = eval_directions or ["negative_to_positive"]
    all_tables = []
    all_pairs = []
    all_baseline = []
    all_raw_inventory = []
    all_direction_inventory = []
    all_eval_rows = []
    all_eval_summary = []
    for prompt_variant in prompt_variants:
        if prompt_variant == "canonical":
            choice_spec = ChoiceSpec(
                negative_label=negative_label,
                positive_label=positive_label,
                negative_letter="A",
                positive_letter="B",
            )
        elif prompt_variant == "label_order_flip":
            choice_spec = ChoiceSpec(
                negative_label=negative_label,
                positive_label=positive_label,
                negative_letter="B",
                positive_letter="A",
            )
        else:
            raise ValueError(f"Unknown prompt variant: {prompt_variant}")

        items = build_binary_items_from_pairs(pairs, choice_spec=choice_spec, prompt_variant=prompt_variant)
        items["dataset"] = pairs["dataset"].iloc[0] if "dataset" in pairs.columns and not pairs.empty else ""
        all_tables.append(items)
        pairs_indexed = indexed_pairs(pairs, items)
        pairs_indexed["prompt_variant"] = prompt_variant
        all_pairs.append(pairs_indexed)
        offsets = (
            items["scenario_char_end"].astype(int).tolist()
            if extraction_position == "scenario_end"
            else [None] * len(items)
        )
        probs, position_activations = collect_choice_probs_and_position_activations(
            model,
            tokenizer,
            items["prompt"].tolist(),
            position_character_offsets={extraction_position: offsets},
            layer_indices=[resolved_layer],
            batch_size=batch_size,
            max_length=max_length,
            choice_letters=["A", "B"],
        )
        activations = position_activations[extraction_position]
        all_baseline.append(baseline_rows(items, probs))
        raw_directions, raw_inventory = contrast_directions(
            pairs_indexed,
            activations[resolved_layer],
            negative_column="negative_item_index",
            positive_column="positive_item_index",
            split_column="split",
            train_split=train_split,
        )
        raw_inventory["prompt_variant"] = prompt_variant
        raw_inventory["extraction_position"] = extraction_position
        all_raw_inventory.append(raw_inventory)
        bank = decompose_directions(
            raw_directions,
            subspace_dims=subspace_dims if include_decomposition_modes else [0],
            mixing_grid=mixing_grid if include_decomposition_modes else [1.0],
            seed=seed,
            include_loco=bool(include_decomposition_modes and include_loco),
            loco_subspace_dim=loco_subspace_dim,
        )
        if not include_decomposition_modes:
            keep_modes = {"raw_caa_direction", "wrong_direction_control", "random_direction_control"}
            bank.vectors = {key: value for key, value in bank.vectors.items() if key[1] in keep_modes}
            bank.inventory = bank.inventory[bank.inventory["mode"].isin(keep_modes)].copy()
        hidden_dim = next(iter(raw_directions.values())).shape[0] if raw_directions else 0
        zero_directions = {}
        for pair_type in raw_directions:
            zero_directions[(pair_type, "zero_direction_control", 0, 1.0)] = np.zeros(hidden_dim, dtype=np.float32)
        directions = {**bank.vectors, **zero_directions}
        direction_inventory = bank.inventory.copy()
        if zero_directions:
            zero_inventory = pd.DataFrame(
                [
                    {
                        "pair_type": key[0],
                        "mode": key[1],
                        "subspace_dim": key[2],
                        "mix_lambda": key[3],
                        "direction_l2": 0.0,
                        "shared_component_l2": 0.0,
                        "residual_component_l2": 0.0,
                        "shared_explained_variance": np.nan,
                    }
                    for key in zero_directions
                ]
            )
            direction_inventory = pd.concat([direction_inventory, zero_inventory], ignore_index=True, sort=False)
        direction_inventory["prompt_variant"] = prompt_variant
        direction_inventory["extraction_position"] = extraction_position
        all_direction_inventory.append(direction_inventory)

        for eval_direction in eval_directions:
            if eval_direction == "negative_to_positive":
                eval_side = "negative"
                target_polarity = "positive"
                eval_directions_for_run = directions
            elif eval_direction == "positive_to_negative":
                eval_side = "positive"
                target_polarity = "negative"
                eval_directions_for_run = {
                    (pair_type, mode, subspace_dim, mix_lambda): -vector
                    if mode not in {"random_direction_control", "zero_direction_control"}
                    else vector
                    for (pair_type, mode, subspace_dim, mix_lambda), vector in directions.items()
                }
            else:
                raise ValueError(f"Unknown eval direction: {eval_direction}")
            eval_rows, eval_summary = evaluate_direction_bank(
                model,
                tokenizer,
                items,
                pairs_indexed,
                probs,
                eval_directions_for_run,
                layer_index=resolved_layer,
                alpha_grid=alpha_grid,
                eval_split=eval_split,
                batch_size=batch_size,
                max_length=max_length,
                eval_side=eval_side,
                target_polarity=target_polarity,
                prompt_variant=prompt_variant,
            )
            eval_rows["eval_direction"] = eval_direction
            eval_summary["eval_direction"] = eval_direction
            eval_rows["extraction_position"] = extraction_position
            eval_summary["extraction_position"] = extraction_position
            all_eval_rows.append(eval_rows)
            all_eval_summary.append(eval_summary)

    items_all = pd.concat(all_tables, ignore_index=True, sort=False) if all_tables else pd.DataFrame()
    pairs_all = pd.concat(all_pairs, ignore_index=True, sort=False) if all_pairs else pd.DataFrame()
    baseline_all = pd.concat(all_baseline, ignore_index=True, sort=False) if all_baseline else pd.DataFrame()
    raw_inventory_all = pd.concat(all_raw_inventory, ignore_index=True, sort=False) if all_raw_inventory else pd.DataFrame()
    direction_inventory_all = pd.concat(all_direction_inventory, ignore_index=True, sort=False) if all_direction_inventory else pd.DataFrame()
    eval_rows_all = pd.concat(all_eval_rows, ignore_index=True, sort=False) if all_eval_rows else pd.DataFrame()
    eval_summary_all = pd.concat(all_eval_summary, ignore_index=True, sort=False) if all_eval_summary else pd.DataFrame()
    return {
        "items": items_all,
        "pairs": pairs_all,
        "baseline_rows": baseline_all,
        "raw_direction_inventory": raw_inventory_all,
        "decomposed_direction_inventory": direction_inventory_all,
        "eval_rows": eval_rows_all,
        "eval_summary": eval_summary_all,
    }


def cleanup_model() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def save_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        saved[name] = path
    return saved
