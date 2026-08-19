"""AraSeg model: shared encoder + per-subtask binary classification heads.

Head options (head_type):
  - "linear" : Dropout -> Linear(hidden, 2)  [default; identical to original]
  - "bilstm" : Dropout -> shared BiLSTM -> per-subtask Linear(2*lstm_hidden, 2)

Both heads emit per-token 2-class logits with the SAME shape, so the entire
downstream eval path (softmax -> per-word prob -> threshold tuning -> ensemble)
is unchanged regardless of head_type.
"""

from typing import Dict, List, Optional
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel


def _key(subtask: str) -> str:
    """Sanitize subtask name for use as a ModuleDict key (hyphens → underscores)."""
    return subtask.replace("-", "_")


_DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]


def _get_hidden_size(config) -> int:
    """Return the hidden/model dimension for configs that don't standardise on
    hidden_size: SSM hybrids use d_model; multimodal wrappers (Gemma4Unified)
    nest the text config under .text_config."""
    for attr in ("hidden_size", "d_model", "model_dim", "dim"):
        if hasattr(config, attr):
            return getattr(config, attr)
    if hasattr(config, "text_config"):
        return _get_hidden_size(config.text_config)
    raise AttributeError(
        f"Cannot infer hidden size from {type(config).__name__}. "
        f"Add the right attribute name to _get_hidden_size() in model.py."
    )


def _get_num_layers(config) -> int:
    """Number of transformer blocks, tolerant of non-standard config schemas."""
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    if hasattr(config, "text_config"):
        return _get_num_layers(config.text_config)
    raise AttributeError(
        f"Cannot infer layer count from {type(config).__name__}. "
        f"Add the right attribute name to _get_num_layers() in model.py."
    )


class AraSegModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        subtasks: List[str],
        dropout: float = 0.1,
        trust_remote_code: bool = False,
        head_type: str = "linear",
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        torch_dtype: str = "float32",
        bidirectional: bool = False,
        lora_r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        gradient_checkpointing: bool = False,
        n_pos_tags: int = 0,
        n_dep_rels: int = 0,
        syntax_dim: int = 32,
        n_aux_classes: int = 0,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # QLoRA (e67+): NF4-quantized frozen base so 14B+ backbones fit 16 GB
        # cards (Kaggle T4). Compute dtype follows torch_dtype (float16 on T4 —
        # no bf16 there; bfloat16 on A100). bitsandbytes only imports when used.
        quant_kwargs = {}
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("load_in_4bit requires a CUDA device (bitsandbytes)")
            from transformers import BitsAndBytesConfig
            quant_kwargs = dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=getattr(torch, torch_dtype),
                ),
                device_map={"": 0},
            )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=getattr(torch, torch_dtype),
            **quant_kwargs,
        )
        self.bidirectional = bidirectional
        # Params4bit tensors report dtype uint8, which would break the
        # finfo() in _bidirectional_bias — pin the bias dtype to the compute
        # dtype instead of sniffing parameters.
        self._bias_dtype = getattr(torch, torch_dtype) if load_in_4bit else None

        if load_in_4bit:
            # Upcasts norms/embeddings to fp32, freezes the base, and (when
            # asked) enables non-reentrant gradient checkpointing — replaces
            # the manual enable below for the quantized path.
            from peft import prepare_model_for_kbit_training
            self.encoder = prepare_model_for_kbit_training(
                self.encoder,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif gradient_checkpointing:
            # use_reentrant=False is required for frozen-input (LoRA) training
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if lora_r > 0:
            from peft import LoraConfig, TaskType, get_peft_model
            self.encoder = get_peft_model(self.encoder, LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=lora_target_modules or _DEFAULT_LORA_TARGETS,
            ))

        self.dropout = nn.Dropout(dropout)
        hidden = _get_hidden_size(config)
        self.head_type = head_type
        self.crfs = None

        # Optional frozen-syntax feature embeddings (Phase B). IDs: 0=PAD, 1=UNK, 2+=real.
        if n_pos_tags > 0:
            self.pos_emb = nn.Embedding(n_pos_tags + 2, syntax_dim, padding_idx=0)
            self.dep_emb = nn.Embedding(n_dep_rels + 2, syntax_dim, padding_idx=0)
            self.syntax_proj = nn.Linear(2 * syntax_dim, hidden, bias=False)
        else:
            self.pos_emb = None

        if head_type == "linear":
            self.lstm = None
            head_in = hidden
        elif head_type == "bilstm":
            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            head_in = 2 * lstm_hidden
        elif head_type == "crf":
            from torchcrf import CRF
            self.lstm = None
            head_in = hidden
            self.crfs = nn.ModuleDict({_key(st): CRF(2, batch_first=True) for st in subtasks})
        else:
            raise ValueError(f"Unknown head_type={head_type!r} (expected 'linear', 'bilstm', or 'crf')")

        self.heads = nn.ModuleDict({_key(st): nn.Linear(head_in, 2) for st in subtasks})
        # Track 2: shared auxiliary punctuation-restoration head (predicts which
        # punctuation class was deleted after each token). Shared across subtasks
        # since the punct signal is condition-agnostic.
        self.aux_head = nn.Linear(head_in, n_aux_classes) if n_aux_classes > 0 else None
        self._subtasks = subtasks
        # Some architectures (XLM-R, ModernBERT) don't use token_type_ids
        self._use_ttype = getattr(config, "type_vocab_size", 1) > 1

    def _bidirectional_bias(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """2D padding mask (bs, kv) -> additive 4D bias (bs, 1, q, kv) with NO
        causal triangle: 0 where visible, -inf where padded. Recent transformers
        pass an already-4D attention_mask through unmodified, which is what
        defeats the decoder's causal mask."""
        bs, kv = attention_mask.shape
        bias = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        return bias.expand(bs, 1, kv, kv)

    def _encode(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        kwargs: Dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.bidirectional:
            enc_dtype = self._bias_dtype or next(self.encoder.parameters()).dtype
            kwargs["attention_mask"] = self._bidirectional_bias(attention_mask, enc_dtype)
        if token_type_ids is not None and self._use_ttype:
            kwargs["token_type_ids"] = token_type_ids

        # .float(): bf16 trunks feed fp32 heads/loss
        seq_out = self.dropout(self.encoder(**kwargs).last_hidden_state.float())

        if self.lstm is not None:
            # Pack so padded positions can't leak into real tokens via the
            # backward LSTM direction. lengths come from the attention mask.
            lengths = attention_mask.sum(dim=1).to("cpu")
            packed = pack_padded_sequence(
                seq_out, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            seq_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_out.size(1)
            )
        return seq_out

    def _gather_words(self, emissions: torch.Tensor, word_idx: torch.Tensor,
                      labels: Optional[torch.Tensor] = None):
        """Pack first-subword emissions into a dense left-contiguous per-word tensor."""
        B, T, C = emissions.shape
        first = word_idx >= 0
        counts = first.sum(dim=1)
        Wmax = max(int(counts.max().item()), 1) if B else 1
        word_em = emissions.new_zeros(B, Wmax, C)
        word_mask = torch.zeros(B, Wmax, dtype=torch.bool, device=emissions.device)
        word_labels = None
        if labels is not None:
            word_labels = labels.new_full((B, Wmax), -100)
        for i in range(B):
            idx = first[i].nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            if n == 0:
                word_mask[i, 0] = True
                continue
            word_em[i, :n] = emissions[i, idx]
            word_mask[i, :n] = True
            if labels is not None:
                word_labels[i, :n] = labels[i, idx]
        return word_em, word_mask, word_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        subtask: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
        word_idx: Optional[torch.Tensor] = None,
        syntax_pos_ids: Optional[torch.Tensor] = None,
        syntax_dep_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        seq_out = self._encode(input_ids, attention_mask, token_type_ids)

        # Additive syntax signal: project (POS emb ‖ DEP emb) → hidden dim
        if self.pos_emb is not None and syntax_pos_ids is not None:
            syn = torch.cat([
                self.pos_emb(syntax_pos_ids),
                self.dep_emb(syntax_dep_ids),
            ], dim=-1)
            seq_out = seq_out + self.syntax_proj(syn).to(seq_out.dtype)

        if subtask is not None:
            emissions = self.heads[_key(subtask)](seq_out)
            if self.head_type == "crf" and labels is not None:
                if word_idx is None:
                    raise ValueError("CRF training requires word_idx to gather per-word emissions")
                word_em, word_mask, word_lab = self._gather_words(emissions, word_idx, labels)
                valid = word_mask & (word_lab != -100)
                tags = word_lab.clamp(min=0)
                return -self.crfs[_key(subtask)](word_em, tags, mask=valid, reduction="mean")
            if return_aux and self.aux_head is not None:
                return emissions, self.aux_head(seq_out)
            return emissions
        # Multi-task: return logits for all heads
        return {st: self.heads[_key(st)](seq_out) for st in self._subtasks}

    @torch.no_grad()
    def crf_decode_words(self, input_ids, attention_mask, token_type_ids, subtask, word_idx):
        """Viterbi-decode dense per-word CRF emissions."""
        seq_out = self._encode(input_ids, attention_mask, token_type_ids)
        emissions = self.heads[_key(subtask)](seq_out)
        word_em, word_mask, _ = self._gather_words(emissions, word_idx)
        return self.crfs[_key(subtask)].decode(word_em, mask=word_mask)


# ── Construction & checkpoint helpers (shared by train.py / ensemble.py) ────
def build_model(cfg, model_path: str) -> AraSegModel:
    """Single place that maps a Config onto AraSegModel kwargs, so train and
    ensemble can never drift apart on architecture flags."""
    return AraSegModel(
        model_path, cfg.subtasks, trust_remote_code=cfg.trust_remote_code,
        head_type=cfg.head_type, lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers,
        torch_dtype=cfg.torch_dtype, bidirectional=cfg.bidirectional,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=cfg.gradient_checkpointing,
        n_pos_tags=getattr(cfg, "n_pos_tags", 0),
        n_dep_rels=getattr(cfg, "n_dep_rels", 0),
        syntax_dim=getattr(cfg, "syntax_dim", 32),
        n_aux_classes=getattr(cfg, "n_aux_classes", 0),
        load_in_4bit=getattr(cfg, "load_in_4bit", False),
    )


def save_checkpoint(model: nn.Module, path) -> None:
    """Full state dict when everything is trainable (BERT path — byte-identical
    to the old behavior). With a frozen LoRA base, save only trainable params
    (adapters + heads): ~100 MB instead of 14–24 GB per checkpoint."""
    if all(p.requires_grad for p in model.parameters()):
        torch.save(model.state_dict(), path)
        return
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    torch.save({k: v for k, v in model.state_dict().items() if k in keep}, path)


def load_checkpoint(model: nn.Module, path, device) -> None:
    """Load either a full or a trainable-only checkpoint into a freshly built
    model. Fails loudly if the checkpoint doesn't match the architecture or
    doesn't cover the trainable params."""
    sd = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in {path}: {list(unexpected)[:5]} ...")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    not_loaded = trainable & set(missing)
    if not_loaded:
        raise RuntimeError(f"Trainable params missing from {path}: {sorted(not_loaded)[:5]} ...")


def load_encoder_weights(model: nn.Module, path, device) -> None:
    """Load only shared encoder weights from a checkpoint, ignoring task heads."""
    sd = torch.load(path, map_location=device)
    enc_sd = {k: v for k, v in sd.items()
              if not (k.startswith("heads.") or k.startswith("crfs.") or k.startswith("lstm."))}
    if not enc_sd:
        raise RuntimeError(f"No encoder weights found in {path} (all keys were heads?)")
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    loaded = [k for k in enc_sd if k not in set(unexpected)]
    if not loaded:
        raise RuntimeError(f"No encoder weights from {path} matched the model - arch mismatch?")
    print(f"[pretrain] loaded {len(loaded)} encoder tensors from {path} "
          f"(skipped {len(sd) - len(enc_sd)} head tensors)")
