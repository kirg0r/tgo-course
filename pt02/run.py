"""Train a small causal Transformer on the BPE tokens from lab 1."""

import argparse
from collections import deque
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pt01.run import BPE, CORPUS, END_WORD, SEED, BigramModel, extract_words, load_words, split_words, write_json

ARTIFACTS = HERE / "artifacts"
BPE_PATH = HERE.parent / "pt01" / "artifacts" / "bpe.json"
PROMPT = "князь андрей"


@dataclass(frozen=True)
class Config:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    context: int = 128
    steps: int = 2000
    batch: int = 32
    lr: float = 1e-3
    seed: int = SEED


class Tokenizer:
    def __init__(self, bpe, words):
        self.pieces = {word: bpe.encode_word(word) for word in dict.fromkeys(words)}
        self.tokens = sorted({token for pieces in self.pieces.values() for token in pieces})
        self.ids = {token: index for index, token in enumerate(self.tokens)}
        self.bpe = bpe

    def encode(self, words):
        ids = []
        for word in words:
            pieces = self.pieces.get(word)
            if pieces is None:
                pieces = self.bpe.encode_word(word)
            ids.extend(self.ids[token] for token in pieces)
        return ids

    def decode(self, ids):
        return "".join(self.tokens[index] for index in ids).replace(END_WORD, " ").strip()


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.head_size = config.d_model // self.heads
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.register_buffer(
            "future",
            torch.ones(config.context, config.context, dtype=torch.bool).triu(diagonal=1),
            persistent=False,
        )

    def split_heads(self, values):
        batch, length, _ = values.shape
        return values.reshape(batch, length, self.heads, self.head_size).transpose(1, 2)

    def forward(self, values):
        batch, length, width = values.shape
        query = self.split_heads(self.query(values))
        key = self.split_heads(self.key(values))
        value = self.split_heads(self.value(values))
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.future[:length, :length], float("-inf"))
        attention = F.softmax(scores, dim=-1)
        combined = (attention @ value).transpose(1, 2).reshape(batch, length, width)
        return self.output(combined)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, values):
        values = values + self.attention(self.attention_norm(values))
        return values + self.feed_forward(self.feed_forward_norm(values))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config.context
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context, config.d_model)
        self.blocks = nn.Sequential(*(DecoderBlock(config) for _ in range(config.n_layers)))
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            nn.init.normal_(layer.weight, std=0.02)
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        values = self.token_embeddings(ids) + self.position_embeddings(positions)
        return self.output(self.output_norm(self.blocks(values)))


class Generator:
    def __init__(self, model, tokenizer, device, seed):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Keep sampling independent of the random training windows.
        self.rng = torch.Generator().manual_seed(seed)

    def choose(self, logits, temperature, top_p):
        if temperature == 0:
            return logits.argmax().item()
        probabilities = F.softmax(logits / temperature, dim=-1).cpu()
        if top_p is None:
            return torch.multinomial(probabilities, 1, generator=self.rng).item()

        ordered, ids = probabilities.sort(descending=True)
        # Keep the first token that makes the total reach top_p.
        before = ordered.cumsum(dim=-1) - ordered
        kept = ordered.masked_fill(before >= top_p, 0)
        index = torch.multinomial(kept, 1, generator=self.rng).item()
        return ids[index].item()

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_p=None):
        prompt_ids = self.tokenizer.encode(extract_words(prompt))
        if not prompt_ids:
            raise ValueError("Prompt must contain at least one Russian word")
        context = deque(prompt_ids, maxlen=self.model.context)
        new_ids = []
        for _ in range(max_new_tokens):
            ids = torch.tensor([list(context)], device=self.device)
            logits = self.model(ids)[0, -1]
            new_id = self.choose(logits, temperature, top_p)
            context.append(new_id)
            new_ids.append(new_id)
        return self.tokenizer.decode(prompt_ids + new_ids), new_ids


def windows(stream, count, context, rng):
    starts = torch.randint(len(stream) - context, (count,), generator=rng)
    positions = starts[:, None] + torch.arange(context + 1)
    batch = stream[positions.to(stream.device)]
    return batch[:, :-1], batch[:, 1:]


def loss_on_windows(model, inputs, targets):
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.inference_mode()
def measure_loss(model, batch):
    return loss_on_windows(model, *batch).item()


def train(model, config, train_ids, val_ids, generator, fast):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    rng = torch.Generator().manual_seed(config.seed)
    val_rng = torch.Generator().manual_seed(config.seed + 1)
    initial_batch = windows(train_ids, config.batch, config.context, rng)
    validation = windows(val_ids, 20, config.context, val_rng)
    interval = 25 if fast else 100
    sample_steps = {0, 50, 100, config.steps} if fast else {0, 100, 500, config.steps}

    start = perf_counter()
    model.eval()
    curve = [{
        "step": 0,
        "train_loss": measure_loss(model, initial_batch),
        "val_loss": measure_loss(model, validation),
    }]
    samples = [{"step": 0, "text": generator.generate(PROMPT, 60, 0.8)[0]}]
    running_loss = 0.0
    for step in range(1, config.steps + 1):
        model.train()
        inputs, targets = windows(train_ids, config.batch, config.context, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on_windows(model, inputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if step % interval == 0 or step == config.steps:
            model.eval()
            train_loss = running_loss / (step - curve[-1]["step"])
            val_loss = measure_loss(model, validation)
            curve.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
            running_loss = 0.0
            elapsed = (perf_counter() - start) / 60
            print(
                f"{step:>4}/{config.steps}  train={train_loss:.4f}  val={val_loss:.4f}"
                f"  {elapsed:.2f} min",
                flush=True,
            )
        if step in sample_steps:
            model.eval()
            samples.append({"step": step, "text": generator.generate(PROMPT, 60, 0.8)[0]})

    minutes = (perf_counter() - start) / 60
    return curve, {"prompt": PROMPT, "temperature": 0.8, "samples": samples}, minutes


def first_repeat(ids):
    seen = set()
    for index in range(len(ids) - 3):
        group = tuple(ids[index:index + 4])
        if group in seen:
            return index
        seen.add(group)
    return None


def generation_results(generator, fast):
    length = 40 if fast else 80
    modes = (("greedy", 0.0, None), ("t0.7", 0.7, None), ("t1.5", 1.5, None), ("top_p0.9", 1.0, 0.9))
    runs = []
    for name, temperature, top_p in modes:
        text, ids = generator.generate(PROMPT, length, temperature, top_p)
        run = {
            "name": name,
            "temperature": temperature,
            "top_p": top_p,
            "text": text,
            "ids": ids,
            "n_tokens": len(ids),
        }
        if temperature == 0:
            run["loop_start"] = first_repeat(ids)
        runs.append(run)
    return {"model": "tiny", "prompt": PROMPT, "max_new_tokens": length, "runs": runs}


@torch.inference_mode()
def score_tokens(model, prefix, targets, device):
    # Evaluate each target with a window of up to context preceding tokens.
    stream = torch.tensor(prefix[-model.context:] + targets, device=device)
    context = min(model.context, len(prefix))
    windows = stream.unfold(0, context + 1, 1)
    losses = []
    for batch in windows.split(32):
        logits = model(batch[:, :-1])[:, -1]
        losses.extend(F.cross_entropy(logits, batch[:, -1], reduction="none").tolist())
    return math.fsum(losses)


def evaluation_results(model, tokenizer, train_words, val_words, train_ids, device, fast):
    words = val_words[:300 if fast else 2000]
    targets = tokenizer.encode(words)
    prefix = train_ids[-model.context:].tolist()

    bigram = BigramModel(alpha=0.01).fit(train_words)
    bigram_nats = -bigram.logprob([train_words[-1], *words])
    tiny_nats = score_tokens(model, prefix, targets, device)

    def metrics(tokens, nats_total):
        nats = nats_total / len(words)
        return {
            "tokens": tokens,
            "nats_total": nats_total,
            "nats_per_word": nats,
            "ppl_per_word": math.exp(nats),
        }

    return {
        "words": len(words),
        "bigram": {**metrics(len(words), bigram_nats), "alpha": bigram.alpha},
        "tiny": metrics(len(targets), tiny_nats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="train for 200 steps")
    parser.add_argument("--bpe", type=Path, default=BPE_PATH, help="path to the lab 1 bpe.json")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    if not args.bpe.is_file():
        parser.error(f"BPE file not found at {args.bpe}. Run pt01/run.py first or pass --bpe.")
    bpe_data = json.loads(args.bpe.read_text(encoding="utf-8"))
    bpe = BPE(bpe_data["merges"])
    words = load_words(CORPUS if CORPUS.is_file() else HERE / "corpus.txt")
    train_words, val_words = split_words(words)
    tokenizer = Tokenizer(bpe, words)
    config = Config(
        vocab_size=len(tokenizer.tokens),
        steps=200 if args.fast else 2000,
        batch=16 if args.fast else 32,
    )
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    train_ids = torch.tensor(tokenizer.encode(train_words), device=device)
    val_ids = torch.tensor(tokenizer.encode(val_words), device=device)
    model = Transformer(config).to(device)
    generator = Generator(model, tokenizer, device, config.seed + 2)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"{device}, {n_params:,} parameters, {len(bpe.merges)} BPE merges", flush=True)

    curve, samples, minutes = train(model, config, train_ids, val_ids, generator, args.fast)
    generation = generation_results(generator, args.fast)
    evaluation = evaluation_results(model, tokenizer, train_words, val_words, train_ids, device, args.fast)

    ARTIFACTS.mkdir(exist_ok=True)
    write_json(ARTIFACTS / "tiny_config.json", {
        **asdict(config),
        "n_params": n_params,
        "train_tokens": train_ids.numel(),
        "val_tokens": val_ids.numel(),
        "bpe_merges": len(bpe.merges),
        "train_minutes": minutes,
        "device": str(device),
    })
    with (ARTIFACTS / "train_curve.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("step", "train_loss", "val_loss"))
        writer.writeheader()
        writer.writerows(curve)
    write_json(ARTIFACTS / "samples.json", samples)
    write_json(ARTIFACTS / "generation.json", generation)
    write_json(ARTIFACTS / "eval.json", evaluation)
    print(f"Results → {ARTIFACTS}", flush=True)


if __name__ == "__main__":
    main()
