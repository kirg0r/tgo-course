"""Train a word bigram model and a character-based BPE tokenizer."""

import argparse
from collections import Counter, defaultdict
from heapq import heapify, heappop, heappush
import json
import math
from pathlib import Path
import random
import re
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)
WORD = re.compile(r"[а-я]+(?:-[а-я]+)*")
END_WORD = "</w>"
UNKNOWN = "<unk>"
SEED = 42


def extract_words(text):
    return WORD.findall(text.lower().replace("ё", "е"))


def load_words(path=CORPUS):
    if not path.is_file():
        with urlopen(CORPUS_URL, timeout=30) as response:
            data = response.read()
        # Publish the cached download only after it completes.
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(data)
        Path(temporary.name).replace(path)
    return extract_words(path.read_text(encoding="utf-8-sig"))


def split_words(words):
    boundary = len(words) * 9 // 10
    return words[:boundary], words[boundary:]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class BigramModel:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, words):
        self.unigrams = Counter(words)
        self.vocabulary = tuple(sorted(self.unigrams)) + (UNKNOWN,)
        self.followers = defaultdict(Counter)
        for previous, following in zip(words, words[1:]):
            self.followers[previous][following] += 1
        # Count contexts that have a following word so each row sums to one.
        self.context_counts = {
            previous: sum(counts.values()) for previous, counts in self.followers.items()
        }
        return self

    def _token(self, word):
        return word if word in self.unigrams else UNKNOWN

    def prob(self, previous, following):
        previous, following = self._token(previous), self._token(following)
        count = self.followers.get(previous, {}).get(following, 0)
        total = self.context_counts.get(previous, 0)
        return (count + self.alpha) / (total + self.alpha * len(self.vocabulary))

    def logprob(self, words):
        return math.fsum(
            math.log(self.prob(previous, following))
            for previous, following in zip(words, words[1:])
        )

    def generate(self, start, length, temperature, rng):
        words = [start]
        for _ in range(length - 1):
            counts = self.followers.get(self._token(words[-1]), {})
            # The denominator is constant for a row and cancels on renormalization.
            weights = [
                (counts.get(word, 0) + self.alpha) ** (1 / temperature)
                for word in self.vocabulary
            ]
            words.append(rng.choices(self.vocabulary, weights=weights)[0])
        return words


def merge_pair(symbols, pair):
    merged = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


class BPE:
    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def fit(cls, word_counts, num_merges, min_frequency=2):
        spellings = {
            word: tuple(word) + (END_WORD,)
            for word, count in word_counts.items()
            if count >= min_frequency
        }
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for word, symbols in spellings.items():
            for pair, occurrences in Counter(zip(symbols, symbols[1:])).items():
                pair_counts[pair] += occurrences * word_counts[word]
                pair_words[pair].add(word)
        queue = [(-count, pair) for pair, count in pair_counts.items()]
        heapify(queue)
        merges = []

        while queue and len(merges) < num_merges:
            negative_count, pair = heappop(queue)
            if -negative_count != pair_counts[pair]:
                continue
            merges.append(pair)
            for word in sorted(pair_words[pair]):
                old = spellings[word]
                new = merge_pair(old, pair)
                spellings[word] = new
                old_pairs = Counter(zip(old, old[1:]))
                new_pairs = Counter(zip(new, new[1:]))
                for changed in old_pairs.keys() | new_pairs.keys():
                    difference = new_pairs[changed] - old_pairs[changed]
                    if not difference:
                        continue
                    pair_counts[changed] += difference * word_counts[word]
                    if new_pairs[changed]:
                        pair_words[changed].add(word)
                    else:
                        pair_words[changed].discard(word)
                    if pair_counts[changed]:
                        heappush(queue, (-pair_counts[changed], changed))
        if len(merges) != num_merges:
            raise ValueError(f"Only {len(merges)} BPE merges available, expected {num_merges}")
        return cls(merges)

    def encode_word(self, word):
        symbols = tuple(word) + (END_WORD,)
        while True:
            candidates = (
                pair for pair in zip(symbols, symbols[1:]) if pair in self.ranks
            )
            best = min(candidates, key=self.ranks.__getitem__, default=None)
            if best is None:
                return symbols
            symbols = merge_pair(symbols, best)


def bigram_results(words, train, validation):
    model = BigramModel().fit(train)
    word_counts = Counter(words)
    perplexity = {}
    for alpha in (1, 0.1, 0.01):
        model.alpha = alpha
        perplexity[str(alpha)] = math.exp(-model.logprob(validation) / (len(validation) - 1))

    sentence = validation[:12]
    shuffled = sentence.copy()
    rng = random.Random(SEED)
    rng.shuffle(shuffled)
    generation = [
        {
            "temperature": temperature,
            "start": start,
            "text": " ".join(model.generate(start, 30, temperature, rng)),
        }
        for temperature in (0.5, 1.0, 2.0)
        for start in ("князь", "она")
    ]
    return {
        "corpus": CORPUS_URL,
        "words": len(words),
        "train_words": len(train),
        "val_words": len(validation),
        "vocab_size": len(word_counts),
        "top20": word_counts.most_common(20),
        "perplexity": perplexity,
        "likelihood": {
            "sentence": " ".join(sentence),
            "shuffled": " ".join(shuffled),
            "alpha": model.alpha,
            "logprob_sentence": model.logprob(sentence),
            "logprob_shuffled": model.logprob(shuffled),
            "unseen_pairs_in_sentence": sum(
                following not in model.followers.get(previous, {})
                for previous, following in zip(sentence, sentence[1:])
            ),
        },
        "generation": generation,
    }


def bpe_results(words, num_merges):
    counts = Counter(words)
    tokenizer = BPE.fit(counts, num_merges)
    chars = sum(len(word) * count for word, count in counts.items())
    tokens = sum(len(tokenizer.encode_word(word)) * count for word, count in counts.items())
    vocabulary = {char for word in counts for char in word} | {END_WORD}
    vocabulary.update(left + right for left, right in tokenizer.merges)
    return {
        "num_merges": len(tokenizer.merges),
        "vocab_size": len(vocabulary),
        "merges": tokenizer.merges,
        "stats": {
            "chars": chars,
            "words": len(words),
            "tokens_bpe": tokens,
            "tokens_per_word": tokens / len(words),
            "compression_vs_chars": chars / tokens,
        },
        "examples": {
            word: tokenizer.encode_word(word)
            for word in ("наполеон", "сказала", "неудовольствие")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="train 200 instead of 1500 merges")
    args = parser.parse_args()

    words = load_words()
    train, validation = split_words(words)
    bigram = bigram_results(words, train, validation)
    bpe = bpe_results(words, 200 if args.fast else 1500)
    ARTIFACTS.mkdir(exist_ok=True)
    write_json(ARTIFACTS / "bigram.json", bigram)
    write_json(ARTIFACTS / "bpe.json", bpe)
    print(f"{len(words):,} words, {bpe['num_merges']} BPE merges → {ARTIFACTS}")


if __name__ == "__main__":
    main()
