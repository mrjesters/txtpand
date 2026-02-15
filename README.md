# txtpand

> *"Type less. Say more."*

Shorthand text expansion library for Python. Expands abbreviated text into full English, enabling faster interaction with AI and messaging.

```python
import txtpand

txtpand.expand("cn y hel me wo on a fe thin")
# → "can you help me work on a few thin"

txtpand.expand("cnyhelme", spaceless=True)
# → "can you help me"
```

## Features

- **Spaced mode**: `cn y hel me` → `can you help me`
- **Spaceless mode**: `cnyhelme` → `can you help me`
- **Zero dependencies** for core functionality
- **3-tier matching**: exact → prefix trie → fuzzy edit distance
- **Bigram context**: uses word co-occurrence to disambiguate
- **LLM fallback**: optional, batched disambiguation for ambiguous tokens
- **AI middleware**: transparent proxies for OpenAI and Anthropic SDKs
- **Custom dictionaries**: add domain-specific words and abbreviations
- **User learning**: tracks corrections to improve over time

## Installation

```bash
pip install txtpand
```

Optional extras:

```bash
pip install txtpand[fast]       # rapidfuzz for faster fuzzy matching
pip install txtpand[openai]     # OpenAI middleware support
pip install txtpand[anthropic]  # Anthropic middleware support
pip install txtpand[all]        # Everything
```

## Quick Start

### Basic Expansion

```python
import txtpand

# Simple expansion
result = txtpand.expand("cn y hel me wo on smth")
print(result)  # "can you help me work on something"

# Detailed report
report = txtpand.expand_detailed("cn y hel me")
print(report.expanded)     # "can you help me"
print(report.confidence)   # 0.94
print(report.elapsed_ms)   # 12.3
```

### Spaceless Mode

```python
txtpand.expand("helloworld", spaceless=True)
# → "hello world"

txtpand.expand("canyouhelp", spaceless=True)
# → "can you help"
```

### Custom Dictionary

```python
expander = txtpand.Expander()
expander.add_words({"kubernetes": 5.0, "nginx": 4.0})
expander.add_abbreviations({"k8s": "kubernetes", "tf": "terraform"})

expander.expand("deploy k8s")  # → "deploy kubernetes"
```

### AI Middleware

Transparent proxies that expand user messages before sending to the API:

```python
import openai
import txtpand

client = txtpand.wrap_openai(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "cn y hel me wri a func"}],
)
# Message silently expanded to "can you help me write a function"
```

```python
import anthropic
import txtpand

client = txtpand.wrap_anthropic(anthropic.Anthropic())
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    messages=[{"role": "user", "content": "cn y hel me wri a func"}],
)
```

### CLI

```bash
# Direct argument
python -m txtpand "cn y hel me"

# Pipe from stdin
echo "cn y hel me" | python -m txtpand

# Spaceless mode
python -m txtpand --spaceless "cnyhelme"

# Detailed output
python -m txtpand --detailed "cn y hel me"
```

## How It Works

### Pipeline

```
"cn y hel me wo on a fe thin"
  → Tokenizer     → ["cn", "y", "hel", "me", "wo", "on", "a", "fe", "thin"]
  → Pass-through  → known words ("me","on","a") skip matching
  → Matcher       → candidates per token (prefix trie + fuzzy)
  → Context       → re-rank using bigram co-occurrence
  → Scorer        → pick top if confident, flag ambiguous
  → LLM fallback  → resolve ambiguous tokens (optional)
  → "can you help me work on a few thin"
```

### Spaceless Mode

```
"cnyhelme"
  → Segmenter     → Viterbi DP finds word boundaries → ["cn", "y", "hel", "me"]
  → Standard pipeline → "can you help me"
```

### Matching Strategy

1. **Exact match** (O(1)) — token is already a word → pass through
2. **Prefix match** (O(L) via trie) — token starts a word → rank by frequency
3. **Fuzzy match** (edit distance) — handles typos/vowel-dropping

**Scoring formula:**
```
score = 0.35 × prefix + 0.25 × edit_similarity + 0.25 × frequency + 0.15 × length_penalty
```

Context resolver adds bigram bonus: `"wo" after "me" → "work" beats "would"`.

## Configuration

```python
from txtpand import Expander, TxtpandConfig

config = TxtpandConfig(
    ambiguity_margin=0.15,      # Top-2 candidate margin for LLM fallback
    min_confidence=0.20,        # Minimum score to accept a candidate
    passthrough_known_words=True,
    llm_enabled=False,          # Enable LLM fallback
    llm_timeout_seconds=2.0,
)

expander = Expander(config)
```

## Project Structure

```
src/txtpand/
├── __init__.py          # Public API
├── __main__.py          # CLI
├── config.py            # Configuration
├── types.py             # Data types
├── exceptions.py        # Error hierarchy
├── core/
│   ├── expander.py      # Main orchestrator
│   ├── tokenizer.py     # Input tokenization
│   ├── matcher.py       # 3-tier matching (trie + fuzzy)
│   ├── context.py       # Bigram context resolver
│   ├── scorer.py        # Candidate scoring
│   └── segmenter.py     # Spaceless word segmentation (Viterbi DP)
├── corpus/
│   ├── loader.py        # Corpus data loader
│   ├── words.json       # Word frequencies
│   └── bigrams.json     # Bigram frequencies
├── llm/
│   ├── fallback.py      # LLM disambiguation
│   ├── prompt.py        # Prompt templates
│   └── providers.py     # OpenAI/Anthropic adapters
├── middleware/
│   ├── openai.py        # OpenAI transparent proxy
│   └── anthropic.py     # Anthropic transparent proxy
└── learning/
    └── user_model.py    # User correction tracking
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=txtpand --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/txtpand/
```

## License

Apache 2.0
