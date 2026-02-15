"""Custom dictionary example — add domain-specific words and abbreviations."""

import txtpand

# Create an expander with custom settings
expander = txtpand.Expander()

# Add domain-specific words
expander.add_words({
    "kubernetes": 5.0,
    "nginx": 4.0,
    "terraform": 4.0,
    "postgresql": 3.5,
    "elasticsearch": 3.0,
})

# Add custom abbreviations
expander.add_abbreviations({
    "k8s": "kubernetes",
    "tf": "terraform",
    "pg": "postgresql",
    "es": "elasticsearch",
    "ng": "nginx",
})

# Now these abbreviations work
tests = [
    "deploy k8s to prod",
    "update tf config",
    "restart ng server",
    "check pg status",
    "fix es index",
]

for text in tests:
    result = expander.expand(text)
    print(f"{text:25s} → {result}")
