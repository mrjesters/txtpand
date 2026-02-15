"""Spaceless mode demo — no word boundaries in input."""

import txtpand

# Spaceless expansion
inputs = [
    "helloworld",
    "canyouhelp",
    "thankyou",
    "goodmorning",
]

for text in inputs:
    result = txtpand.expand(text, spaceless=True)
    print(f"{text:20s} → {result}")

# Detailed spaceless report
print("\n--- Detailed Report ---")
report = txtpand.expand_detailed("helpmewiththis", spaceless=True)
print(f"Input:    {report.input}")
print(f"Segments: {report.segments}")
print(f"Expanded: {report.expanded}")
