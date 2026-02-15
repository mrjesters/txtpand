"""Basic usage of txtpand."""

import txtpand

# Simple expansion
result = txtpand.expand("cn y hel me wo on a fe thin")
print(f"Expanded: {result}")

# Detailed report
report = txtpand.expand_detailed("cn y hel me")
print(f"\nDetailed report:")
print(f"  Input:      {report.input}")
print(f"  Expanded:   {report.expanded}")
print(f"  Confidence: {report.confidence:.2f}")
print(f"  Elapsed:    {report.elapsed_ms:.1f}ms")
for token in report.tokens:
    if token.original != token.expanded:
        print(f"  {token.original!r} → {token.expanded!r} (confidence: {token.confidence:.2f})")
    else:
        print(f"  {token.original!r} (passthrough)")
