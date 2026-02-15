"""Anthropic middleware example — auto-expand user messages.

Requires: pip install txtpand[anthropic]
Set ANTHROPIC_API_KEY environment variable.
"""

import txtpand

# Uncomment when you have anthropic installed and API key set:
#
# import anthropic
#
# client = txtpand.wrap_anthropic(anthropic.Anthropic())
#
# response = client.messages.create(
#     model="claude-haiku-4-5-20251001",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "cn y hel me wri a py func"}
#     ],
# )
# print(response.content[0].text)

# Demo without API call:
print("Anthropic middleware wraps client to auto-expand user messages.")
print(f"  'cn y hel me' → '{txtpand.expand('cn y hel me')}'")
