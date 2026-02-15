"""OpenAI middleware example — auto-expand user messages.

Requires: pip install txtpand[openai]
Set OPENAI_API_KEY environment variable.
"""

import txtpand

# Uncomment when you have openai installed and API key set:
#
# import openai
#
# client = txtpand.wrap_openai(openai.OpenAI())
#
# # User messages are automatically expanded before sending
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "user", "content": "cn y hel me wri a py func"}
#     ],
# )
# # "cn y hel me wri a py func" is expanded to
# # "can you help me write a python function" before sending
# print(response.choices[0].message.content)

# Demo without API call:
print("OpenAI middleware wraps client to auto-expand user messages.")
print(f"  'cn y hel me' → '{txtpand.expand('cn y hel me')}'")
