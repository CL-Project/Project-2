"""Text normalization and Hindi-aware tokenization.

Spec (to implement and test first):
- normalize(text): Unicode NFC, unify nukta variants, strip zero-width chars (ZWJ/ZWNJ), collapse whitespace.
- tokenize(text, lang): split into word tokens WITHOUT breaking Devanagari words at vowel signs (matras).
  Do NOT use regex \\w for Hindi. Keep numerals and Latin tokens.
- sentence_split(text, lang): split on danda (\u0964), '?', '!', '.'.
"""
