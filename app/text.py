"""Shared lexical helpers.

Stopword handling is not cosmetic here. It affects two things that matter:

1. **FTS recall precision.** The lexical branch builds an OR-query from the
   user's words. Leaving "what / is / the / of" in means every document
   containing "the" is a candidate, so a completely out-of-scope question
   ("what is the capital of France") still returns five confident-looking
   support articles.

2. **Confidence calibration.** The BM25-only confidence proxy divides the top
   score by the query's term count. Counting stopwords inflates the divisor
   for verbose questions and makes a genuinely good answer look uncertain.

Both are fixed by scoring on content terms only.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
    a about after again against all am an and any are aren as at be because been before being
    below between both but by can cannot could couldn did didn do does doesn doing don down
    during each few for from further had hadn has hasn have haven having he her here hers
    herself him himself his how i if in into is isn it its itself just ll me more most my
    myself no nor not now of off on once only or other ought our ours ourselves out over own
    re s same shan she should shouldn so some such t than that the their theirs them themselves
    then there these they this those through to too under until up ve very was wasn we were
    weren what when where which while who whom why will with won would wouldn you your yours
    yourself yourselves
    """.split()
)

# Domain words that are technically common but carry real retrieval signal on
# this corpus; never strip these even though they look generic.
KEEP_ALWAYS = frozenset({"no", "not", "cannot", "can", "will", "days", "day"})

# Apostrophes are deliberately NOT part of a token. They are FTS5 string
# delimiters, so emitting `don't` into a MATCH expression raises a syntax
# error and silently returns zero rows — which looked, from the outside,
# exactly like "we have nothing about unrecognised charges".
_WORD = re.compile(r"[a-z0-9]+")


def content_terms(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords and single characters removed."""
    tokens = _WORD.findall(text.lower())
    return [t for t in tokens if len(t) > 1 and (t not in STOPWORDS or t in KEEP_ALWAYS)]
