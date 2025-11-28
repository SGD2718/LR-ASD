import math
points = [(0, 0), (1, 2), (21, 44), (19, 63), (31, 8), (5, 114)]  # or some really big list


import re

def estimate_syllables(word: str) -> int:
    """
    Estimate number of syllables in an English word.
    Strategy:
      1) try the CMU-based libraries (pronouncing / nltk.corpus.cmudict) if installed
      2) fallback to a heuristic: count vowel groups, adjust for silent 'e' and final 'le'
    Returns an integer >= 1 for non-empty alphabetic input, 0 for empty input.
    """
    if not word:
        return 0
    w = word.lower().strip()
    # keep only basic letters and apostrophe (handle contractions)
    w_clean = re.sub(r"[^a-z']", "", w)
    if not w_clean:
        return 0

    # 1) try pronouncing (wrapper around CMUdict)
    try:
        import pronouncing
        phones = pronouncing.phones_for_word(w_clean)
        if phones:
            return pronouncing.syllable_count(phones[0])
    except Exception:
        pass

    # 2) try nltk's cmudict if available
    try:
        from nltk.corpus import cmudict
        d = cmudict.dict()
        if w_clean in d:
            # each pronunciation is a list of ARPABET phonemes; vowels have stress digits
            counts = [sum(1 for p in pron if p[-1].isdigit()) for pron in d[w_clean]]
            if counts:
                return min(counts)
    except Exception:
        pass

    # 3) heuristic fallback
    word = w_clean
    # common tiny words
    if len(word) <= 3:
        return 1

    vowels = "aeiouy"
    # count contiguous vowel groups as a baseline
    groups = re.findall(r"[aeiouy]+", word)
    syll = len(groups)

    # subtract 1 for a trailing silent 'e' (but not 'le' cases)
    if word.endswith("e"):
        if not word.endswith("le") and syll > 1:
            syll -= 1

    # add 1 for words ending with consonant + 'le' (e.g., "table" -> 2)
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        syll += 1

    # common adjustments for some suffix patterns
    # subtract 1 for 'es' and 'ed' endings that form an extra written vowel but not a spoken one (e.g., 'walked')
    if word.endswith(("es", "ed")) and len(re.findall(r"[aeiouy]", word[:-2])) > 0:
        # avoid removing valid syllables like 'blessed' pronounced with two syllables
        if not re.search(r"(ted|ded|ses|ces|ges|ges)$", word):
            if syll > 1:
                syll -= 1

    # minimum 1
    syll = max(1, syll)
    return syll


print(estimate_syllables("neighborhood"))
print(estimate_syllables("Jonathan Virak"))
print(estimate_syllables("deque"))
print(estimate_syllables("beautiful"))
print(estimate_syllables("Audivize"))
