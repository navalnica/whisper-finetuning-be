import re
import unicodedata
from typing import Iterable

import regex


class BelarusianTextNormalizer:
    """
    Based on transformers.models.whisper.english_normalizer.BasicTextNormalizer
    but with support not to remove certain characters.
    e.g. apostrophe (') - a symbol from Belarusian alphabet - was removed using BasicTextNormalizer.
    """

    def __init__(self, split_letters: bool = False):
        self.split_letters = split_letters
        self.allowed_symbols = ("'",)

    @staticmethod
    def clean(s: str, allowed_symbols: Iterable[str] = None):
        """
        Replace any other markers, symbols, punctuations with a space, keeping diacritics
        """
        if allowed_symbols is None:
            allowed_symbols = []
        res = "".join(
            " " if unicodedata.category(c)[0] in "MSP" and c not in allowed_symbols else c
            for c in unicodedata.normalize("NFKC", s)
        )
        return res

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s, allowed_symbols=self.allowed_symbols).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s
