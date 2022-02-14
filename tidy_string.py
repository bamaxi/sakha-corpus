import re

RU_LC = 'абвгдеёжзийклмнопрстуфхцчшщъьыэюя'
RU_UC = 'АБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'
SA_ONLY_LC = 'өүҥһҕ'
SA_ONLY_UC = 'ӨҮҤҺҔ'
LAT_LC = 'abcdefghijklmnopqrstuvwxyz'
LAT_UC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'
PUNC = r""",.:;!?'"(){}[]<>-=+_/\|%@#$&№*^"""  # what to do with `_`?
WHITESPACE = " \t\n\r"
# these are unicode characters potentially useful for tokenization
PUNC_RARE_USEFUL = [
    '−',      # minus sign (\u2212)
    '°'       # degree sign (\xb0)
]

# ₳฿₿￠₡¢₢₵₫€￡£₤₣ƒ₲₭Ł₥₦₽₱＄$₮ℳ₶₩￦¥￥₴₸¤₰៛₪₯₠₧﷼円元圓㍐원৳₹₨৲௹
# TODO: translation may be needed from unusual signs (wide dollar / pound) to normal
CURRENCIES = """\u20B3\u0E3F\u20BF\uFFE0\u20A1\u00A2\u20A2\u20B5\u20AB\u20AC\uFFE1
\u00A3\u20A4\u20A3\u0192\u20B2\u20AD\u0141\u20A5\u20A6\u20BD\u20B1\uFF04\u0024\u20AE
\u2133\u20B6\u20A9\uFFE6\u00A5\uFFE5\u20B4\u20B8\u00A4\u20B0\u17DB\u20AA\u20AF\u20A0
\u20A7\uFDFC\u5186\u5143\u5713\u3350\uC6D0\u09F3\u20B9\u20A8\u09F2"""

ALPHABET = list(
    RU_LC + RU_UC + SA_ONLY_LC + SA_ONLY_UC + LAT_LC + LAT_UC
    + DIGITS + PUNC + WHITESPACE
)

ALPHABET_EXT = ALPHABET + list(CURRENCIES)


# this can perhaps be multilingual
symbols = {
    '\xad': '',                     # soft hyphens
    '\u2012': '-', '\u2013': '-',   # figure dash, en dash
        # '\u2212': '-',              # minus sign
    '\u200b': '', '\u200c': '',     # zero-width space and non-joiner
        '\u200d': '',               # zero-width joiner
        # '\u2800': '',               # braille pattern blank (IG new line)
    '\u2014': ' -- ',               # em dash TODO: is it actually needed?
    '\xa0': ' ', '\u202f': ' ',     # non-breaking space (and narrow)
    '\u2018': '"', '\u2019': '"',   # single quotes, left and right `‘`, `’`
    '\xab': '"', '\xbb': '"',       # Double Angle Quotation Marks `«`, `»`
    '\u201c': '"', '\u201d': '"',   # Double Quotation Marks `“`, `”`
    # TODO: there must be more quotes
    '\u201e': '"',                  # double low-9 quote `„`
    # '\u2033': '"',                  # Double Prime `″`
    '\xd7': '*',                    # multiplication sign `×`
}

# TODO: potential conversion to make - number symbols to literal numbers
#  https://unicode-table.com/en/sets/numerals/
#  https://unicode-table.com/en/sets/superscript-and-subscript-numbers/
#  https://unicode-table.com/en/sets/roman-numerals/ (roman numbers?)

# TODO: что с эмодзи? потенциально их есть смысл оставить для продвинутого нлп
#  типа intent detection. Текстов якутских много, много интернетных, почему нет?
#  у них часто есть \ufe0f

letters = {
    '\u019f': '\u04e8', '\u0275': '\u04e9',     # latin o bar to cyrillic o-bar
    '\u0472': '\u04e8', '\u0473': '\u04e9',     # cyrillic fita to cyrillic o-bar
    '\u048b': 'й',                              # I with tail `ҋ` to common `й`
    '\u04A2': 'ҥ'
}

# TODO: many-to-X
# и + \u0306 (и + combining breve) -> й

many_to_one_letters = {
    'и\u0306': 'й'
}

class MappingWithDefault(dict):
    def __missing__(self, key):
        return None


def check_pairwise_prefixhood(l):
    prefixes = {}
    sorted_l = sorted(l)
    for i, possible_prefix in enumerate(sorted_l):
        # print(f"pref is {possible_prefix}")
        for j, word in enumerate(sorted_l[i+1:]):
            # print(f"word is {word}")
            if possible_prefix[0] != word[0]:
                # print(f"quitting word, checking next prefix")
                break
            if word != possible_prefix and word.startswith(possible_prefix):
                # print(f"found prefix")
                prefixes.setdefault(possible_prefix, []).append(word)
    return prefixes


def multiple_replace(replacement_dict, text):
    # Create a regular expression  from the dictionary keys
    # keys = sorted(dict.keys(), key=len, reverse=True) # TODO: reverse?
    keys = replacement_dict.keys()
    prefixes = check_pairwise_prefixhood(keys)
    if prefixes:
        print(f"dictionary contains prefixed keys:\n{prefixes}")
        keys = sorted(keys)
        print(f"default policy is alphabetic sorting:\n{keys}"
              f"\nthis gives priority to strict prefixes, but not necessarily to shortest match")

    print(keys)
    regex = re.compile(f"({'|'.join(map(re.escape, keys))})")

    # For each match, look-up corresponding value in dictionary
    # return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)
    return regex.sub(lambda mo: replacement_dict[mo.group(0)], text)


def make_translation(include_alphabet=False, include_letters=False,
                     extra_translations=None, map_other_to_none=False):
    translation = symbols
    if include_alphabet:
        translation.update({ch: ch for ch in ALPHABET})
    if include_letters:
        translation.update(letters)
    if extra_translations:
        translation.update(extra_translations)
    translation = str.maketrans(translation)
    if map_other_to_none:
        translation = MappingWithDefault(translation)

    return translation

# TODO: tokenization may require replacements like `(\w+)(PUNC)(\w+)` -> `\1 \2 \3`


def make_string_tidier(translation, add_space=False, enspace_puncs=[',']):
    if add_space:
        import re
        pat = re.compile(rf"(\w+)([{''.join(enspace_puncs)}])(\w+)")

        def tidy_string(text):
            text = pat.sub("\1\2 \3", text)
            return text.translate(translation)

    else:
        def tidy_string(text):
            return text.translate(translation)

    return tidy_string


class Tidier:
    def __init__(self, translation=None, add_space=False, space_after_puncs=(',')):
        if not translation:
            translation = make_translation()

    # def apply(self, text):



print(ALPHABET)
