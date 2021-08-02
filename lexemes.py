import re

from bs4 import NavigableString, Tag


def flatten_tag(tag):
    out = []
    if tag.name is None:  # we have a NavigableString instance
        print(f"current tag is:\n{tag}\nthis is a string\n\n")
        return [str(tag)]
    else:
        print(f"current tag is:\n{tag}\nthis is a tag. diving into descendants\n\n")
        tag_name = tag.name
        out.append(f"<{tag_name}>")
        print(f"added opening tag {tag_name} at index {len(out)}")

        i=0
        for i, subtag in enumerate(tag.children):
            out.extend(flatten_tag(subtag))

        print(f"added {i+1} descendants of {tag_name}")
        if not tag_name == "br":
            out.append(f"</{tag_name}>")
        print(f"added closing tag {tag_name} at index {len(out)}")

        return out


class InputStream:
    """
    Main input class
    (after https://lisperator.net/pltut/parser/input-stream)
    """
    def __init__(self, inp):
        self.pos, self.line, self.col = 0, 1, 0
        self.inp = inp
        self.flat_inp = flatten_tag(inp)

    def next(self):
        try:
            # this indexing is interpreted as key check in 'attrs'
            #   so KeyError is used below. TODO: May not be robust
            try:
                ch_or_tag = self.inp[self.pos]
                self.pos += 1
                if ch_or_tag == "\n":
                    self.line += 1
                    self.col = 0
                else:
                    self.col += 1
            except (IndexError) as e:
                # TODO: closing tag should be added here
                ch_or_tag = self.inp.next_elements[self.pos]
                self.pos += 1

        except (KeyError) as e:
            ch_or_tag = self.inp.next_elements[self.pos]
            self.pos += 1
            # # to add closing tags to parsing we save the next tag and then check later
            # if self.elements_after_cur_tag:
            #     self.elements_after_cur_tag[0] == ch_or_tag
        return str(ch_or_tag)

    def peek(self):
        try:
            # this indexing is interpreted as key check in 'attrs'
            #   so KeyError is used below. TODO: May not be robust
            ch_or_tag = self.inp[self.pos]
        except (KeyError) as e:
            ch_or_tag = self.inp.next_elements[self.pos]

        return str(ch_or_tag)

    def eof(self):
        # TODO: rework
        return self.peek().next_sibling is None

    def croak(self, msg):
        raise ValueError(msg + " (" + str(self.line) + ":" + str(self.col) + ")")


def _get_rus_alphabet(file="ru_alphabet.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        letters = frozenset(f.read().split())
    return letters

def _get_sakha_alphabet(file="sa_alphabet.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        letters = frozenset(f.read().split())
    return letters

def _get_sakhaonly_letters(file="sa_uniquealphabet.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        letters = frozenset(f.read().split())
    return letters

SAKHA_ALPHABET = _get_sakha_alphabet()
SAKHAONLY_LETTERS = _get_sakhaonly_letters()
RUS_ALPHABET = _get_rus_alphabet()


def is_sakha(ch):
    return ch in SAKHA_ALPHABET


def is_rus(ch):
    return ch in RUS_ALPHABET


def is_digit(ch: str, digits=frozenset('0123456789')):
    return ch in digits


def read_while(predicate):
    str = ""
    while (not input.eof() and predicate(input.peek())):
        str += input.next()
    return str


def transl_num(item: str, transl_num_regex = re.compile('^[1-9][0-9]* $')):
    if transl_num_regex.search(item):
        return True
    return False

# def



