import re


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
    while (!input.eof() and predicate(input.peek())):
        str += input.next()
    return str


def transl_num(item: str, transl_num_regex = re.compile('^[1-9][0-9]* $')):
    if transl_num_regex.search(item):
        return True
    return False

def



