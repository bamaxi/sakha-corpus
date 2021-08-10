import re
import logging
import logging.config  # TODO: will have to be defined in main module in the future
from functools import partial
from typing import Union, Dict, Callable

from bs4 import NavigableString, Tag

from utils import SAKHA_ALPHABET, SAKHAONLY_LETTERS, RUS_ALPHABET

logging.config.fileConfig('logging.conf')
log = logging.getLogger(__name__)


def flatten_soup_tag(tag):
    out = []
    if tag.name is None:  # we have a NavigableString instance
        # print(f"current tag is:\n{tag}\nthis is a string\n\n")
        return [{"basic_type": "string", "value": str(tag)}]
    else:
        # print(f"current tag is:\n{tag}\nthis is a tag. diving into descendants\n\n")
        tag_name = tag.name
        out.append({"basic_type": "tag", "kind": "<", "value": f"{str(tag_name)}"})
        # print(f"added opening tag {tag_name} at index {len(out)}")
        i = 0
        for i, subtag in enumerate(tag.children):
            out.extend(flatten_soup_tag(subtag))
        # print(f"added {i+1} descendants of {tag_name}")
        if not tag_name == "br":
            out.append({"basic_type": "tag", "kind": "</", "value": f"{str(tag_name)}"})
        # print(f"added closing tag {tag_name} at index {len(out)}")
        return out


def tag_dict_to_str(tag):
    tag_kind_to_str = {'</': '/', '<': ''}
    if tag['basic_type'] == 'string':
        return tag['value']
    elif tag['basic_type'] == 'tag':
        return f"<{tag_kind_to_str[tag['kind']]}{tag['value']}>"
    else:
        raise ValueError(f"Incorrect tag dict: `{str(tag)}`")


class InputStream:
    """
    Main input class
    (after https://lisperator.net/pltut/parser/input-stream)
    """
    def __init__(self, inp):
        self.pos, self.line, self.col = 0, 1, 0

        self.inp = inp
        self.flat_inp = flatten_soup_tag(inp)
        self.flat_list_pos = 0

        # append sentinel value
        self.eof_tag_value = '%EOF%'
        self.flat_inp.append(
            dict(basic_type='sentinel', value=self.eof_tag_value))

    def next(self):
        """
        :return: next whole tag or symbol in stream, popping it from stream
        """
        # TODO: what to return, dict or string? depends on tokenizer
        next_el = self.flat_inp[self.flat_list_pos]
        if next_el['basic_type'] == 'tag':
            self.pos = 0

            ch_or_tag = next_el
            self.flat_list_pos += 1

        elif next_el['basic_type'] == 'string':
            try:
                ch_or_tag = next_el['value'][self.pos]
                self.pos += 1
            except IndexError as e:
                # the string has ended, so we move to the next element
                self.flat_list_pos += 1
                self.pos = 0

                further_el = self.flat_inp[self.flat_list_pos]
                if further_el['basic_type'] == 'tag':
                    ch_or_tag = further_el
                    self.flat_list_pos += 1
                elif further_el['basic_type'] == 'string':
                    # TODO: this may be an impossible route actually
                    ch_or_tag = further_el['value'][self.pos]
                    self.pos += 1
                else:
                    self.croak("Basic type not supported")

        elif next_el['basic_type'] == 'sentinel':
            ch_or_tag = next_el['value']
        else:
            self.croak("Unknown basic type")

        if (isinstance(ch_or_tag, dict) and ch_or_tag['value'] == 'br'
                or ch_or_tag == '\n'):
            self.line += 1
            self.col = 0
        elif ch_or_tag != self.eof_tag_value:
            self.col += len(ch_or_tag)

        return ch_or_tag

    def peek(self):
        cur_el = self.flat_inp[self.flat_list_pos]
        if cur_el['basic_type'] == 'tag':
            ch_or_tag = cur_el
        elif cur_el['basic_type'] == 'string':
            try:
                ch_or_tag = cur_el['value'][self.pos]
            except IndexError as e:
                # the string has ended, so we move to the next element
                flat_list_pos = self.flat_list_pos + 1
                pos = 0

                next_el = self.flat_inp[flat_list_pos]
                if next_el['basic_type'] == 'tag':
                    ch_or_tag = next_el
                elif next_el['basic_type'] == 'string':
                    # TODO: this may be an impossible route actually
                    ch_or_tag = next_el['value'][pos]  # TODO: although route may be impossible, there should be pos, not self.pos
                else:
                    self.croak("Basic type not supported")

        elif cur_el['basic_type'] == 'sentinel':
            ch_or_tag = cur_el['value']
        else:
            self.croak("Unknown basic type")

        return ch_or_tag

    def eof(self):
        return self.peek() == self.eof_tag_value

    def croak(self, msg):
        # note: positions are presented as they are in output of `lxml` parser
        #   this isn't necessarily the same as in browser DOM
        # TODO: improve so the spot is shown (check correctness of number along the way!)
        #   (through history of tokens?)
        raise ValueError(f"{msg} ({str(self.line)}:{str(self.col)})")

    def warn(self, msg):
        # note: positions are presented as they are in output of `lxml` parser
        #   this isn't necessarily the same as in browser DOM
        raise RuntimeWarning(f"{msg} ({str(self.line)}:{str(self.col)})")


def compose_predicate(subpredicates, type='and'):
    composition = {'and': all, 'or': any}[type]

    def predicate(inp):
        return composition(subpredicate(inp) for subpredicate in subpredicates)
    return predicate


class TokenStream:
    def __init__(self, inp: InputStream):
        self.inp = inp
        self.unknown = object()
        self.current = None

    @staticmethod
    def is_sakha_only(ch):
        return ch in SAKHAONLY_LETTERS

    @staticmethod
    def is_sakha(ch):
        return ch in SAKHA_ALPHABET

    @staticmethod
    def is_rus(ch):
        return ch in RUS_ALPHABET

    @staticmethod
    def is_word_char(ch, word_regex=re.compile("[A-Za-zА-Яа-яЁёҤҥҔҕӨөҺһҮү-]")):
        return bool(word_regex.match(ch))

    @staticmethod
    def is_whitespace(ch, re_whitespace=re.compile("[ \t]")):
        return ch in (' ', '\t')

    @staticmethod
    def is_punc(ch, excl='', incl=None):
        if not incl:
            return ch in """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" and ch not in excl
        else:
            return ch in incl

    @staticmethod
    def is_roman(ch, roman_set=frozenset('MDCLXVI')):
        return ch in roman_set

    def read_while(self, predicate: Callable[[Union[str, Dict]], bool]):
        inp = self.inp
        string = ""
        # TODO: isinstance is needed for predicates with regex and membership checking
        while not inp.eof() and isinstance(inp.peek(),str) and predicate(inp.peek()):
            string += inp.next()
        return string

    def read_arabic_number(self, num_regex=re.compile('^[1-9][0-9]*$')):
        string = self.read_while(str.isdecimal)
        if num_regex.fullmatch(string):
            return dict(type='arabic_number', valid=True, value=int(string))
        else:
            log.debug(f"arabic_number string `{string}` not matched fully by regex pattern")
            return dict(type='arabic_number', valid=False, value=string)

    # def read_bracketed(self, start, end):
    #     inp = self.inp
    #
    #     string = ''
    #     first_char = inp.next()
    #     if first_char != start:
    #         log.debug(f"first_char of the string isn't start=`{start}`")
    #         # TODO: to allow this or not?
    #     while not inp.eof():
    #         ch = inp.next()
    #         if ch == end:
    #             break
    #         elif isinstance(ch, dict) and ch['basic_type'] == 'tag':
    #             log.warning(f"encountered tag!")
    #             break
    #         else:
    #             string += ch
    #
    #     return string
    #
    # def read_rus_comment(self):
    #     return self.read_bracketed("(", ")")

    def read_roman_number(
        self, re_roman=re.compile(
            "^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
            re.M
        )
    ):
        string = self.read_while(self.is_roman)
        if re_roman.fullmatch(string):
            return dict(type='roman_number', valid=True, value=string)
        else:
            log.debug(f"roman_number string `{string}` not matched fully by regex pattern")
            return dict(type='roman_number', valid=False, value=string)

    def read_word(self):
        string = self.read_while(self.is_word_char)
        if any(map(self.is_sakha_only, string)):
            return dict(type='word', lang='sa', value=string)
        else:
            return dict(type='word', lang=0, value=string)

    def read_russian(self):
        # predicate = compose_predicate(
        #     [
        #         self.is_rus,
        #         lambda ch: not self.is_sakha_only(ch),
        #         partial(self.is_punc, excl=';'),
        #         self.is_whitespace
        #     ],
        #     type='or'
        # )
        # string = self.read_while(predicate)

        predicate = compose_predicate(
            [
                self.is_rus,
                lambda ch: not self.is_sakha_only(ch),
            ],
        )
        string = self.read_while(predicate)
        return dict(type='word', value='string', lang='pos_russian')

    def read_next(self):
        inp = self.inp
        if inp.eof():
            return None

        ch_or_tag = inp.peek()
        logging.debug(f"ch_or_tag is `{ch_or_tag}`")
        # TODO: if or else if?
        if isinstance(ch_or_tag, dict):
            if ch_or_tag['value'] == 'br':
                inp.next()
                return dict(type='newline', kind='br')
            else:
                inp.next()
                ch_or_tag['type'] = 'tag'
                ch_or_tag.pop('basic_type')
                return ch_or_tag

            # TODO: this code attempts parsing instead of tokenizing

            # elif ch_or_tag['value'] == 'em' and ch_or_tag['kind'] == '<':
            #     # TODO: is it only grammar comments in russian that are like this?
            #     tok = dict(type='string', value=self.read_rus_comment())
            #     # TODO: move through a token, but check if it's the correct one
            #     tag = inp.next()
            #     if not (tag['value'] == 'em' and ch_or_tag['kind'] == '</'):
            #         inp.warn(f"Tag not closed: `{inp}`, found `{tag}` instead")
            #     return tok  # TODO:
            #
            # elif ch_or_tag['value'] == 'strong':
            #     # TODO: can't do or, because `.next()` consumes. Should peek to decide
            #     further_el = inp.peek()
            #     return self.read_roman_number() or self.read_russian()
            # else:
            #     print(f'not implemented for `{ch_or_tag}`')
            #     inp.next()
            #     return object()

        else:
            if ch_or_tag == '\n':
                return dict(type='newline', kind=inp.next())
            elif self.is_whitespace(ch_or_tag):
                return dict(type='whitespace', kind=inp.next())
            # elif ch_or_tag == '(':
            #     return dict(type='string', value=self.read_bracketed('(', ')'))
            elif ch_or_tag == '=':  # affixes and clitics in Sakha are coded so
                return dict(type='string', lang='sa',
                            value=inp.next()+self.read_while(str.isalpha))
            elif self.is_punc(ch_or_tag):
                return dict(type='punc', value=inp.next())
            elif ch_or_tag.isdecimal():
                return self.read_arabic_number()
            elif self.is_roman(ch_or_tag):
                return self.read_roman_number()
            elif ch_or_tag.isalpha():
                return self.read_word()
            else:
                log.error(f'not implemented for `{ch_or_tag}`')
                inp.next()
                return self.unknown

    # TODO: что если вовзращать объект класса такой же как дикт, но был бы аргумент
    #   типа длины который для строк - их длину, для тэгов - длину внутренностей

    def peek(self):
        if not self.current:
            self.current = self.read_next()
        return self.current

    def next(self):
        tok = self.current
        self.current = None
        return tok or self.read_next()