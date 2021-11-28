import re
import logging
import logging.config  # TODO: will have to be defined in main module in the future
from collections.abc import MutableMapping
from functools import partial, wraps
from typing import Union, Dict, Callable

from bs4 import NavigableString, Tag

from utils import SAKHA_ALPHABET, SAKHAONLY_LETTERS, RUS_ALPHABET

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def flatten_soup_tag(tag):
    out = []
    if tag.name is None:  # we have a NavigableString instance
        return [{"basic_type": "string", "value": str(tag)}]
    else:
        tag_name = tag.name
        out.append({"basic_type": "tag", "kind": "<", "value": f"{str(tag_name)}"})

        i = 0
        for i, subtag in enumerate(tag.children):
            out.extend(flatten_soup_tag(subtag))
        if not tag_name == "br":
            out.append({"basic_type": "tag", "kind": ">", "value": f"{str(tag_name)}"})

        return out


def tag_dict_to_str(tag):
    tag_kind_to_str = {'>': '/', '<': ''}
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
        self.flat_inp.append(dict(basic_type='sentinel'))

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
            ch_or_tag = None
        else:
            self.croak("Unknown basic type")

        if (isinstance(ch_or_tag, dict) and ch_or_tag['value'] == 'br'
                or ch_or_tag == '\n'):
            self.line += 1
            self.col = 0
        elif ch_or_tag is not None:
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
            ch_or_tag = None
        else:
            self.croak("Unknown basic type")

        return ch_or_tag

    def eof(self):
        return self.peek() is None

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


def make_read_next_skip_space_glob(func):
    print("inside decorator")

    @wraps(func)
    def wrapped_read_next(self):
        print(f"inside wrapped")
        self.read_while(self.is_whitespace)
        return func(self)

    return wrapped_read_next


class TokenStream:
    """
    reads input characters (and tags) one by one and forms tokens
    tokens are of one of the following types:
      newline:
        kind='br','\n'
      whitespace:
        kind=' ', '\t'
      tag:
        kind='<','>'
      arabic_number
        1, 16, 53
      roman_number
        I, II, X
      word
      punct
    all have `value` field except whitespace and newline
    """
    def __init__(self, inp: InputStream, skip_space=True):
        self.inp = inp
        self.unknown = object()
        self.current = None

        self.__class__.read_next = self.__class__.read_next_keep_space
    #     self.skip_space = skip_space
    #
    #     if self.skip_space:
    #         self.__class__.read_next = self.__class__.read_next_skip_space
    #     else:
    #         self.__class__.read_next = self.__class__.read_next_keep_space
    #
    # def make_read_next_skip_space(func):
    #     @wraps(func)
    #     def wrapped_read_next(self):
    #         # self.read_while(self.is_whitespace)
    #         # print(f"before first while")
    #         while not self.inp.eof() and self.is_whitespace(self.inp.peek()):
    #             self.inp.next()
    #         # print(f"after first while, w1_c = {w1_c}")
    #         res = func(self)
    #         logger.info(f"after res: {res}")
    #         return res
    #     return wrapped_read_next
    #
    # @make_read_next_skip_space
    # def read_next_skip_space(self):
    #     return self.read_next_keep_space()
    #
    # make_read_next_skip_space = staticmethod(make_read_next_skip_space)

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
    # recent change: `=` added to char list (to account for it representing
    #  affix nature of expression in sakhatyla
    def is_word_char(ch, word_regex=re.compile("[A-Za-zА-Яа-яЁёҤҥҔҕӨөҺһҮү=-]")):
        return bool(word_regex.match(ch))

    @staticmethod
    def is_whitespace(ch):
        return ch in (' ', '\t')

    @staticmethod
    def is_punc(ch, excl='', incl=None):
        if not incl:
            return ch in """!"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\""" and ch not in excl
        else:
            return ch in incl

    @staticmethod
    def is_roman(ch, roman_set=frozenset('MDCLXVI')):
        return ch in roman_set

    def read_while(self, predicate: Callable[[Union[str, Dict]], bool]):
        inp = self.inp
        string = ""
        # TODO: isinstance is needed for predicates with regex and membership checking
        while not inp.eof() and isinstance(inp.peek(), str) and predicate(inp.peek()):
            string += inp.next()
        return string

    def read_arabic_number(self, num_regex=re.compile('^[1-9][0-9]*$')):
        string = self.read_while(str.isdecimal)
        if num_regex.fullmatch(string):
            tok = dict(type='arabic_number', valid=True, value=int(string))
        else:
            logger.debug(f"arabic_number string `{string}` not matched fully by regex pattern")
            tok = dict(type='arabic_number', valid=False, value=string)
        if self.inp.peek() == '.':
            logger.debug(f"\tchecking dot")
            self.inp.next() # TODO: should dot be consumed here?
            tok['dotted'] = True

        return tok

    def read_roman_number(
        self, re_roman=re.compile(
            "^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
            re.M
        )
    ):
        string = self.read_while(self.is_roman)
        # TODO: check correctness of dot appending
        if re_roman.fullmatch(string):
            return dict(type='roman_number', valid=True, value=string)
        else:
            logger.debug(f"roman_number string `{string}` not matched fully by regex pattern")
            return dict(type='roman_number', valid=False, value=string)

    def read_word(self):
        string = self.read_while(self.is_word_char)
        if any(map(self.is_sakha_only, string)):
            word = dict(type='word', lang='sa', value=string)
        else:
            word = dict(type='word', lang=0, value=string)
        if string[0] == '=':
            word["affix"] = "suff"
        elif len(string) > 1 and string[-1] == '=':
            if word["lang"] == "sa":
                word.update({"affix": "root", "pos": "V"})
            else:
                word["affix"] = "ru_pref|sa_root"

        return word

    def read_next_keep_space(self):
        inp = self.inp
        ch_or_tag = inp.peek()
        if ch_or_tag is None:
            return None

        logger.debug(f"ch_or_tag is `{ch_or_tag}`, current is `{self.current}`")
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
            #     if not (tag['value'] == 'em' and ch_or_tag['kind'] == '>'):
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
            elif ch_or_tag == '=':  # affixes and clitics in Sakha are coded so
                # TODO: not only sakha ones are codded (`в=`)
                return dict(type='word', lang='sa',
                            value=inp.next()+self.read_while(str.isalpha))
            elif self.is_punc(ch_or_tag):
                # # TODO: delete. This can't be working as whitespace is still present
                # if ch_or_tag == ';':
                #     # this is in order to skip `;` at the end of example, which
                #     #   otherwise messes everyting up
                #     value = inp.next()
                #     next_ch_or_tag = inp.peek()
                #     if next_ch_or_tag.isdecimal():
                #         return self.read_arabic_number()
                # else:
                #     value = inp.next()
                return dict(type='punc', value=inp.next())
            elif ch_or_tag.isdecimal():
                return self.read_arabic_number()
            elif self.is_roman(ch_or_tag):
                return self.read_roman_number()
            elif ch_or_tag.isalpha():
                return self.read_word()
            else:
                logger.error(f'not implemented for `{ch_or_tag}`')
                inp.next()
                return self.unknown

    # TODO: что если вовзращать объект класса такой же как дикт, но был бы аргумент
    #   типа длины который для строк - их длину, для тэгов - длину внутренностей

    def peek(self):
        logger.debug(f"in `peek`: current is {self.current}")
        if not self.current:
            self.current = self.read_next()
        return self.current

    def next(self):
        logger.info(f"in `next`: current is {self.current}")
        tok = self.current
        self.current = None
        return tok or self.read_next()

    def eof(self):
        logger.debug(f"in `eof`: current is {self.current}")
        return self.peek() is None

    def croak(self, *args, **kwargs):
        self.inp.croak(*args, **kwargs)


class TokenFeeder():
    def __init__(self, inp: TokenStream, skip_space=True):
        self.inp = inp
        self.skip_space = skip_space

        # self.tag_to_skip = None

        if self.skip_space:
            # self.__class__.next = self.__class__.next_skip_space
            self.__class__.peek = self.__class__.peek_skip_space
        else:
        #     self.__class__.next = self.__class__.next_keep_space
            self.__class__.peek = self.__class__.peek_skip_space

    def make_next_skip_tag(func_next):
        def wrapped_read_next(self):
            res = func_next(self)
            logger.debug(f"in `wrapped_read_next`, before `if`: {res}")
            if not self.eof() and bool(res) and res['type'] == 'tag':

                and (not self.tag_to_skip.get('value')
                     or res['value'] == self.tag_to_skip['value'])
                and (not self.tag_to_skip.get('kind')
                     or res['kind'] == self.tag_to_skip['kind']


                logger.info(f"skipping extraneous tag {self.tag_to_skip}")
                # self.inp.current = None
                del self.tag_to_skip
                new_res = self.inp.next()
                return new_res
            else:
                return res

        return wrapped_read_next

    @make_next_skip_tag
    def next(self):
        return self.inp.next()

    make_next_skip_tag = staticmethod(make_next_skip_tag)

    def peek_skip_space(self):
        while (self.inp.peek() or {}).get("type") == "whitespace":
            print(self.inp.peek(), self.inp.current)
            self.inp.next()

        return self.inp.peek()

    def peek_keep_space(self):
        res = self.inp.peek()
        return res

    def eof(self):
        return self.inp.eof()

    def croak(self, *args, **kwargs):
        self.inp.croak(*args, **kwargs)


def compose_predicates_or(*predicates):
    def is_any():
        return any(predicate() for predicate in predicates)
    return is_any


class Parser:
    def __init__(self, inp: TokenStream):
        self.inp = inp
        self._tok_type_to_method = dict(
            whitespace=self.is_whitespace, tag=self.is_tag,
            number=self.is_number, punc=self.is_punc,
            word=self.is_word
        )

    def is_whitespace(self):
        tok = self.inp.peek()
        return bool(tok) and tok['type'] == 'whitespace'

    def is_number(self, kind='arabic', value=None, dotted=None):
        tok = self.inp.peek()
        return (bool(tok) and tok['type'] == f"{kind}_number"
                and (not value or tok['value'] == value)
                and (not dotted or tok.get('dotted') == dotted))

    def is_tag(self, tag_value=None, tag_kind=None):
        tok = self.inp.peek()
        return (bool(tok) and tok['type'] == 'tag'
                and (not tag_value or tok['value'] == tag_value)
                and (not tag_kind or tok['kind'] == tag_kind))

    def is_punc(self, value=None):
        tok = self.inp.peek()
        return (bool(tok) and tok['type'] == 'punc'
                and (not value or tok['value'] == value))

    def is_word(self):
        tok = self.inp.peek()
        return bool(tok) and tok['type'] == 'word'

    def is_tok(self, desired_tok):
        tok_type = desired_tok.pop('type')
        # if tok_type in ('tag', 'number'):
        #     self._tok_type_to_method[tok_type](
        #         desired_tok['value'], desired_tok.get('kind')
        #     )
        # else:
        #     self._tok_type_to_method[tok_type]()
        return self._tok_type_to_method[tok_type](**desired_tok)

    def skip_whitespace(self):
        if self.is_whitespace():
            self.inp.next()
        else:
            self.inp.croak(f"Expecting whitespace")

    def skip_number(self, kind='arabic', dotted=None):
        if self.is_number(kind, dotted=dotted):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting number")

    def skip_tag(self, tag_value, tag_kind=None):
        if self.is_tag(tag_value, tag_kind):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting tag {tag_kind}{tag_value}>")

    def skip_punc(self, value=None):
        if self.is_punc(value):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting punctuation `{value}`")

    def skip_word(self):
        if self.is_word():
            self.inp.next()
        else:
            self.inp.croak(f"Expecting word")

    def skip_tok(self, desired_tok):
        if self.is_tok(desired_tok):
            self.inp.next()
        else:
            self.inp.croak(f"Expecting token {desired_tok}")

    # def delimited(self, start, stop, parser):
    #     inp = self.inp
    #
    #     a = []
    #     self.skip_tok(start)
    #     while not inp.eof():
    #         if self.is_tok(stop):
    #             break
    #         a.append(parser())
    #     self.skip_tok(stop)
    #     return a

    # def parse_gram_info(self):
    #     self.skip_tag('em', '<')
    #
    def parse_word(self):
        if self.is_word():
            return self.inp.next()
        else:
            self.inp.croak(f"Expecting word")

    def parse_words(self):
        inp = self.inp
        # TODO: tag reliance here currently
        #   note: in sa-ru sah example is strong and russian normal

        def get_subarray_of(arr):
            l = []
            arr.append(l)
            return l

        ru_example = []
        sa_example = []
        sa_ru_example = []
        open_tag = None
        array = None
        cur_array = None  # TODO: better changed to source / targ for generality later
        # while not inp.eof():
        while self.is_word() or self.is_punc(',') or self.is_tag('strong'):
            # TODO:
            if self.is_tag('strong', '<'):
                self.skip_tag('strong', '<')
                if not open_tag:
                    open_tag = True
                    array = ru_example # change array
                    cur_array = 'ru'
                else:
                    inp.croak(f"Unexpected tag: `{inp.peek()}`. <strong> already open")

            if self.is_tag('strong', '>'):
                self.skip_tag('strong', '>')
                if open_tag:
                    open_tag = None
                    array = get_subarray_of(sa_example) # change array
                    cur_array = 'sa'
                else:
                    inp.croak("Expected word or closing </strong>")
            elif self.is_tag():
                inp.croak(f"Unexpected tag: `{inp.peek()}`")

            if self.is_punc(','):
                # translation to target lang has multiple options
                if cur_array != 'sa':  # likely part of source lang example
                    array.append(inp.next())
                else:
                    self.skip_punc()
                    array = get_subarray_of(sa_example)

            if array is None:
                array = sa_ru_example

            if self.is_word():
                array.append(inp.next())

        res = {"type": "example"}
        # TODO: does introduction of subarrays mess with truthiness check?
        if len(sa_example) == 1:
            sa_example = sa_example[0]

        if ru_example and sa_example:
            res.update(dict(ru_example=ru_example, sa_example=sa_example))
        elif not(ru_example or sa_example):
            if sa_ru_example:
                res.update(dict(sa_ru_example=sa_ru_example))
            else:
                # inp.croak("No results to show")
                return None

        return res

    def parse_delimited(self, start, separator, parser,
                        stop_punc=None, stop_punc_list=None, stop_cond_func=None):
        # TODO: whitespace will interfere!
        a = []
        first = True
        if start:
            self.skip_punc(start)
        while not self.inp.eof():
            # # in case passed argument is a function first clause should do
            # if self.is_punc(stop):
            #     break
            if stop_punc and self.is_punc(stop_punc):
                break
            elif (stop_punc_list
                  and any(self.is_punc(punc) for punc in stop_punc_list)):
                break
            elif stop_cond_func and stop_cond_func():
                break

            if first:
                first = False
            else:
                # TODO: `;` before next number at the end of numbered example is skipped too
                #  looks like it is only remedied by parser
                # TODO: no, it should be skipped here or it'll loop otherwise
                self.skip_punc(separator)

            parse = parser()
            if parse:
                a.append(parse)  # make sure parsers don't consume separator or stop

        if stop_punc:
            self.skip_punc(stop_punc)
        return a

    def parse_numbered_sense(self):
        inp = self.inp
        numbered_sense = {}

        # TODO: that belongs higher up
        # if self.is_punc('('):
        #     # TODO: what parser to choose? perhaps need something simple for words
        #     synonyms = self.parse_delimited('(', ')', ',', self.parse_word)
        #     numbered_sense["synonyms"] = synonyms

        # gram_desc
        if self.is_tag('em', '<'):
            gram_desc = []
            gram_desc_ru = []
            self.skip_tag('em', '<')
            # TODO: language choice should be made here
            #  or rather type should be simply 'gram_desc' with no lang
            while not inp.eof():
                if self.is_tag('em', '>'):
                    break
                gram_desc_ru.append(inp.next())
            self.skip_tag('em', '>')
            numbered_sense['gram_desc_ru'] = gram_desc_ru

            # optionaly there can be translation, usually with affix, represented as =...
            if not self.is_punc(';'):
                # TODO: the separator here often (always?) isn't `,`, it's `;`
                #  possible solution: allow passing function for `stop` and pass
                #   `self.is_number` or composion of `self.is_tag`` and `self.is_number`

                # TODO: another problem: after translation there could be
                #   (what?? grammar explanation in target lang?) in ()
                sa_transl = self.parse_delimited(None, ',', self.parse_word,
                    # stop_punc_list=[';', '(']
                    stop_cond_func=compose_predicates_or(
                        lambda: self.is_punc(';'), lambda: self.is_tag('em')
                    )
                )
                if self.is_punc(';'):
                    self.skip_punc()
                elif self.is_tag('em'):
                    gram_desc_sa = []
                    self.skip_tag('em')
                    while not inp.eof():
                        if self.is_tag('em', '>'):
                            break
                        if self.is_punc(')') or self.is_punc(';'):
                            self.skip_punc()
                            continue
                        gram_desc_sa.append(inp.next())
                    self.skip_tag('em', '>')
                    numbered_sense['gram_desc_sa'] = (
                        gram_desc_sa if gram_desc_sa[-1]['value'] != ')'
                        else gram_desc_sa[:-1])

                numbered_sense['sah_transl'] = sa_transl

        # word translation
        # if self.is_whitespace():
        if not self.is_punc(';'):
            # self.skip_whitespace()
            # TODO: <strong> in в.9 messes is_number checking
            #   one idea: prevent parse_delimited from consuming `;` which isn't separator
            #   but is actual stop. But that was the solution to its indistinguishability
            #   from separator `;`
            #   possible solution: split `parse_words` into smaller functions
            #   and use maybe_stop in `parse_delimited`, and if it's not stop just parse it
            #     this it tied to another TODO: pass mutable dicts and lists and add on the go
            sah_sense_translations = self.parse_delimited(
                None, ';', self.parse_words, stop_cond_func=compose_predicates_or(
                        self.is_number
                    )
            )
            numbered_sense['sah_sense_translations'] = sah_sense_translations

        return numbered_sense

    def parse_sense(self):
        return "NOT IMPLEMENTED"

    def maybe_pos(self):
        inp = self.inp
        word = self.parse_word()

        if self.is_word():
            words = [word]
            while not inp.eof() and self.is_word():
                words.append(self.parse_word())
            return dict(type='grammar_desc???', value=words)
        elif self.is_number() or self.is_punc('(') or self.is_tag(): # TODO: how is that supposed to work?
            return dict(type='pos', value=word)
        else:
            inp.croak("Unexpected token")

    def parse_atom(self):
        inp = self.inp
        tok = inp.peek()

        if self.is_tag('em', '<'):
            self.skip_tag('em', '<')
            res = self.maybe_pos()
            self.skip_tag('em', '>')
            return res
        elif self.is_tag():
            # TODO: decide what to do with tags
            # turns out this accidentally skips extraneous tag surrounding number for example!
            #   (ex. `в`.3.9). Something still needs to be done about the closing tag though
            print(f"cur tok is tag: {tok}")
            tag = inp.next()
            self.inp.tag_to_skip = dict(type="tag", value=tag['value'])
            return None

        if self.is_number():
            if self.is_number(kind='arabic', dotted=True):
                # TODO: logging example number could be done here
                #   e.g. `parse_numbered_sense` fails (then we take all till next number or <p> tag)
                num = inp.next()['value']
                if hasattr(inp, "tag_to_skip"):
                    print(f"attr value: {inp.tag_to_skip}, next el would be {inp.peek()}")
                else:
                    print(f"no attr `tag_to_skip`, next el would be {inp.peek()}")
                numbered_sense = {'type': 'numbered_sense', 'num': num}
                numbered_sense.update(self.parse_numbered_sense())
                return numbered_sense
            elif self.is_number(kind='roman'):
                self.skip_number(kind='roman')
                return dict(type="sense", value=self.parse_sense)

        if self.is_punc('('):
            synonyms = self.parse_delimited('(', ',', self.parse_word, stop_punc=')')
            return dict(type='synonyms', value=synonyms)

    def parse_entry(self):
        """parses <div> - the whole lexeme entry"""
        inp = self.inp
        entry = []
        while inp and not inp.eof():  # and inp needed due to None instead of whitespace in
            entry.append(self.parse_atom())
        return dict(type='entry', entry=entry)
