import re
from functools import partial

from bs4 import NavigableString, Tag

from utils import SAKHA_ALPHABET, SAKHAONLY_LETTERS, RUS_ALPHABET


def flatten_soup_tag(tag):
    out = []
    if tag.name is None:  # we have a NavigableString instance
        # print(f"current tag is:\n{tag}\nthis is a string\n\n")
        return [{"basic_type": "string", "value": str(tag)}]
    else:
        # print(f"current tag is:\n{tag}\nthis is a tag. diving into descendants\n\n")
        tag_name = tag.name
        out.append({"basic_type": "tag", "kind": "opening", "value": f"{str(tag_name)}"})
        # print(f"added opening tag {tag_name} at index {len(out)}")

        i = 0
        for i, subtag in enumerate(tag.children):
            out.extend(flatten_soup_tag(subtag))

        # print(f"added {i+1} descendants of {tag_name}")
        if not tag_name == "br":
            out.append({"basic_type": "tag", "kind": "closing", "value": f"{str(tag_name)}"})
        # print(f"added closing tag {tag_name} at index {len(out)}")

        return out


def tag_dict_to_str(tag):
    tag_kind_to_str = {'closing': '/', 'opening': ''}
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
                    ch_or_tag = next_el['value'][self.pos]
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
        raise ValueError(msg + " (" + str(self.line) + ":" + str(self.col) + ")")

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
    def is_whitespace(ch, re_whitespace=re.compile("[ \t]")):
        return re_whitespace.match(ch)

    @staticmethod
    def is_punc(ch, excl='', incl=None):
        if not incl:
            return ch in """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~""" and ch not in excl
        else:
            return ch in incl

    def read_while(self, predicate):
        inp = self.inp
        string = ""
        while not inp.eof() and predicate(inp.peek()):
            string += inp.next()
        return string

    def read_bracketed(self, start, end):
        inp = self.inp

        string = ''
        first_char = inp.next()
        if first_char != start:
            print(f"first_char of the string isn't start=`{start}`")
            # TODO: to allow this or not?
        while not inp.eof():
            ch = inp.next()
            if ch == end:
                break
            elif isinstance(ch, dict) and ch['basic_type'] == 'tag':
                inp.warn(f"encountered tag!")
                break
            else:
                string += ch

        return string

    def read_rus_comment(self):
        return self.read_bracketed("(", ")")

    def read_roman(
        self, re_roman=re.compile(
        "^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$", re.M
    )):
        string = self.read_while(str.isalpha)
        if re_roman.match(string):
            return dict(type='roman_number', value='string')
        # else


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
        # TODO: if or else if?
        if isinstance(ch_or_tag, dict):
            if ch_or_tag['value'] == 'br':
                inp.next()
                return dict(type='newline', kind='br')
            elif ch_or_tag['value'] == 'em' and ch_or_tag['kind'] == 'opening':
                # TODO: is it only grammar comments in russian that are like this?
                tok = dict(type='string', value=self.read_rus_comment())
                # TODO: move through a token, but check if it's the correct one
                tag = inp.next()
                if not (tag['value'] == 'em' and ch_or_tag['kind'] == 'closing'):
                    inp.warn(f"Tag not closed: `{inp}`, found `{tag}` instead")
                return tok

            elif ch_or_tag['value'] == 'strong':
                # TODO: can't do or, because `.next()` consumes. Should peek to decide
                further_el = inp.peek()
                return self.read_roman() or self.read_russian()
            else:
                print(f'not implemented for `{ch_or_tag}`')
                inp.next()
                return object()

        else:
            if ch_or_tag == '(':
                return dict(type='string', value=self.read_bracketed('(', ')'))
            else:
                print(f'not implemented for `{ch_or_tag}`')
                inp.next()
                return object()


# def is_digit(ch: str, digits=frozenset('0123456789')):
#     return ch in digits
# str.isdigit()




def transl_num(item: str, transl_num_regex=re.compile('^[1-9][0-9]*$')):
    if transl_num_regex.search(item):
        return True
    return False
