import re

from bs4 import NavigableString, Tag


def flatten_soup_tag(tag):
    out = []
    if tag.name is None:  # we have a NavigableString instance
        print(f"current tag is:\n{tag}\nthis is a string\n\n")
        return [{"basic_type": "string", "value": str(tag)}]
    else:
        print(f"current tag is:\n{tag}\nthis is a tag. diving into descendants\n\n")
        tag_name = tag.name
        out.append({"basic_type": "tag", "kind": "opening", "value": f"{str(tag_name)}"})
        print(f"added opening tag {tag_name} at index {len(out)}")

        i = 0
        for i, subtag in enumerate(tag.children):
            out.extend(flatten_soup_tag(subtag))

        print(f"added {i+1} descendants of {tag_name}")
        if not tag_name == "br":
            out.append({"basic_type": "tag", "kind": "closing", "value": f"{str(tag_name)}"})
        print(f"added closing tag {tag_name} at index {len(out)}")

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

        self.is_cur_inp_type_tag = True

    def next(self):
        """

        :return: next whole tag or symbol in string
        """
        # TODO: what to return, dict or string? depends on tokenizer
        if self.is_cur_inp_type_tag:
            self.pos = 0

            ch_or_tag = self.flat_inp[self.flat_list_pos]['value']
            self.flat_list_pos += 1

            if self.flat_inp[self.flat_list_pos]['basic_type'] == 'string':
                self.is_cur_inp_type_tag = False

        else:
            try:
                ch_or_tag = self.flat_inp[self.flat_list_pos]['value'][self.pos]
                self.pos += 1
            except IndexError as e:
                # the string has ended, so we move to the next element
                self.flat_list_pos += 1
                self.pos = 0

                next_el = self.flat_inp[self.flat_list_pos]
                if next_el['basic_type'] == 'tag':
                    self.is_cur_inp_type_tag = True
                    ch_or_tag = next_el['value']

                    self.flat_list_pos += 1

                elif next_el['basic_type'] == 'string':
                    # TODO: this may be an impossible route actually
                    self.is_cur_inp_type_tag = False
                    self.pos = 0
                    ch_or_tag = next_el['value'][self.pos]
                else:
                    self.croak("Basic type not supported")

        if (ch_or_tag in ('<br>', '<br/>', '<br />')
                or ch_or_tag == '\n'):
            self.line += 1
            self.col = 0
        else:
            self.col += len(ch_or_tag)

        return ch_or_tag

    def peek(self):
        pass

    def eof(self):
        # TODO: rework
        return self.peek().next_sibling is None

    def croak(self, msg):
        # note: positions are presented as they are in output of `lxml` parser
        #   this isn't necessarily the same as in browser DOM
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



