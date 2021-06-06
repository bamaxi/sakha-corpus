from typing import List, Dict, Union
import csv
import re

import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString, Tag

from utils import HEADERS

sakha_link = "https://sakhatyla.ru/" + "translate?q="

words = ("и", "в")

# def get_similar
#     TODO

Translation = Dict[str, Union[str, Tag]]

def get_word_transl(word: str) -> Dict[str, Union[str, List[Translation]]]:
    session = requests.session()
    link = sakha_link + requests.utils.quote(word)
    response = session.get(link, headers=HEADERS)

    if response.status_code == 200:
        print(f"Successfully loaded `{sakha_link+word}`")
    else:
        raise ValueError("Something went wrong while getting result")

    words = []

    soup = BeautifulSoup(response.text, 'lxml')

    # NEW: delete linebreaks
    for linebreak in soup.find_all('br'):
        linebreak.extract()

    # # найти <div> в котором
    # transl_headers = soup.find_all('h2')
    # if not any(header.string == 'Русский → Якутский'
    #            for header in transl_headers):
    #     comment = (f"нет русского перевода. "
    #                f"есть переводы `{','.join(transl_headers)}`")
    #     print(comment)
    #     return ... # TODO
    # достать тэг `<h2>Русский → Якутский</h2>` и смотреть его сестёр дальше
    #   пока не попадём на очередной <h2> (или `ещё переводы`)
    Translation = Dict[str, Union[str, Tag]]
    translation_tags: List[Translation] = []
    res: Dict[str, Union[str, List[Translation]]]
    comment = ''

    transl_header = soup.find_all('h2', string="Русский → Якутский")
    if not transl_header:
        comment = f"нет русского перевода. "
                   # f"есть переводы `{','.join(transl_headers)}`")
        print(comment)
        res = dict(word=word, translations=[], link=link, comment=comment)
        return res

    for tag in transl_header[0].next_siblings:
        if isinstance(tag, NavigableString):
            continue
        elif tag.name == 'h2':
            print(f"encountered `<h2>`, ending loop (`{tag.string}`)")
            break
        elif tag.name == 'p' or tag.name == 'hr':
            print(f"encountered `<p>` or `<hr>`, ending loop (`{tag.string}`)")
            break
        else:
            # TODO: сохранять все тэги или сразу убирать словосочетания?
            rus_word_or_phrase = tag.h3.string.strip()
            translation = tag.find('div', class_='article-text')
            try:
                lexical_category = tag.find('div', class_='article-category').string
            except (AttributeError) as e:
                lexical_category = ''
            word_or_phrase_res = dict(
                rus=rus_word_or_phrase, translation=translation,
                lexical_category=lexical_category)
            translation_tags.append(word_or_phrase_res)

    print(f'Successfully parsed `{sakha_link+word}`')
    res = dict(word=word, translations=translation_tags, link=link, comment=comment)
    return res


def rec_parse_translation(par, transl_sep=' : '):
    res = []

    def is_sense_in_rus(tag):
        return tag.name == 'em' and not ';' in tag.string

    senses = par.find_all(is_sense_in_rus)
    translations, examples = [], []
    if not senses:
        translation = ' '.join(par.stripped_strings)
        res.append(dict(sense='', translation=translation, example=''))

    for sense in senses:
        translation = sense.next_sibling
        example_parts = []
        for sibling in translation.next_siblings:
            if sibling.name == 'em':
                if sibling not in senses:
                    example_parts.append(sibling.string.strip().split(';')[0])
                else:
                    break
            elif sibling.name == 'strong':
                if sibling.string in [
                  'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']:
                    continue
                else:
                    example_parts.append(sibling.string.strip() + transl_sep)
            elif isinstance(sibling, NavigableString):
                # `; ` is likelier to be translation end, but it's irregular
                #   and `;` could capture more so..
                example_parts.append(sibling.strip().split(';')[0])
            elif sibling.name == 'br':
                continue
            else:
                print(f"Unexpected tag in `p`: {sibling}")
        translation = translation.strip().strip(';')
        example = ' '.join(example_parts)

        translations.append(translation)
        examples.append(example)

    names = ('sense', 'translation', 'example')
    lists = (senses, translations, examples)
    for i in range(len(senses)):
        res.append(dict(sense=senses[i].string, translation=translations[i],
                        example=examples[i]))
    return res


def parse_translation(translation: Translation):
    pars = translation.get('translation').find_all('p', recursive=False)
    ru_gram_info, ru_gram_info2 = '', ''

    for par in pars:
        children_tags = par.contents
        res = rec_parse_translation(par)
        print(f"Parsed paragraph\n\t`{par}`")
        print(res)

        # for children_tag in children_tags:
        #     print(children_tag.name, children_tag)

        # if children_tags[0].name == 'em':
        #     ru_gram_info = children_tags[0]
        # else:
        #     print(f"First element in `<p>` isn't `<em>`: {children_tags[0]}")


res = get_word_transl('в')
print(res)

for res_ in res['translations']:
    parse_translation(res_)

# with open('translations.csv', 'w', newline='') as csvout:
#     fieldnames = ['lexeme', 'word_phrase_rus', 'transl', 'comment', 'link']:
#     writer = csv.DictWriter(csvout, fieldnames=fieldnames)