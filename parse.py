from typing import List, Dict, Union, Tuple
import csv
import re
import copy
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString, Tag

from utils import HEADERS, write_to_csv

sakhatyla_site = "https://sakhatyla.ru/"
sakha_link = sakhatyla_site + "translate?q="

words = ("и", "в")

session = requests.session()
session.headers.update(HEADERS)

# def get_similar
#     TODO

Translation = Dict[str, Union[str, Tag]]


def get_word_page(word: str) -> Tuple[str]:
    word_link = "translate?q=" + requests.utils.quote(word)
    link = sakhatyla_site + word_link
    response = session.get(link)

    if response.status_code != 200:
        raise ValueError("Something went wrong while getting result")

    return response.text, link, word_link, word


def save_page(page: str, word: str, folder="sakhatyla.ru"):
    path = f"{folder}/{word}"
    with open(path, 'w', encoding='utf-8') as fout:
        fout.write(page)


def collect_lexical_entries(
        word: str, link: str, path: Path = None
) -> Dict[str, Union[str, List[Translation]]]:
    if not path:
        page, _, _, _ = get_word_page(word)
    else:
        with open(path, 'r', encoding = 'utf-8') as f:
            page = f.read()

    soup = BeautifulSoup(page, 'lxml')

    # достать тэг `<h2>Русский → Якутский</h2>` и смотреть его сестёр дальше
    #   пока не попадём на очередной <h2> (или `<p>ещё переводы</p>`)
    translation_tags: List[Translation] = []
    comment = ''

    # header_soup = soup.find_all('h2', string="Русский → Якутский")
    directions = [
        dict(source_l="ru", targ_l="sa", header="Русский → Якутский"),
        dict(source_l="sa", targ_l="ru", header="Якутский → Русский"),
        dict(source_l="sa", targ_l="en", header="Якутский → Английский")
    ]
    existing_directions = []
    
    for direction_dict in directions:
        direction = direction_dict['header']
        header_soup = soup.find('h2', string=direction)
        if not header_soup:
            print(f"word: {word}, no `{direction}`")
            continue
        direction_dict.update(dict(header_soup=header_soup))
        existing_directions.append(direction_dict)

    # ru_sa_header = soup.find('h2', string="Русский → Якутский")
    # sa_ru_header = soup.find('h2', string="Якутский → Русский")
    # sa_en_header = soup.find('h2', string="Якутский → Английский")

    # if not header_soup:
    #     comment = f"нет русского перевода. "
    #                # f"есть переводы `{','.join(header_soups)}`")
    #     print(comment)
    #     res = dict(word=word, translations=[], link=link, comment=comment)
    #     return res
    
    for direction_dict in existing_directions:
        header_soup = direction_dict['header_soup']
        
        for tag in header_soup.next_siblings:
            if isinstance(tag, NavigableString):
                continue
            elif tag.name in ('h2', 'p', 'hr'):
                print(f"encountered `{tag.name}`, ending loop (`{tag.string}`)")
                break
            else:
                # TODO: сохранять все тэги или сразу убирать словосочетания?
                # TODO: filtering of words not equal to query should be done here
                source_word_or_phrase = tag.h3.string.strip()
                translation = copy.copy(tag.find('div', class_='article-text'))
    
                # add sentinel value to delimit
                # TODO: may not be needed with copies?
    
                lexical_category_tag = tag.find('div', class_='article-category')
                if lexical_category_tag:
                    lexical_category = lexical_category_tag.string.split(': ')[1]

                entry_dict = {k: v for k, v in direction_dict.items()
                              if k not in ('header', 'header_soup')}
                entry_dict.update(dict(source=source_word_or_phrase,
                    translation=translation, lexical_category=lexical_category))
                translation_tags.append(entry_dict)
    
    print(f'Successfully parsed `{sakha_link+word}`')
    res = dict(word=word, translations=translation_tags, link=link, comment=comment)
    
    return res


def rec_parse_translation(par, transl_sep=' : ', examples_sep='|'):
    res = []

    def is_sense_in_rus(tag):
        return tag.name == 'em' and not ';' in tag.string

    senses = par.find_all(is_sense_in_rus)
    translations, examples = [], []
    if not senses:
        translation = ' '.join(par.stripped_strings)
        res.append(dict(sense='', translation=translation, example=''))

    prev_sibling_type = ''
    # пройти по найденным русским лексемам / фразам и получить для них
    #   перевод и примеры
    for sense in senses:
        translation = sense.next_sibling
        example_parts = []

        for sibling in translation.next_siblings:
            if sibling.name == 'em':
                if sibling not in senses:
                    # бывает, что пример (или перевод? TODO понять!)
                    #   даётся сразу рядом тоже в `<em>`, `в`: https://sakhatyla.ru/translate?q=%D0%B2
                    example_parts.append(
                        sibling.string.strip().split(';')[0] + examples_sep)
                    break
                else:
                    break
            elif sibling.name == 'strong':
                if sibling.string in [
                  'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']:
                    # номер значения (ср. `в`: https://sakhatyla.ru/translate?q=%D0%B2)
                    continue
                else:
                    # добавим перед этой русской фразой разделитель переводов,
                    #   на случай если это не первая переводимая фраза
                    #   позже уберём лишнее strip
                    example_parts.append(examples_sep + ' '
                                         + sibling.string.strip() + transl_sep)
            elif isinstance(sibling, NavigableString):
                # `; ` is likelier to be translation end, but it's irregular
                #   and `;` could capture more so..
                example_parts.append(sibling.strip().split(';')[0])
            elif sibling.name == 'br':
                continue
            else:
                print(f"Unexpected tag in `p`: {sibling}")
        translation = translation.strip().strip(';')
        # либо слеплять пробелом пример с уже расставленными знаками
        #   (` : ` (`transl_sep`) между русским и якутским,
        #   `|` (examples_sep) между примерами
        #   и тогда слеплять пробелом, либо
        example = ' '.join(example_parts).strip(examples_sep)

        translations.append(translation)
        examples.append(example)

    names = ('sense', 'translation', 'example')
    lists = (senses, translations, examples)
    for i in range(len(senses)):
        res.append(dict(sense=senses[i].string, translation=translations[i],
                        example=examples[i]))

    # чаще всего самое первое выделенное русское слово - грам. сведения о русском
    #   переводимом слове, чтобы лучше понять перевод. TODO: их можно хранить отдельно
    # TODO: удалять совсем те вхождения где нет примеров и перевод типа "1."
    # first_sense = res[0]['sense']

    return res


def parse_translation(translation: Translation) -> List[Dict[str, str]]:
    # найти все разделы перевода. обычно он один, но бывает несколько
    #   (`к`: https://sakhatyla.ru/translate?q=%D0%BA)
    pars = translation.pop('translation').find_all('p', recursive=False)
    rus = translation.get('rus')
    lexical_category = translation.get('lexical_category')

    res = []
    for par in pars:
        children_tags = par.contents
        par_res = rec_parse_translation(par)
        res.extend(par_res)
        print(f"Parsed paragraph\n\t`{par}`")
        print(par_res)

    for res_ in res:
        res_.update(translation)

    return res


def get_word_data(word: str):
    general_info_translations = collect_lexical_entries(word)
    print(general_info_translations)

    entries = []
    for translations in general_info_translations.pop('translations'):
        entry_data = parse_translation(translations)
        entries.extend(entry_data)

    # добавить общую информацию к каждому слову для записи позже в csv
    for entry_dict in entries:
        entry_dict.update(general_info_translations)

    return entries

# res = collect_lexical_entries('в')
# print(res)
#
# results = []
# for translations in res['translations']:
#     entry_data = parse_translation(translations)

# N = 2
# with open('ru_words.txt', 'r', encoding='utf-8') as f:
#     words = [next(f).strip() for i in range(N)]


# with open('ru_words.txt', 'r', encoding='utf-8') as f:
#     words = f.read().split('\n')


# for word in words:
#     page, link, word_link, word = get_word_page(word)
#     save_page(page, word)

#
# entries = []
# for word in words:
#     try:
#         entry = get_word_data(word)
#         entries.extend(entry)
#     except (AttributeError, ValueError) as e:
#         print(e)
#
# # entries = get_word_data('в')
# # entries += get_word_data('к')
# print(words, entries)
# write_to_csv(entries)


if __name__ == "__main__":
    collect_lexical_entries()
    