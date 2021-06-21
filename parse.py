from typing import List, Dict, Union
import csv
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString, Tag

from utils import HEADERS, write_to_csv

sakha_link = "https://sakhatyla.ru/" + "translate?q="

words = ("и", "в")

# def get_similar
#     TODO

Translation = Dict[str, Union[str, Tag]]


def get_word_transl(word: str) -> Dict[str, Union[str, List[Translation]]]:
    session = requests.session()
    link = sakha_link + requests.utils.quote(word)
    response = session.get(link, headers=HEADERS)

    if response.status_code != 200:
        raise ValueError("Something went wrong while getting result")

    soup = BeautifulSoup(response.text, 'lxml')
    # NEW: delete linebreaks
    for linebreak in soup.find_all('br'):
        linebreak.extract()

    # достать тэг `<h2>Русский → Якутский</h2>` и смотреть его сестёр дальше
    #   пока не попадём на очередной <h2> (или `<p>ещё переводы</p>`)
    translation_tags: List[Translation] = []
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
                lexical_category = tag.find(
                    'div', class_='article-category').string.split(': ')[1]
            except (AttributeError) as e:
                lexical_category = ''

            word_or_phrase_res = dict(
                rus=rus_word_or_phrase, translation=translation,
                lexical_category=lexical_category)
            translation_tags.append(word_or_phrase_res)

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
    general_info_translations = get_word_transl(word)
    print(general_info_translations)

    entries = []
    for translations in general_info_translations.pop('translations'):
        entry_data = parse_translation(translations)
        entries.extend(entry_data)

    # добавить общую информацию к каждому слову для записи позже в csv
    for entry_dict in entries:
        entry_dict.update(general_info_translations)

    return entries

# res = get_word_transl('в')
# print(res)
#
# results = []
# for translations in res['translations']:
#     entry_data = parse_translation(translations)

N = 2
with open('ru_words.txt', 'r', encoding='utf-8') as f:
    words = [next(f).strip() for i in range(N)]

entries = []
for word in words:
    try:
        entry = get_word_data(word)
        entries.extend(entry)
    except (AttributeError, ValueError) as e:
        print(e)

# entries = get_word_data('в')
# entries += get_word_data('к')
print(words, entries)
write_to_csv(entries)