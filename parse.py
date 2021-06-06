import requests
from bs4 import BeautifulSoup

from utils import HEADERS

sakha_link = "https://sakhatyla.ru/" + "translate?q="

words = ("и", "в")


# def get_
#     TODO

def get_word_transl(link):
    session = requests.session()
    response = session.get(link, headers=HEADERS)

    if response.status_code == 200:
        print(f"Successfully loaded `{requests.utils.unquote(link)}`")
    else:
        raise ValueError("Something went wrong while getting result")

    words = []

    soup = BeautifulSoup(response.text, 'html.parser')
    for list_element in soup.find(
            'div', id="mw-content-text").find('ol').find_all('li'):
        word = list_element.string
        if word is not None:
            words.append(word)

    print(f'Successfully parsed `{requests.utils.unquote(link)}`')
    return words