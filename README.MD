# Sakha corpus

This hosts code for the future sakha language corpus

**Пометкой `ДЕВ` отмечаю приватные вещи, которые потом надо удалить**

Contents
========
 * [Installation](#installation)
 * [Usage](#usage)
 * [ДЕВ файлы](#дев-файлы)
 * [ДЕВ планы](#дев-планы)
 * [Contributing](#contributing)
 * [License](#license)
 
## Installation
This was developed with python `3.7.1`, then `3.9.10`.
Most code should be compatible with the former, except for `spacy`

1. Use venv (create it and switch to it [as described here](https://docs.python.org/3/tutorial/venv.html))
   Windows `cmd`:
   ```bash
    python -m venv venv
    venv\Scripts\activate.bat 
    ```
   Windows `PowerShell`:
   ```bash
    python -m venv venv
    venv\Scripts\Activate.ps1
    ```
   Linux:
   ```bash
    python -m venv venv
    source venv/Scripts/activate
    ```
2. Use [pip](https://pip.pypa.io/en/stable/) to install the dependencies.
    This is needed only for the pipeline (`requirements_pipeline.txt`),
    which uses `nltk.tokenizer.Punkt` for sentences and `spacy` for russian analysis
    and general tokenization.
    ```bash
    pip install -r requirements_pipeline.txt
    ```
 
## Usage

Pipeline (ДЕВ: половина из этого ещё не работает):
* Файлы (названия не отражают сути, поменяю)
  * `data_models.py` &mdash; API для итерации и индексирования данных
    (пока только edersaas за 05.12.2021),
    в будущем все данные таким или похожим образом
  * `tidy_string.py` &mdash; разные группы преобразований с целью нормализации.  
  Их надо дополнять и уметь параметризовывать
  (наверное что-то совсем базовое нужно будет всегда?)
  * `profile_characters.py` &mdash; посмотреть, что неловимое общими правилами
  отсутствует в небольшом алфавите
  * `pipeline.py` &mdash; общий пайплайн обработки текста.
  Пока мало кода, даже не успел попробовать, что придумал
    (`spacy` для токенизации)
  * `train_sentence_tok.py` &mdash; тренировка токенизатора предложений (первая,
  без препроцессинга. Её результат - файл `sakha_edersaas_0.pickle`, см. ниже)
* `nltk.tokenizer.Punkt` for sentence parsing
* [`spacy`](https://spacy.io) for russian analysis and general tokenization.  (пока не до конца)
* [`foma`](https://fomafst.github.io/) for rule-based sakha words parsing (**NB:** not yet in the pipeline)
    + `/foma/rules2foma.py` converts csv files to _foma_ source files (`.lexc` and ` .foma`) 

ДЕВ (наверное потом надо убрать же ?!) парсинг сайта словаря [sakhatyla.ru](http://sakhatyla.ru):
* `parse.py`, `lexemes.py`
 
 
 
## ДЕВ: файлы

[Основная папка](https://drive.google.com/drive/folders/1790t5iNs0BSfC4eqe_Ca5yJX4GHOpQ5z?usp=sharing),
в ней пока что
 * гугл-таблица с морфемами, правилами и тестами
 * в `/texts` архив по новостному сайту `edersaas.7z` ([edersaas.ru](https://edersaas.ru))  
    (их, конечно, надо будет переделывать, там то ли из лишнего тэга берётся текст,
    то ли наоборот не берётся.  
    + может понадобиться базовая нормализация текста перед сохранением)
 * модель токенизатора предложений `sakha_edersaas_0.pickle`
 (созданная по тексту без преобразований => не такая хорошая)  
 (**NB:** поначалу ошибся с ней и сохранил тренировочный класс, это не совсем модель.
 После разогуречивания надо ещё `.get_params()`, код может уже не отражать этого, простите...
 хотя в `pipeline.py` вроде отражено)
 * файлы фомы, если вдруг будет желание запустить 
 * общий список слов на сайте [sakhatyla.ru](http://sakhatyla.ru) (ru/sah/eng)
 * страницы сайта [sakhatyla.ru](http://sakhatyla.ru) (`sakhatyla.7z`) 
 * якутские слова с сайта, проанализированные (ещё надо улучшать) `sakhatyla.csv`

## ДЕВ: планы
* обсудить большой вопрос:  
  может быть можно научиться понимать, когда текст полностью русский,
  а когда смешанный и размечать в последних и русское и якутское?  
  Это в основном про нелитературные источники, типа соцсетей, форумов, которых я не 
  так много видел, но мультиязычность здесь была бы супер.  

* обсудить содержание и порядок применения символьных юникодовских преобразований (нормализацию)
    * тексты сохранять до всяких преобразований?
    или с частичными крайне общими типа удаления разновидных пробелов,
    `всевозможные_кавычки -> "`?
    
    * может эмодзи можно оставить? 🥺  
        было бы круто их иметь в исходном тексте.
        Может помочь кому-то в будущем для какого-то продвинутого НЛП.
        Сходу не пойму, но не мешает ли оставление выравниванию разметки и исходных форм?  
        ещё есть же очень сложные эмодзи типа [🤦‍♂️ Man Facepalming](https://emojipedia.org/man-facepalming/),
        которые [из кучи скалярных code points состоят](https://hsivonen.fi/string-length/).  
        Как их вообще может токенизировать вместе [udpipe](https://lindat.mff.cuni.cz/services/udpipe/)?..  
        Представляю разве что эвристики, что брать всю группу символов от нормального словарного
        до технического типа [`\ufeof`](https://unicode-table.com/en/FE0F/). Но
        простые текстовые эмодзи и без него бывают.
            
* закончить наконец правиловый парсер...
    * (**NB:** нужно проверить) кажется, можно иметь даже выравнивание морфема-значение 
    * дописать собственно морфологию и морфонологию
    * (**NB: на обсуждение**) сделать какие-то **тесты**     
      вариант: залить на какой-нибудь [`pythonanywhere.com`](https://pythonanywhere.com)
      и попросить людей из экспы попроверять там как размечается.
      Разрешать вводить, скажем, предложение.
        * установить фому туда
        * фронт и бэк для сбора фидбэка / указаний на ошибки
        * автоматический залив фидбэка в гугл-щитс

* **токенизация предложений** после тренировки не такая плохая, но надо улучшать  
    *    добавить код, который бы считал частоту паттерна типа `'\w+.(\w.)+(\s|[,:;]'`  
        и добавлял бы в конце тренировочные контексты для парсера
        ([вроде так можно суметь](https://stackoverflow.com/questions/64776093/how-to-get-nltk-punkt-sentence-tokenizer-to-recognize-abbreviations-that-occur-i),
        но непонятно, как это делать когда много. Добавлять найденные частые "аббревиатуры"
        в один и тот же контекст?)
    
* обсудить токенизацию
    * что-то типа [`spacy`](https://spacy.io/usage/linguistic-features#tokenization)?  
    там последовательно от штук разделённых пробелами отщепляются
    возможные префиксы и суффиксы (знаки препинания), также слово если в 
    специальном специальном словаре, разворачивается как указано
    **NB:** всё вручную, необучаемо
    
    * ещё, насколько либеральное правило для ссылок?  
    видел [кучу](https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url),
    [кучу 2](https://stackoverflow.com/questions/6718633/python-regular-expression-again-match-url) всяких,
    [на одно](https://gist.github.com/gruber/249502) часто ссылаются (плюс там в теме исправления).
    
* ещё раз обсудить как надо хранить файлы..
    * сейчас сохранены `.json` со всякими метаданными.  
    При этом можно итерироваться по `train_paragraphs.py.` `EdersaasJSON`/`Dataset`
    и получать либо просто куски текстов `return_meta = False`
    или куски текстов + некоторые метаданные 

* добить парсинг статей с [sakhatyla.ru](http://sakhatyla.ru)..  
    неплох, но надо сделать лучше, чтобы был хороший словарь с частеречными данными
    * слова в русском переводе (для целей определить и исходную часть речи)
    лучше парсить не пайморфи..  
    лучше парсить целое предложение перевода udpipe'ом, он справляется и с разделителем
    типа `|` и в целом круче. Можно брать часть речи синтаксической вершины перевода
    

* добавлять ресурсы, скачивать с них текст...

* ...


## Contributing
N/A

## License
N/A