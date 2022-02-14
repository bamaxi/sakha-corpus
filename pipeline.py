import pickle

from tqdm import tqdm
from nltk.tokenize.punkt import PunktSentenceTokenizer
import spacy

from tidy_string import make_translation
from data_models import Dataset, EdersaasJSON

from LangAnalysis.lang_analysis import Language_Determiner


sent_tokenizer_params_pickled = 'sakha_edersaas_0.pickle'


def unpickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


class Pipeline:
    def __init__(
        self, sent_tokenizer_params, tidying_translation=make_translation(),

    ):
        self.translation = tidying_translation

        self.sent_tokenizer_params = sent_tokenizer_params
        self.sent_tokenizer = PunktSentenceTokenizer(train_text=sent_tokenizer_params)

    def apply(self, text):
        # tidier_text = text.translate(self.translation)
        tidier_text = text
        return self.sent_tokenizer.tokenize(tidier_text)


sent_tokenizer_params = unpickle(sent_tokenizer_params_pickled).get_params()

pipeline = Pipeline(sent_tokenizer_params)

LD = Language_Determiner()
LD.load_categories()

dataset = Dataset()

print(sent_tokenizer_params)

for i, paragraph in enumerate(dataset):
    langs = LD.compare_by_rank(paragraph)
    print(langs)
    print(paragraph)


    print(pipeline.apply(paragraph))
    if i > 50:
        break