from pathlib import Path
import pickle

from tqdm import tqdm
from nltk.tokenize.punkt import PunktTrainer

from data_models import Dataset, EdersaasJSON


def collect_tok_testdata(textpiece):
    # TODO: use regex here to detect possible abbreviations
    #   to later feed them to tokenizer if it doesn't detect them
    raise NotImplementedError


def train(dataset, punkt_trainer: PunktTrainer):
    for textpiece in tqdm(dataset):
        # TODO: this needs some kind of preprocessing
        # TODO: collect_tok_testdata(textpiece)
        punkt_trainer.train(textpiece)


def main():
    save_dir = './punkt_models'
    tokenizer_filename = 'sakha_edersaas.pickle'

    dataset = Dataset()
    punkt_trainer = PunktTrainer()

    train(dataset, punkt_trainer)

    # TODO: needed not here but perhaps before usage?
    punkt_trainer.finalize_training()
    params = punkt_trainer.get_params()

    with open(Path(save_dir) / Path(tokenizer_filename), 'wb') as f:
        pickle.dump(params, f)

    print(f'saved tokenizer trainer at {tokenizer_filename}')

    print(f"tokenizer abbreviations:\n{punkt_trainer.get_params().abbrev_types}")
    print(f"tokenizer collocations:\n{punkt_trainer.get_params().collocations}")
    print(f"tokenizer sentence starters:\n{punkt_trainer.get_params().sent_starters}")
    # print(f"tokenizer ortho_context:\n{punkt_trainer.get_params().ortho_context}")


if __name__ == "__main__":
    main()
