from gensim.models import FastText as FT
from gensim.utils import tokenize
from gensim import utils
import sys
import os


class EmbedText:
    def __init__(self):
        self.model = None

    @staticmethod
    def iter_doc(file_path):
        with utils.open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(line))

    def init_model(self, file_path):

        corpus = EmbedText.iter_doc(file_path)

        model = FT(size=150, window=5, min_count=1, workers=4)
        model.build_vocab(sentences=corpus)
        model.train(corpus_file=file_path, epochs=model.epochs,
                    total_examples=model.corpus_count,
                    total_words=model.corpus_total_words, model='skipgram')
        self.model = model

    def update_model(self, file_path):

        corpus = EmbedText.iter_doc(file_path)

        self.model.build_vocab(
            sentences=corpus, update=True)
        self.model.train(corpus_file=file_path, epochs=self.model.epochs,
                         total_examples=self.model.corpus_count,
                         total_words=self.model.corpus_total_words,
                         model='skipgram')

    def save_model(self, save_path):

        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.model.save(save_path)


def main():

    model = EmbedText()
    model.init_model(sys.argv[1])

    print(len(model.model.wv.vocab))


if __name__ == '__main__':
    main()
