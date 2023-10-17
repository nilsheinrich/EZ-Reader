import numpy as np


def custom_raise():
    raise ValueError()


class word():
    def __init__(self, line, pos0, pos1, wn):
        items = line.split()  # split the word line into the 4 columns
        self.frequency = float(items[0])
        self.nlog_frequency = np.log(self.frequency)
        self.frequency_class = np.log10(self.frequency - 0.5)  # ????
        self.length = int(items[1])
        self.cloze = float(items[2])
        self.string = items[3]
        self.pos0 = pos0
        self.pos1 = pos1
        self.posN = self.pos1 + self.length
        self.OVP = self.pos1 + self.length * 0.5
        self.wn = wn

    def __repr__(self):
        return self.string


class sentence():
    def __init__(self, block):
        self.words = []
        pos0 = 0  # this indexes the left border of the first letter in the sentence or the left border of a preceding whitespace
        pos1 = 0  # this indexes the left border of the first letter of a word
        # this might be different from reichle, he starts counting from a space field to the left of the first word
        # however, the distances are correct (just shifted by one)
        for wn, line in enumerate(block.strip().split("\n"), 1):  # split the block into lines (each word is one line)
            new_word = word(line, pos0, pos1, wn)  # call to class "word" (method "word" is assigned after init)
            self.words.append(new_word)
            pos0 = new_word.posN  # words are separated with a white space
            pos1 = pos0 + 1

    def word(self, word_number):
        if type(word_number) == str:
            if word_number == "last":
                return self.words[-1]
            elif word_number == "first":
                return self.words[0]
            else:
                raise
        elif type(word_number) == int and self.index_ok(word_number):
            return self.words[word_number - 1]
        elif type(word_number) == list:
            return [self.words[wn - 1] if self.index_ok(wn) else custom_raise() for wn in word_number]
        else:
            # print(f"word_number: {word_number}; type: {type(word_number)}; self.index_ok: {self.index_ok(word_number)}")
            raise TypeError()

    def index_ok(self, ix):
        return True if (0 < ix <= len(self)) else False

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return " ".join(w.string for w in self.words)


class corpus():
    def __init__(self, path_to_corpus):
        self.path = path_to_corpus
        self.sentences = []
        with open(self.path, "r") as f:
            for block in f.read().split("@"):  # split file contents at the "@" into blocks
                if len(block.strip()) != 0:  # escape the last block, as it only contains whitespace
                    new_sentence = sentence(block)  # call to class "sentence" (method "sentence" is assigned after init)
                    self.sentences.append(new_sentence)
        self.N_sentences = len(self.sentences)

    def sentence(self, sentence_number):
        if type(sentence_number) == int and self.index_ok(sentence_number):
            return self.sentences[sentence_number - 1]
        elif type(sentence_number) == list:
            return [self.sentences[sn - 1] if self.index_ok(sn) else custom_raise() for sn in sentence_number]
        else:
            raise TypeError()

    def index_ok(self, ix):
        return True if (0 < ix <= self.N_sentences) else False

    def __repr__(self):
        return f"Corpus file: {self.path}"


if __name__ == "__main__":  # some usecases and info on the corpus

    c = corpus("../Data/SRC98Corpus.txt")
    asd = c.sentence(1)
    print(len(asd))
    # print(c.sentence(1))
    # N_words = 0
    # for s in c.sentences:
    #     N_words += len(s.words)

    # for sn_idx, sentence in enumerate(c.sentences, 1):
    #     print(f"Sentence {sn_idx} contains {len(sentence.words)} words")
    #     print(sentence)
    #     if sn_idx == 1:
    #         break

    # print("\nComplete sentence 1:")
    # print(c.sentence(1))
    # print("\nWords in sentence 1:")
    # print(c.sentence(1).words)
    # print("\nWord 4 of sentence 1")
    # print(c.sentence(1).word(4))
    # print("\nFrequency of word 4 of sentence 1")
    # print(c.sentence(1).word(4).frequency)

    # for w_ix, w in enumerate(sentence.words, 1):
    #     print(f"word {w_ix} starts at {w.pos0}: {w.pos1} {w.posN} {w.OVP}")

    # print(sentence.word("last"))

    # # test lists as input
    # print(c.sentence([2, 1]))
    # print(c.sentence(1).word([3, 2, 1]))
