# This class instantiates fixations.


class Fixation:
    def __init__(self, number, pos, word):
        self.dur = 0  # duration in ms
        self.number = number  # fixation nr. (1-N)
        self.pos = pos  # cumulative within-sentence character position of fixation
        self.word = word  # cumulative within-sentence word nr. of fixation
