from scipy.stats import bernoulli, uniform
import numpy as np


class BernoulliDecisionMaker:
    """
    This decision maker is oblivious to the stimulus, it is a pure Bernoulli r.v.
    """

    def __init__(self, name, p=None, bias=0):
        """
        :param p: sucess probability of r.v.
        """
        self.name = name
        self.p = p
        self.bias = bias
        self.decisions = None
        self.perceived_stimulus = None
        self.accuracy = None

    def __str__(self):
        return f"name: {self.name}\n" \
               f"bias: {self.bias}\n" \
               f"accuracy: {self.accuracy}\n"

    def compute_accuracy(self, correct):
        if self.decisions is None:
            f"No decisions have been made"
        else:
            self.accuracy = np.sum(self.decisions == correct) / len(correct)

    def decide(self, n):
        """
        makes n decisions, independent of stimulus, and using constant success probability
        :param n: number of independent samples to draw
        """
        self.decisions = bernoulli.rvs(self.p, size=n)

    def present_stimulus(self, s):
        """
        modifies stimulus s as if its perception were biased towards the extremes {0,1}
        :param s: array of stimulus values between 0 and 1
        :param bias: number between 0 and 1
        """
        if self.bias == 0:
            self.perceived_stimulus = s
        else:
            def apply_bias(ss):
                gap = (1 - self.bias) / 2
                if ss < 0.5:
                    return uniform.rvs(loc=0, scale=gap, size=1)
                else:
                    return uniform.rvs(loc=0.5 + self.bias / 2, scale=gap, size=1)

            self.perceived_stimulus = np.array([apply_bias(sss) for sss in s])

    @classmethod
    def bernoulli_decide(cls, probabilities):
        """
        makes N Bernoulli decisions with specified success probabilities, where N is the length of 'probabilities'
        :param probabilities: array of success probabilities
        """
        return np.array([bernoulli.rvs(pp) for pp in probabilities])


class StimulusUnitInterval:
    sequence = None

    def __init__(self, size=10):
        self.size = size
        self.sequence = None
        self.correct = None

    def __str__(self):
        return f"stim 1-10: {str([round(x,2) for x in self.sequence[0:10]])}\n" \
               f"correct 1-10: {str(self.correct[0:10])}"

    def generate_sequence(self, mode):
        if mode == 'random':
            self.sequence = self.generate_random_sequence(self.size)
        else:
            raise ValueError(f"only supported mode is 'random' for now")
        self.correct = self.sequence > 0.5

    @classmethod
    def generate_random_sequence(cls, n):
        return uniform.rvs(size=n)
