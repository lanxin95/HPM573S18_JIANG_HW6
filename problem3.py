import numpy as np
import scr.FigureSupport as figureLibrary
import scr.SamplePathClass as SamplePathSupport
import scr.StatisticalClasses as Stat
import scipy.stats as scipy


class Game(object):
    def __init__(self, id, prob_head):
        self._id = id
        self._rnd = np.random
        self._probHead = prob_head  # probability of flipping a head
        self._countWins = 0  # number of wins, set to 0 to begin

    def simulate(self, n_of_flips):

        count_tails = 0  # number of consecutive tails so far, set to 0 to begin

        # flip the coin 20 times
        for i in range(n_of_flips):

            # in the case of flipping a heads
            if self._rnd.random_sample() < self._probHead:
                if count_tails >= 2:  # if the series is ..., T, T, H
                    self._countWins += 1  # increase the number of wins by 1
                count_tails = 0  # the tails counter needs to be reset to 0 because a heads was flipped

            # in the case of flipping a tails
            else:
                count_tails += 1  # increase tails count by one

    def get_reward(self):
        # calculate the reward from playing a single game
        return 100*self._countWins - 250


class SetOfGames:
    def __init__(self, prob_head, n_games):
        self._gameRewards = [] # create an empty list where rewards will be stored
        self._ngames=n_games
        # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())
            # after simulating all cohorts
            # summary statistics of mean survival time
        self._sumStat_Rewards = Stat.SummaryStat('Rewards', self._gameRewards)

    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return self._sumStat_Rewards.get_mean()

    def get_reward_list(self):
        """ returns all the rewards from all game to later be used for creation of histogram """
        return self._gameRewards

    def get_max(self):
        """ returns maximum reward"""
        return max(self._gameRewards)

    def get_min(self):
        """ returns minimum reward"""
        return min(self._gameRewards)

    def get_probability_loss(self):
        """ returns the probability of a loss """
        count_loss = 0
        for value in self._gameRewards:
            if value < 0:
                count_loss += 1
        self._prob_loss=count_loss / len(self._gameRewards)
        return self._prob_loss

    def get_mean_tCI(self, alpha):
        """ :returns: the t-based confidence interval of the mean reward"""
        return self._sumStat_Rewards.get_t_CI(alpha)

    def get_portion_tCI(self,alpha):
        """:returns: the t-based confidence interval of the loss probability"""
        CIP = [0, 0]
        SE = pow(self._prob_loss* (1 - self._prob_loss), 0.5) / pow(self._ngames, 0.5)
        half_length = scipy.t.ppf(1 - alpha / 2, self._ngames) * SE
        CIP[0] = self._prob_loss - half_length
        CIP[1] = self._prob_loss + half_length
        return CIP

    def get_mean_PI(self, alpha):
        """ :returns: the prediction interval of the mean survival time"""
        return self._sumStat_Rewards.get_PI(alpha)

class MultiCohort:
    """ simulates multiple cohorts with different parameters """
    def __init__(self, ids, pop_sizes, prob_head):
        """
        :param ids: a list of ids for cohorts to simulate
        :param pop_sizes: a list of population sizes of cohorts to simulate
        :param prob_head: a list of the probabilities for head
        """
        self._ids = ids
        self._popSizes = pop_sizes
        self._probhead = prob_head

        self._rewards = []      # two dimensional list of player rewards from each simulated cohort
        self._meanreward = []   # list of mean rewards for each simulated cohort
        self._lossprob=[]       # list of probability of losing the game
        self._sumStat_meanreward = None
        self._sumStat_lossprob = None

    def simulate(self, n_time_steps):
        """ simulates all cohorts """

        for i in range(len(self._ids)):
            # create a cohort
            game = SetOfGames(self._probhead[i], self._popSizes[i])
            # store all rewards from this cohort
            self._rewards.append(game.get_reward_list())
            # store average rewards time for this cohort
            self._meanreward.append(game.get_ave_reward())
            # store estimate of the probability of losing the game
            self._lossprob.append(game.get_probability_loss())

        # after simulating all cohorts
        # summary statistics of mean survival time
        self._sumStat_meanreward = Stat.SummaryStat('Mean reward', self._meanreward)
        self._sumStat_gamelossprob = Stat.SummaryStat('Mean loss prob', self._lossprob)
    def get_cohort_mean_reward(self, cohort_index):
        """ returns the mean reward of an specified cohort
        :param cohort_index: integer over [0, 1, ...] corresponding to the 1st, 2ndm ... simulated cohort
        """
        return self._meanreward[cohort_index]

    def get_all_mean_reward(self):
        " :returns a list of mean reward for all simulated cohorts"
        return self._meanreward

    def get_overall_mean_reward(self):
        """ :returns the overall mean reward"""
        return self._sumStat_meanreward.get_mean()

    def get_cohort_tCI_mean_reward(self, alpha):
        return self._sumStat_meanreward.get_t_CI(alpha)

    def get_cohort_lossprob(self, cohort_index):
        """ returns the probability of loss of an specified cohort
        :param cohort_index: integer over [0, 1, ...] corresponding to the 1st, 2ndm ... simulated cohort
        """
        return self._lossprob[cohort_index]

    def get_all_lossprob(self):
        " :returns a list of probability of loss"
        return self._lossprob

    def get_overall_lossprob(self):
        """ :returns the overall mean reward"""
        return self._sumStat_gamelossprob.get_mean()

    def get_tCI_lossprob(self, alpha):
        return self._sumStat_gamelossprob.get_t_CI(alpha)

    def get_cohort_PI_reward(self, cohort_index, alpha):
        """ :returns: the prediction interval of the survival time for a specified cohort
        :param cohort_index: integer over [0, 1, ...] corresponding to the 1st, 2ndm ... simulated cohort
        :param alpha: significance level
        """
        st = Stat.SummaryStat('', self._rewards[cohort_index])
        return st.get_PI(alpha)

    def get_PI_mean_reward(self, alpha):
        """ :returns: the prediction interval of the mean rewards"""
        return self._sumStat_meanreward.get_PI(alpha)

    def get_PI_loss_prob(self, alpha):
        """ :returns: the confidence interval of the loss probabiltiy"""
        return self._sumStat_gamelossprob.get_PI(alpha)

#3.1. The casino owner who gets to play this game many times. -- Steady Model
# run multiple cohort
NUM=1000
CasinoOwner = MultiCohort(
    ids=range(NUM),   # [0, 1, 2 ..., NUM_SIM_COHORTS-1]
    pop_sizes=[1000] * NUM,  # [REAL_POP_SIZE, REAL_POP_SIZE, ..., REAL_POP_SIZE]
    prob_head=[0.5]*NUM  # [p, p, ....]
)
# simulate all cohorts
CasinoOwner.simulate(NUM)

print("The average expected reward for the casino owner is:", CasinoOwner.get_overall_mean_reward())
print('95% Confidence Interval', CasinoOwner.get_cohort_tCI_mean_reward(0.05))
print('Interpretation: if the casino owner simulate the games for a lot of times,\n\
and get a confidence interval in each time, 95% of these intervals will cover true mean')

#3.2   A gambler who gets to play this game only 10 times.--Transient Model
NUM2=10
Gambler = MultiCohort(
    ids=range(NUM2),   # [0, 1, 2 ..., NUM_SIM_COHORTS-1]
    pop_sizes=[1000] * NUM2,  # [REAL_POP_SIZE, REAL_POP_SIZE, ..., REAL_POP_SIZE]
    prob_head=[0.5]*NUM2  # [p, p, ....]
)
# simulate all cohorts
Gambler.simulate(NUM2)
print("The average expected reward for the Gambler is:", Gambler.get_overall_mean_reward())
# print projection interval
print('95% projection interval',Gambler.get_PI_mean_reward(0.05))
print('Interpretation: if the Gambler simulate the games for limitedd times,\n\
and the next realization of expected reward of the gambler will fall in the above projection interval')