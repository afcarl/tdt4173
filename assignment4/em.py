import math


class ExpectationMaximization(object):
    def __init__(self, k, mu, sigma, x):
        """
        k: number of gaussians
        mu: list of initial values for mean values of the gaussians
        sigma: standard deviation of the gaussians
        x: the samples
        """

        self.k = k
        self.sigma = sigma
        self.x = x

        self.E = []
        for j in range(k):
            self.E.append([0] * len(x))
        self.h = mu

    def run(self, num_iterations):
        for i in range(num_iterations):
            print 'iteration', i, self.h
            self.do_one_iteration()
        print 'finally', self.h

    def do_one_iteration(self):
        self.expectation_step()
        self.maximization_step()

    def expectation_step(self):
        for j in range(self.k):
            for i in range(len(self.x)):
                numerator = self.probability_density_function(self.x[i], self.h[j], self.sigma)
                denominator = 0
                for n in range(self.k):
                    denominator += self.probability_density_function(self.x[i], self.h[n], self.sigma)

                self.E[j][i] = numerator / denominator

    def maximization_step(self):
        for j in range(self.k):
            numerator = 0
            denominator = 0
            for i in range(len(self.x)):
                numerator += self.E[j][i] * self.x[i]
                denominator += self.E[j][i]

            self.h[j] = numerator / denominator

    def probability_density_function(self, x, mu, sigma):
        """
        Normal distribution
        """
        return math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))


class DataManager(object):
    def __init__(self):
        filename = 'sample-data.txt'
        # read file
        f = open(filename, 'r')
        lines = []
        for line in f:
            lines.append(line.strip())
        f.close()

        self.data = map(float, lines)


if __name__ == '__main__':
    data_manager = DataManager()
    em = ExpectationMaximization(
        k=2,
        mu=[-0.5, 2.5],
        sigma=1,
        x=data_manager.data
    )
    em.run(num_iterations=20)
