import random
import argparse


class Regression(object):
    def __init__(self, p=2, b=0.0, alpha=0.01):
        """
        :param w: weight vector. Set this to None to initialize weights randomly
        :param p: the number of features
        :param b: the bias term
        :param alpha: learning rate
        :return:
        """
        self.p = p
        self.weights = [0.0] * p
        for i in xrange(p):
            self.weights[i] = random.uniform(-0.5, 0.5)
        self.bias = b
        self.alpha = alpha

    def calculate_output(self, x):
        output = 0
        for i in xrange(len(x)):
            output += self.weights[i] * x[i]
        output += self.bias
        return output

    def calculate_error(self, xs, ys):
        error = 0
        for i in xrange(len(xs)):
            actual_output = self.calculate_output(xs[i])
            desired_output = ys[i]
            error += (actual_output - desired_output) ** 2
        error /= len(xs)
        return error

    def calculate_gradient_descent(self, xs, ys):
        w_correction = [0.0] * self.p
        b_correction = 0.0
        for i in xrange(len(xs)):
            output = self.calculate_output(xs[i])
            error = output - ys[i]
            for j in range(self.p):
                w_correction[j] += error * xs[i][j]
            b_correction += error
        for j in range(len(w_correction)):
            w_correction[j] *= 2.0 / len(xs)
        b_correction *= 2.0 / len(xs)
        return w_correction, b_correction

    def update_parameters(self, w_correction, b_correction):
        for j in xrange(len(self.weights)):
            self.weights[j] -= self.alpha * w_correction[j]
        self.bias -= self.alpha * b_correction


class Main(object):
    def __init__(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument(
                '-i',
                '--input',
                dest='filename',
                type=str,
                help='The name of the input file (csv)',
                required=False,
                default="data-train.csv"
        )
        arg_parser.add_argument(
                '-s',
                '--seed',
                dest='seed',
                type=int,
                help='A seed for the pseudo-random number generator',
                required=False,
                default="1"
        )
        args = arg_parser.parse_args()

        random.seed(args.seed)

        f = open(args.filename, 'r')
        lines = []
        for line in f:
            lines.append(line.strip())
        f.close()

        xs = []
        ys = []

        for line in lines:
            exploded = line.split(',')
            x = map(float, exploded[:-1])
            y = float(exploded[-1])
            xs.append(x)
            ys.append(y)

        r = Regression(p=2, b=random.uniform(-0.5, 0.5), alpha=0.1)
        print 'initial parameters: w = {0}, b = {1}'.format(r.weights, r.bias)

        for i in range(100):
            print
            error = r.calculate_error(xs, ys)
            print 'iteration', i + 1
            print 'error', error
            w_correction, b_correction = r.calculate_gradient_descent(xs, ys)
            r.update_parameters(w_correction, b_correction)
            print 'weights', r.weights
            print 'bias', r.bias


if __name__ == '__main__':
    Main()
