import random
import argparse


class Regression(object):
    def __init__(self, w=None, p=2, b=0.0, alpha=0.1):
        """
        :param w: weight vector. Set this to None to initialize weights randomly
        :param p: the number of features
        :param b: the bias term
        :param alpha: learning rate
        :return:
        """
        if w is None:
            self.weights = [0.0] * p
            for i in xrange(p):
                self.weights[i] = random.uniform(-0.5, 0.5)
        else:
            self.weights = w
        self.bias = b
        self.alpha = alpha

    def learn_from_example(self, inputs, desired_output):
        actual_output = self.calculate_output(inputs)
        print 'input', inputs, 'output', actual_output
        self.update_weights(inputs, actual_output, desired_output)
        print 'weights are now', self.weights

    def update_weights(self, inputs, actual_output, desired_output):
        error = desired_output - actual_output
        print 'error', error
        for i in xrange(len(self.weights)):
            correction = self.alpha * inputs[i] * error
            self.weights[i] += correction

    def calculate_output(self, inputs):
        output = 0
        for i in xrange(len(inputs)):
            output += self.weights[i] * inputs[i]
        output -= self.bias
        return output


if __name__ == '__main__':
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
    args = arg_parser.parse_args()

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

    r = Regression(w=None, p=2, b=0.1, alpha=0.1)

    for i in range(len(xs)):
        r.learn_from_example(xs[i], ys[i])
