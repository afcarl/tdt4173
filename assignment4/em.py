class ExpectationMaximization(object):
    def __init__(self):

        pass


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
