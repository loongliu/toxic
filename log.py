import sys
import datetime
import os


class Logger(object):
    def __init__(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = str(datetime.datetime.now()) + '.log'
        file_path = os.path.join(output_dir, filename)
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        if message == '\n':
            return
        time = datetime.datetime.now()
        message = f'{time}: {message}\n'
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()


def init_log():
    sys.stdout = Logger('logs')


if __name__ == '__main__':
    init_log()
    print(10)
    print('ok')
    print('file will open\nnew line')
