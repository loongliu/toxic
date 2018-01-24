import sys
import datetime
import os


class Logger(object):
    def __init__(self, output_dir):
        ensure_dir(output_dir)

        filename = str(datetime.datetime.now())
        self.log_dir = os.path.join(output_dir, filename)
        ensure_dir(self.log_dir)
        file_path = os.path.join(self.log_dir, 'log.log')
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


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_log():
    logger = Logger('logs')
    sys.stdout = logger
    sys.stderr = logger
    return logger.log_dir


if __name__ == '__main__':
    init_log()
    print(10)
    print('ok')
    print('file will open\nnew line')
