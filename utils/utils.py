import sys


def sqrt(x):
    return x**0.5


class Logger:
    def __init__(self, file_name: str):
        self.file = open(file_name, 'w', encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.terminal.write(message)

    def flush(self):
        self.file.flush()
        self.terminal.flush()

    def close(self):
        self.file.close()
        self.terminal.close()