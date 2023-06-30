class FileParser:
    def __init__(self, file):
        self.f = file
        self.buffer = ""

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            line = self.f.readline()
            if line == "":
                if self.buffer != "":
                    to_return = self.buffer
                    self.buffer = ""
                    return to_return[:-1]
                raise StopIteration
            elif line == "\n":
                to_return = self.buffer
                self.buffer = ""
                return to_return[:-1]
            else:
                self.buffer += line