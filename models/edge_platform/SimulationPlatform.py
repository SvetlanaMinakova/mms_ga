

class SimulationPlatform:
    def __init__(self, name):
        self.name = name
        self.__processors = []
        self.__buffers = []

    def add_processor(self, proc):
        self.__processors.append(proc)

    def add_buffer(self, buf):
        self.__buffers.append(buf)

    def get_buffer_by_name(self, name):
        for buffer in self.__buffers:
            if buffer.name == name:
                return buffer

    def get_processors(self):
        return self.__processors

    def get_buffers(self):
        return self.__buffers

    def __str__(self):
        return "{name: " + self.name + ", processors: " + str(len(self.__processors)) +\
               ", buffers:" + str(len(self.__buffers)) + "}"

    def print_details(self, print_processors=True, print_buffers=True):
        print(self)
        if print_processors:
            print("Processors: ")
            for proc in self.__processors:
                print(" ", proc)
            print()
        if print_buffers:
            print("Buffers: ")
            for buf in self.__buffers:
                print(" ", buf)
            print()


class SimulationProc:
    def __init__(self, name):
        self.name = name
        self.task = None

    def assign(self, task):
        self.task = task

    def reset(self):
        self.free()

    def free(self):
        self.task = None

    def is_free(self):
        return self.task is None

    def __str__(self):
        return "{name: " + self.name + "}"


class SimulationBuffer:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.occupied = 0

    def reset(self):
        self.occupied = 0

    def write_tokens(self, tokens):
        if self.occupied + tokens <= self.size:
            self.occupied += tokens
        else:
            raise OverflowBufferException(buffer=self, tokens=tokens)

    def read_tokens(self, tokens):
        if self.occupied >= tokens:
            self.occupied -= tokens
        else:
            raise EmptyBufferException(buffer=self, tokens=tokens)

    def is_empty(self):
        return self.occupied == 0

    def is_full(self):
        return self.occupied == self.size

    def free_space(self):
        return self.size - self.occupied

    def __str__(self):
        return "{name: " + self.name + ", size: " + str(self.size) + "}"


class EmptyBufferException(Exception):
    def __init__(self, buffer, tokens):
        self.message = "Trying to read " + str(tokens) + \
                        " from buffer " + buffer.name + \
                        " that has", str(buffer.occupied) + " tokens."


class OverflowBufferException(Exception):
    def __init__(self, buffer, tokens):
        self.message = "Trying to write " + str(tokens) + \
                        " to buffer " + buffer.name + \
                        " that has size", str(buffer.size) + " tokens."

