

class TextTransformer:
    """Map characters to integers and vice versa"""
    def __init__(self, enrich_target=True, length=10):
        self.char_map = self.__char_map(length)

        if enrich_target:
            self.char_map['*'] = length
            # add a blank symbol
            self.char_map[''] = length + 1
        else:
            self.char_map[''] = length
        self.index_map = {value: key for (key, value) in self.char_map.items()}

    def txt2int(self, text):
        if not type(text) == str:
            print(text)
            raise ValueError(f'{type(text)}')
        return [self.char_map[char] for char in text]

    def int2txt(self, ints):
        return ''.join([self.index_map[idx] for idx in ints])

    def __char_map(self, length):
        return {str(i): i for i in range(length)}
