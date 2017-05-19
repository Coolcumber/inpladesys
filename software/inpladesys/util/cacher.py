import pickle
import os
import os.path as path
from inpladesys.util.directory import get_files


class Cacher:
    def __init__(self, dir, dummy=False):
        self._dummy = dummy
        if dummy:
            return
        self._dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        self._files = set(path.splitext(file)[0] for file in os.listdir(dir))

    @staticmethod
    def dummy():
        return Cacher(dir=None, dummy=True)

    def clear(self):
        for f in self._files:
            os.remove(f)
        self._files = set()

    def _open(self, name, mode):
        return open(path.join(self._dir, name + ".p"), mode)

    def __contains__(self, item):
        return item in self._files

    def __getitem__(self, item):
        return pickle.load(self._open(item, 'rb'))

    def __setitem__(self, key, value):
        pickle.dump(value, self._open(key, 'wb'))

    def __call__(self, name=None):
        """
        :param name: name of the file to store the result in - set to function name if not stated
        """
        def decorator(func):
            nonlocal name
            if name is None:
                name = func.__name__
            if self._dummy:
                return func
            elif name in self:
                def wrapper(*args, **kwargs):
                    return self[name]
                return wrapper
            else:
                def wrapper(*args, **kwargs):
                    value = func(*args, **kwargs)
                    self[name] = value
                    return value
                return wrapper
        return decorator


if __name__ == "__main__":
    c = Cacher(".cache-test")
    c["a"] = [i for i in range(5)]
    c["b"] = [i ** 2 for i in range(5)]
    print(c["a"])


    @c()
    def test():
        import time
        return time.ctime(), "foo"


    print(test())
