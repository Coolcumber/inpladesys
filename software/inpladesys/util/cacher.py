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
        pass

    def _open(self, name, mode):
        return open(path.join(self._dir, name + ".p"), mode)

    def __contains__(self, item):
        return item in self._files

    def __getitem__(self, item):
        return pickle.load(self._open(item, 'rb'))

    def __setitem__(self, key, value):
        pickle.dump(value, self._open(key, 'wb'))

    def cache(self):
        def decorator(func):
            if self._dummy:
                return lambda: value
            elif func.__name__ in self:
                return lambda: self[func.__name__]
            else:
                value = func()
                self[func.__name__] = value
                return lambda: value
        return decorator


if __name__ == "__main__":
    c = Cacher(".cache-test")
    c["a"] = [i for i in range(5)]
    c["b"] = [i ** 2 for i in range(5)]
    print(c["a"])

    @c.cache()
    def test():
        import time
        return time.ctime(), "kupus"

    print(test())
