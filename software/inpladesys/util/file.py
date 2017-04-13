
def write_all_text(path: str, text: str):
    with open(path, mode='w') as fs:
        fs.write(text)
        fs.flush()

def read_all_text(path: str) -> object:
    with open(path, encoding="utf8", mode='r') as fs:
        return fs.read()
