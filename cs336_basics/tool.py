import pickle

def save_bpe(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

def load_bpe(path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["vocab"], data["merges"]

def save_bpe_vocab(vocab: dict[int, bytes], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"vocab": vocab}, f)

def load_bpe_vocab(path: str) -> dict[int, bytes]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["vocab"] if isinstance(data, dict) and "vocab" in data else data

def save_bpe_merges(merges: list[tuple[bytes, bytes]], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"merges": merges}, f)

def load_bpe_merges(path: str) -> list[tuple[bytes, bytes]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["merges"] if isinstance(data, dict) and "merges" in data else data