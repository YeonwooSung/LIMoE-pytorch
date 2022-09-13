from argparse import ArgumentParser
from tokenizers import Tokenizer, models, ByteLevelBPETokenizer
from pathlib import Path


def select_wordpiece():
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    return tokenizer

def select_bpe():
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    return tokenizer

def select_bytelevelbpe():
    tokenizer = ByteLevelBPETokenizer()
    return tokenizer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="wordpiece")
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--dataset_path", type=str, default="./data/")
    args = parser.parse_args()

    if args.tokenizer == "wordpiece":
        tokenizer = select_wordpiece()
    elif args.tokenizer == "bpe":
        tokenizer = select_bpe()
    elif args.tokenizer == "bytelevelbpe":
        tokenizer = select_bytelevelbpe()
    else:
        raise ValueError("Unknown tokenizer: {}".format(args.tokenizer))

    dataset = args.dataset_path if args.dataset_path.endswith("/") else args.dataset_path + "/"
    paths = [str(x) for x in Path(dataset).glob("**/*.txt")]

    # train tokenizer
    tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=2, special_tokens=[
        "[UNK]",
        "[SEP]",
        "[PAD]",
        "[CLS]",
        "[MASK]",
    ])

    # save tokenizer
    tokenizer.save(".", args.tokenizer)
    # tokenizer.save("tokenizer.json")
