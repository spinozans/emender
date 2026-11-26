"""Invented byte documents and an instantaneous lexicon. No corpus or tokenizer downloads."""

from .index import TokenIndex, spelling_step

NEW, OLD, YORK, JERSEY, CITY, SHORE, ROAD = 0, 1, 701, 902, 1027, 50_300, 255
PIECES = {
    NEW: b"New",
    OLD: b"Old",
    YORK: b" York",
    JERSEY: b" Jersey",
    CITY: b" City",
    SHORE: b" Shore",
    ROAD: b" Road",
}
DOCS = (
    b"New York City",
    b"New York City",
    b"New Jersey Shore",
    b"Old York Road",
    b"Old Jersey Road",
    b"Old Jersey Road",
)


def tokens(raw):
    # This particular finite vocabulary is prefix-free. A token commits as soon as
    # the last byte of its spelling arrives. Never read the next byte to decide.
    decode = {b: t for t, b in PIECES.items()}
    pending = b""
    for b in raw:
        pending += bytes((b,))
        if pending in decode:
            yield decode[pending]
            pending = b""
        elif not any(piece.startswith(pending) for piece in decode):
            raise ValueError("outside invented vocabulary")
    if pending:
        raise ValueError("incomplete token")


def main():
    index = TokenIndex(map(tokens, DOCS))
    h = index.seek(tokens(b"New"))
    answer = index.branch(h)

    def show(a):
        return [(PIECES[e.token].decode("ascii"), e.count) for e in a.edges]

    print("After New:", show(answer))
    for prefix in [b"", b" "]:
        print(
            f"Additional observed bytes {prefix!r}:",
            [
                (bytes((b,)), n)
                for b, n in spelling_step(
                    answer, PIECES, prefix
                ).continuing_bytes.items()
            ],
        )
    york = index.follow(h, YORK)
    print("After following the York edge:", show(index.branch(york)))
    print("After dropping New:", show(index.branch(index.suffix(york))))


if __name__ == "__main__":
    main()
