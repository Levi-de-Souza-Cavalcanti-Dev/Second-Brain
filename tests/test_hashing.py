from secondbrain.ingestion.hashing import compute_file_content_hash_utf8, normalize_text_for_hash


def test_normalize_crlf_trailing_spaces() -> None:
    raw = "  alfa \r\nbeta  \r\n"
    hashed = normalize_text_for_hash(raw)
    assert hashed == b"alfa\nbeta\n"


def test_hash_changes_with_content() -> None:
    a = compute_file_content_hash_utf8("alfa\nbeta\n")
    b = compute_file_content_hash_utf8("alfa\ngamma\n")
    assert a != b


def test_hash_empty_stable() -> None:
    blank = normalize_text_for_hash("")
    non_blank = normalize_text_for_hash("x")
    assert blank != non_blank

