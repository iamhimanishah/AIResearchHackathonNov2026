_CHOICES = ["yes", "no", "maybe"]


def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["context"]["contexts"])
    return f"Abstract: {ctxs}\nQuestion: {doc['question']}\nAnswer:"


def doc_to_target(doc) -> int:
    return _CHOICES.index(doc["final_decision"])
