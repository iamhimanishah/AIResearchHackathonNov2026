MAX_HOLDING_CHARS = 2000  # a handful of test rows carry a ~20k-token "holding" (data error)


def process_docs(dataset):
    return dataset.filter(lambda d: all(len(d[f"holding_{i}"]) <= MAX_HOLDING_CHARS for i in range(5)))


def doc_to_text(doc) -> str:
    # citing_prompt ends with the citation whose holding is masked as <HOLDING>
    return f"{doc['citing_prompt']}\nHolding:"


def doc_to_choice(doc) -> list[str]:
    return [f" {doc[f'holding_{i}']}" for i in range(5)]


def doc_to_target(doc) -> int:
    return int(doc["label"])
