from indicvoicerag.evaluation import hit_at_k, mean_reciprocal_rank, recall_at_k


def test_hit_recall_and_mrr() -> None:
    relevant = {"d2", "d3"}
    ranked = ["d1", "d2", "d4"]
    assert hit_at_k(relevant, ranked, k=1) == 0.0
    assert hit_at_k(relevant, ranked, k=2) == 1.0
    assert recall_at_k(relevant, ranked, k=2) == 0.5
    assert mean_reciprocal_rank([]) == 0.0
