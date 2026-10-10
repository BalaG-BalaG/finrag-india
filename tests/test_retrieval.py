from finrag.retrieve.hybrid import analyse_query, mmr_select, rrf_fuse


def test_rrf_prefers_items_ranked_well_by_both_arms():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    lex = [{"chunk_id": "c"}, {"chunk_id": "a"}, {"chunk_id": "z"}]
    fused = rrf_fuse([dense, lex], k=60)
    assert fused[0]["chunk_id"] == "a"          # 1st + 2nd beats 3rd + 1st
    assert {r["chunk_id"] for r in fused} == {"a", "b", "c", "z"}


def test_query_analysis_extracts_filters_and_metrics():
    p = analyse_query("What was the consolidated Gross NPA in FY2023-24?", strict_filters=True)
    assert p.filters["scope"] == "consolidated"
    assert p.filters["fiscal_year"] == "FY2023-24"
    assert "gross npa" in p.metrics


def test_comparison_questions_are_not_year_filtered():
    """A YoY question needs BOTH years; pre-filtering to one would guarantee failure."""
    p = analyse_query("How did net profit change from FY2022-23 to FY2023-24?", strict_filters=True)
    assert "fiscal_year" not in p.filters and p.comparative


def test_mmr_drops_near_duplicates():
    cands = [{"chunk_id": "1", "content": "gross npa 2.24", "rerank_score": 0.9},
             {"chunk_id": "2", "content": "gross npa 2.24", "rerank_score": 0.88},
             {"chunk_id": "3", "content": "provision coverage 91.89", "rerank_score": 0.5}]
    picked = [c["chunk_id"] for c in mmr_select(cands, 2, lam=0.5)]
    assert picked == ["1", "3"]
