from finrag.ingest.metadata import ParsedTable, detect_fiscal_years, detect_scope
from finrag.ingest.table_chunker import (chunk_table, detect_header_rows, flatten_headers,
                                         row_label_path, stitch_continuations)
from finrag.ingest.text_chunker import naive_chunk, pack_sentences

GRID = [
    ["Particulars", "As at 31 March 2024", "As at 31 March 2024", "As at 31 March 2023"],
    ["", "Consolidated", "Standalone", "Consolidated"],
    ["Advances", "27,65,432.10", "26,10,000.00", "24,32,109.87"],
    ["  Term loans", "18,90,123.45", "17,00,000.00", "16,54,321.09"],
    ["Deposits", "36,78,901.45", "35,00,000.00", "33,10,456.78"],
]


def table():
    return ParsedTable(table_id="t1", page_no=4, grid=[r[:] for r in GRID],
                       caption="Statement 2: Balance Sheet", unit_name="crore",
                       unit_multiplier=1e7, n_header_rows=2, row_indents=[0, 0, 0, 1, 0])


def test_multi_row_header_is_flattened_with_span_fill():
    cols = flatten_headers(GRID, 2)
    assert cols[1] == "As at 31 March 2024 > Consolidated"
    assert cols[2] == "As at 31 March 2024 > Standalone"   # spanner forward-filled
    assert cols[3] == "As at 31 March 2023 > Consolidated"


def test_header_band_detection():
    assert detect_header_rows(GRID) == 2


def test_row_hierarchy_from_geometric_indent():
    assert row_label_path(GRID, 2, 3, [0, 0, 0, 1, 0]) == "Advances > Term loans"
    assert row_label_path(GRID, 2, 4, [0, 0, 0, 1, 0]) == "Deposits"


def test_row_chunks_are_self_contained_and_typed():
    chunks = chunk_table(table(), "doc1", "Test Bank", "consolidated")
    kinds = {c.element_type for c in chunks}
    assert kinds == {"table", "table_row", "table_rowgroup"}
    row = next(c for c in chunks if c.element_type == "table_row"
               and c.content.startswith("Advances > Term loans"))
    assert "18,90,123.45" in row.content and "crore" in row.content
    assert row.parent_id == next(c.chunk_id for c in chunks if c.element_type == "table")
    assert "Statement 2" in row.embed_text          # contextual header for retrieval


def test_multi_period_row_is_not_pinned_to_one_fiscal_year():
    row = next(c for c in chunk_table(table(), "d", "E", None)
               if c.element_type == "table_row")
    assert row.fiscal_year == ""                    # else a FY filter would hide it


def test_continuation_tables_are_stitched():
    a = ParsedTable("t1", 4, [r[:] for r in GRID])
    b = ParsedTable("t2", 5, [GRID[0][:], GRID[1][:], ["Borrowings", "1", "2", "3"]],
                    caption="Statement 2 (contd.)")
    out = stitch_continuations([a, b])
    assert len(out) == 1 and out[0].page_end == 5
    assert any(r[0] == "Borrowings" for r in out[0].grid)


def test_naive_baseline_is_a_fixed_window():
    chunks = naive_chunk(" ".join(str(i) for i in range(2000)), "d", "E")
    assert len(chunks) > 1 and all(c.pipeline == "naive" for c in chunks)


def test_sentence_packing_respects_budget():
    sents = ["This is sentence number %d." % i for i in range(40)]
    packed = pack_sentences(sents, max_tokens=40, overlap_tokens=10)
    assert len(packed) > 1


def test_fiscal_year_and_scope_normalisation():
    assert detect_fiscal_years("FY2023-24")[0] == "FY2023-24"
    assert detect_fiscal_years("year ended 31 March 2024")[0] == "FY2023-24"
    assert detect_scope("Consolidated balance sheet") == "consolidated"
    assert detect_scope("no marker") is None
