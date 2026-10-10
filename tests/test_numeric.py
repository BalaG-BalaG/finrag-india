from finrag.numeric import (SCALES, detect_unit_scale, extract_numbers, format_indian,
                            numbers_match, parse_number)


def n(text, **kw):
    return extract_numbers(text, **kw)


def test_indian_grouping_parsed():
    assert parse_number("1,23,45,678.90") == 12345678.90
    assert parse_number("(1,234.5)") == -1234.5
    assert parse_number("Nil") == 0.0
    assert parse_number("-") is None


def test_periods_and_standards_are_not_quantities():
    for t in ["FY2023-24", "as at 31 March 2024", "Ind AS 109", "Note 12", "Q3", "Basel III"]:
        assert n(t) == [], f"{t} should not yield a quantity"


def test_percent_and_scale_detection():
    x = n("Gross NPA was 2.24 per cent")[0]
    assert x.is_percent and x.value == 2.24
    y = n("profit of 1.2 lakh crore")[0]
    assert y.scale == SCALES["lakh crore"] and y.absolute() == 1.2e12


def test_rounding_tolerance_is_precision_aware():
    ctx = n("1,234.56", keep_years=True)[0]
    assert numbers_match(n("1,234.6")[0], ctx)      # stated to 1 dp -> rounds correctly
    assert numbers_match(n("1,235")[0], ctx)        # stated to 0 dp -> rounds correctly
    assert not numbers_match(n("1,200")[0], ctx)    # not a rounding of the source
    assert not numbers_match(n("1,234.5678")[0], ctx)   # over-precise = invented digits


def test_scale_normalisation_catches_lakh_crore_swap():
    cell = n("1,20,345", keep_years=True)[0]        # a cell in a "(Rs. in crore)" table
    assert numbers_match(n("1.20345 lakh crore")[0], cell, ctx_default_scale=1e7)
    assert not numbers_match(n("1,20,345 lakh")[0], cell, ctx_default_scale=1e7)


def test_unit_banner_detection():
    assert detect_unit_scale("(Rs. in crore)")[1] == 1e7
    assert detect_unit_scale("Amounts in Lakhs")[1] == 1e5
    assert detect_unit_scale("no banner here")[0] is None


def test_format_indian_grouping():
    assert format_indian(12345678.9) == "1,23,45,678.90"
    assert format_indian(-1234.5, 1) == "-1,234.5"
