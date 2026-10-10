"""Generates a synthetic Indian-bank annual-report PDF (ruled tables, Rs. crore
banners, consolidated vs standalone, multi-page table continuation, indented row
hierarchy) so the whole pipeline is testable before the real RBI/NSE PDFs land."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

S = getSampleStyleSheet()
H = ParagraphStyle("h", parent=S["Heading2"], spaceAfter=6)
B = ParagraphStyle("b", parent=S["BodyText"], fontSize=9, leading=12)

GRID = TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
])


def build(path="data/raw/sample_bank_ar.pdf"):
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm)
    st = []
    st.append(Paragraph("Bharat National Bank Limited - Annual Report 2023-24", S["Title"]))
    st.append(Paragraph(
        "The financial statements have been prepared in accordance with Ind AS notified under "
        "the Companies Act, 2013. All amounts are in Indian Rupees unless otherwise stated. "
        "The Bank's asset quality improved during the year ended 31 March 2024, aided by higher "
        "recoveries and upgradations in the corporate portfolio.", B))
    st.append(Spacer(1, 8))

    st.append(Paragraph("Statement 1: Key Financial Indicators (Consolidated)", H))
    st.append(Paragraph("(Rs. in crore)", B))
    st.append(Table([
        ["Particulars", "FY2023-24", "FY2022-23", "FY2021-22"],
        ["Total income", "3,45,678.90", "3,01,234.56", "2,78,901.23"],
        ["Interest earned", "2,98,765.43", "2,61,098.77", "2,40,112.09"],
        ["  Interest on advances", "2,10,456.78", "1,85,320.44", "1,70,998.10"],
        ["  Income on investments", "78,234.56", "68,912.33", "60,455.87"],
        ["Operating profit", "89,012.34", "72,345.67", "64,210.98"],
        ["Provisions and contingencies", "31,245.67", "29,876.54", "34,567.89"],
        ["Net profit", "57,766.67", "42,469.13", "29,643.09"],
        ["Total assets", "48,76,543.21", "43,21,098.76", "39,87,654.32"],
    ], colWidths=[62 * mm, 35 * mm, 35 * mm, 35 * mm], style=GRID))
    st.append(Spacer(1, 10))

    st.append(Paragraph("Statement 2: Asset Quality Ratios (Standalone)", H))
    st.append(Table([
        ["Ratio", "As at 31 March 2024", "As at 31 March 2023"],
        ["Gross NPA ratio (%)", "2.24", "2.78"],
        ["Net NPA ratio (%)", "0.57", "0.67"],
        ["Provision coverage ratio (%)", "91.89", "90.14"],
        ["Capital adequacy ratio - CRAR (%)", "14.28", "14.68"],
        ["  of which Tier I (%)", "12.02", "12.30"],
        ["Return on assets (%)", "1.18", "0.96"],
    ], colWidths=[75 * mm, 46 * mm, 46 * mm], style=GRID))
    st.append(Spacer(1, 8))
    st.append(Paragraph(
        "Gross non-performing assets declined to 2.24 per cent as at 31 March 2024 from 2.78 per "
        "cent a year earlier, a reduction of 54 basis points. The provision coverage ratio "
        "improved to 91.89 per cent. Deposits grew by 11.13 per cent during the year.", B))
    st.append(PageBreak())

    st.append(Paragraph("Statement 3: Segment Results (Consolidated) (Rs. in lakh)", H))
    st.append(Table([
        ["Segment", "Revenue FY2023-24", "Result FY2023-24", "Revenue FY2022-23"],
        ["Treasury", "98,76,543", "12,34,567", "87,65,432"],
        ["Corporate banking", "1,45,67,890", "23,45,678", "1,32,10,987"],
        ["Retail banking", "1,78,90,123", "31,20,456", "1,55,43,210"],
        ["Other banking operations", "22,43,210", "4,56,789", "19,87,654"],
    ], colWidths=[50 * mm, 40 * mm, 40 * mm, 40 * mm], style=GRID))
    st.append(Spacer(1, 8))
    st.append(Paragraph("Statement 3: Segment Results (contd.)", H))
    st.append(Table([
        ["Segment", "Revenue FY2023-24", "Result FY2023-24", "Revenue FY2022-23"],
        ["Unallocated", "3,21,098", "76,543", "2,98,765"],
        ["Total", "4,48,98,864", "72,34,033", "3,98,06,048"],
    ], colWidths=[50 * mm, 40 * mm, 40 * mm, 40 * mm], style=GRID))
    st.append(Spacer(1, 10))
    st.append(Paragraph("Deposits and Advances", H))
    st.append(Table([
        ["Particulars", "31 March 2024", "31 March 2023"],
        ["Deposits", "36,78,901.45", "33,10,456.78"],
        ["  Current account deposits", "3,12,345.67", "2,87,654.32"],
        ["  Savings bank deposits", "12,45,678.90", "11,98,765.43"],
        ["Advances", "27,65,432.10", "24,32,109.87"],
        ["  Term loans", "18,90,123.45", "16,54,321.09"],
    ], colWidths=[70 * mm, 45 * mm, 45 * mm], style=GRID))
    st.append(Paragraph("(Rs. in crore)", B))
    doc.build(st)
    print("wrote", path)


if __name__ == "__main__":
    build()
