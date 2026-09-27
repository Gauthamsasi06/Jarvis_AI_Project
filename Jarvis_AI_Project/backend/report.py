"""
JARVIS AI - Report Generator
Owner: Product Lead
Builds a downloadable PDF summary of the analysis for demo/judging handouts.
"""

from fpdf import FPDF
from datetime import datetime

ACCENT = (0, 200, 255)
GOOD = (60, 220, 130)
WARN = (255, 200, 40)
BAD = (255, 70, 130)


def _verdict_color(verdict: str):
    return {
        "Excellent": GOOD, "Good": ACCENT, "Needs cleanup": WARN, "Poor": BAD,
    }.get(verdict, ACCENT)


class JarvisReport(FPDF):
    def header(self):
        self.set_fill_color(10, 14, 24)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(*ACCENT)
        self.rect(0, 0, 210, 3, "F")
        self.set_text_color(*ACCENT)
        self.set_font("Helvetica", "B", 20)
        self.set_y(10)
        self.cell(0, 12, "JARVIS AI - Analysis Report", ln=True)
        self.set_text_color(150, 150, 160)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        self.ln(4)

    def section(self, title: str):
        self.set_text_color(*ACCENT)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, title, ln=True)
        self.set_draw_color(*ACCENT)
        y = self.get_y()
        self.line(10, y, 60, y)
        self.ln(3)
        self.set_text_color(220, 224, 235)
        self.set_font("Helvetica", "", 10)

    def body(self, text: str):
        self.set_text_color(220, 224, 235)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def health_badge(self, score: int, verdict: str):
        color = _verdict_color(verdict)
        self.set_fill_color(*color)
        self.set_text_color(15, 15, 20)
        self.set_font("Helvetica", "B", 14)
        self.cell(60, 12, f"  Health Score: {score}/100  ", fill=True, border=0)
        self.set_text_color(*color)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 12, f"   {verdict}", ln=True)
        self.set_text_color(220, 224, 235)
        self.ln(3)


def build_pdf_report(filename: str, profile: dict, insights: str, corr_pairs: list,
                      anomalies_count: int, health: dict | None = None):
    pdf = JarvisReport()
    pdf.add_page()

    if health:
        pdf.health_badge(health["score"], health["verdict"])

    pdf.section("Dataset Overview")
    pdf.body(
        f"Rows: {profile['rows']}   |   Columns: {profile['columns']}\n"
        f"Missing data: {profile['missing_pct']}%   |   Duplicate rows: {profile['duplicate_rows']}\n"
        f"Numeric columns: {', '.join(profile['numeric_cols']) or 'None'}\n"
        f"Categorical columns: {', '.join(profile['categorical_cols']) or 'None'}"
    )

    pdf.section("AI-Generated Insights")
    pdf.body(insights or "No insights generated.")

    pdf.section("Top Correlations")
    if corr_pairs:
        for c in corr_pairs:
            pdf.body(f"- {c['a']} vs {c['b']}: r = {c['corr']}")
    else:
        pdf.body("Not enough numeric columns to compute correlations.")

    pdf.section("Anomaly Detection")
    pdf.body(f"{anomalies_count} anomalous rows flagged by Isolation Forest.")

    pdf.output(filename)
    return filename
