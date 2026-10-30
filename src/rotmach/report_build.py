"""Public report-building entry point shared with the presentation builder."""
from .slides_build import build_report, render_report_pdf

__all__ = ["build_report", "render_report_pdf"]
