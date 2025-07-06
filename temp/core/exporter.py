import pandas as pd
from fpdf import FPDF
import os

def export_to_csv(df: pd.DataFrame, filepath: str):
    """
    Exports a DataFrame to CSV.

    Parameters:
        df (pd.DataFrame): The data to export
        filepath (str): Output file path
    """
    df.to_csv(filepath, index=False)

def export_to_pdf(summary: str, filepath: str):
    """
    Exports summary text to a simple PDF file.

    Parameters:
        summary (str): The content to write
        filepath (str): Output file path
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary.splitlines():
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(filepath)

