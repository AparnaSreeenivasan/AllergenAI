from __future__ import annotations

import hashlib
import io
import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import py3Dmol

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1, molecular_weight
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AllergenAI | Protein Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. VISUAL SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17343b;
        --muted: #687f83;
        --teal: #087f78;
        --teal-dark: #075e59;
        --mint: #e8f5f1;
        --line: #dce9e5;
        --surface: #ffffff;
        --canvas: #f4f8f6;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 248, 246, 0.92);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2.0rem;
        padding-bottom: 3.5rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.025em;
    }

    p, li, label, .stMarkdown {
        color: #405c61;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }
