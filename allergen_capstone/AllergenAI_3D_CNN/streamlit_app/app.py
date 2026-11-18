
from __future__ import annotations

import hashlib
import io
import textwrap
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import py3Dmol

# Streamlit's Markdown parser treats indented HTML as a code block.
# Dedent only Markdown calls that explicitly allow HTML, preserving
# ordinary Markdown content and all existing application functionality.
_original_st_markdown = st.markdown
def _dedented_st_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html", False) and isinstance(body, str):
        body = textwrap.dedent(body).strip()
    return _original_st_markdown(body, *args, **kwargs)
st.markdown = _dedented_st_markdown

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AllergenAI | Protein Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL UI DESIGN
# ============================================================

st.markdown(
    """
    <style>

    @import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
    );

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
        background: rgba(244,248,246,0.92);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.025em;
    }

    p, li, label {
        color: #405c61;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .brand-card {
        padding: 20px 18px;
        border: 1px solid #cfe5df;
        border-radius: 18px;
        background: linear-gradient(
            145deg,
            #f5fbf8,
            #e7f4ef
        );
        margin-bottom: 24px;
    }

    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        color: #087f78;
    }

    .brand-caption {
        color: #67817e;
        font-size: 12px;
        margin-top: 7px;
        letter-spacing: .04em;
    }

    .eyebrow {
        color: #087f78;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .15em;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .hero {
        padding: 34px 36px;
        border-radius: 24px;
        background: linear-gradient(
            115deg,
            #ffffff 0%,
            #edf7f3 100%
        );
        border: 1px solid #d7e9e3;
        box-shadow: 0 8px 28px rgba(28,75,67,0.045);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: clamp(30px, 4vw, 46px);
        line-height: 1.1;
        margin: 0 0 12px 0;
    }

    .hero p {
        max-width: 760px;
        font-size: 16px;
        line-height: 1.7;
        color: #587276;
        margin: 0;
    }

    .section-kicker {
        color: #087f78;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .13em;
        margin-bottom: 5px;
    }

    .soft-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 19px 20px;
        box-shadow: 0 5px 18px rgba(28,75,67,0.035);
    }

    .soft-card h3 {
        font-size: 17px;
        margin: 0 0 8px 0;
    }

    .soft-card p {
        font-size: 13px;
        line-height: 1.6;
        margin: 0;
        color: #6b8083;
    }

    .small-label {
        color: #708589;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: 10px;
        font-weight: 700;
    }

    .result-title {
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 11px;
        border: 1px solid #087f78;
        background: #087f78;
        color: #ffffff;
        font-weight: 700;
        padding: .62rem 1.05rem;
        transition: all .18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #075e59;
        border-color: #075e59;
        color: #ffffff;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #9bc9bd;
        border-radius: 15px;
        background: #fbfdfc;
        padding: 12px;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 15px 17px;
        box-shadow: 0 4px 15px rgba(28,75,67,0.025);
    }

    [data-testid="stMetricValue"] {
        color: #087f78;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #dce9e5;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 24px 21px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. PROJECT PATHS AND MODEL SETTINGS
# ============================================================

APP_FOLDER = Path(__file__).resolve().parent

MODEL_PATH = APP_FOLDER / "allergen3d_cnn_final.pt"

METRICS_PATH = APP_FOLDER / "3d_cnn_evaluation_metrics.csv"

GRID_SIZE = 48

VOXEL_SIZE = 2.5

ELEMENTS = ["C", "N", "O", "S", "P"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. EXISTING 3D CNN ARCHITECTURE
# ============================================================

class Allergen3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.AdaptiveAvgPool3d(1)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(32, 1)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading trained 3D CNN...")

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep the trained model in the same folder as app.py."
        )

    model = Allergen3DCNN().to(DEVICE)

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

    except TypeError:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            checkpoint = checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):

        checkpoint = {
            key.replace("module.", "", 1)
            if key.startswith("module.")
            else key: value

            for key, value in checkpoint.items()
        }

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

    return model



# ============================================================
# 6. PDB STRUCTURE VALIDATION
# ============================================================

def parse_pdb_text(pdb_text):

    if not pdb_text or not pdb_text.strip():

        raise ValueError(
            "The uploaded PDB file is empty."
        )

    parser = PDBParser(QUIET=True)

    handle = io.StringIO(pdb_text)

    try:

        structure = parser.get_structure(
            "uploaded_protein",
            handle
        )

    except Exception as exc:

        raise ValueError(
            f"Could not parse PDB file: {exc}"
        )

    models = list(structure.get_models())

    if not models:

        raise ValueError(
            "No protein model found in PDB file."
        )

    atoms = list(structure.get_atoms())

    if not atoms:

        raise ValueError(
            "No atoms found in PDB file."
        )

    supported = [

        atom for atom in atoms

        if (atom.element or "").strip().upper()
        in ELEMENTS

    ]

    if not supported:

        raise ValueError(
            "No supported atoms found. "
            "Expected C, N, O, S or P atoms."
        )

    return structure


# ============================================================
# 7. PDB TO 3D VOXEL CONVERSION
# ============================================================

def pdb_to_voxel(pdb_text):

    structure = parse_pdb_text(pdb_text)

    coordinates = []

    elements = []

    first_model = next(
        structure.get_models()
    )

    for atom in first_model.get_atoms():

        element = (
            atom.element or ""
        ).strip().upper()

        if element in ELEMENTS:

            coordinates.append(
                atom.coord
            )

            elements.append(
                element
            )

    if not coordinates:

        raise ValueError(
            "No supported atoms available for voxelization."
        )

    coordinates = np.asarray(
        coordinates,
        dtype=np.float32
    )

    if not np.isfinite(coordinates).all():

        raise ValueError(
            "Invalid atom coordinates detected."
        )

    # Center protein coordinates

    center = coordinates.mean(axis=0)

    coordinates = coordinates - center

    # Initialize five-channel voxel grid

    voxel = np.zeros(
        (
            5,
            GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE
        ),
        dtype=np.float32
    )

    center_grid = GRID_SIZE // 2

    outside_atoms = 0

    # Map atoms into voxel grid

    for coord, element in zip(
        coordinates,
        elements
    ):

        index = np.rint(
            coord / VOXEL_SIZE + center_grid
        ).astype(int)

        x, y, z = index.tolist()

        if (

            0 <= x < GRID_SIZE

            and 0 <= y < GRID_SIZE

            and 0 <= z < GRID_SIZE

        ):

            channel = ELEMENTS.index(
                element
            )

            voxel[
                channel,
                x,
                y,
                z
            ] = 1.0

        else:

            outside_atoms += 1

    return (
        voxel,
        outside_atoms,
        len(coordinates)
    )


# ============================================================
# 8. PROTEIN PREDICTION
# ============================================================

def predict_protein(pdb_text):

    voxel, outside_atoms, total_atoms = (
        pdb_to_voxel(pdb_text)
    )

    input_tensor = torch.from_numpy(
        voxel
    ).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.inference_mode():

        output = model(
            input_tensor
        )

        probability = torch.sigmoid(
            output
        ).item()

    prediction = (

        "Allergen"

        if probability >= 0.5

        else "Non-Allergen"

    )

    return (
        prediction,
        probability,
        outside_atoms,
        total_atoms
    )


# ============================================================
# 9. EXTRACT PROTEIN INFORMATION
# ============================================================

def get_protein_information(pdb_text):

    structure = parse_pdb_text(
        pdb_text
    )

    first_model = next(
        structure.get_models()
    )

    sequences = {}

    residue_count = 0

    atom_count = 0

    element_counts = Counter()

    # Extract amino acid sequences

    for chain in first_model:

        sequence = ""

        for residue in chain:

            if residue.id[0] == " ":

                residue_count += 1

                try:

                    amino_acid = seq1(
                        residue.resname,
                        custom_map={
                            "UNK": "X"
                        }
                    )

                except Exception:

                    amino_acid = "X"

                sequence += amino_acid

        sequences[chain.id] = sequence

    # Extract atom composition

    for atom in first_model.get_atoms():

        atom_count += 1

        element = (
            atom.element or ""
        ).strip().upper()

        if element:

            element_counts[element] += 1

    combined_sequence = "".join(
        sequences.values()
    )

    weight = None

    try:

        if (
            combined_sequence
            and "X" not in combined_sequence
        ):

            weight = ProteinAnalysis(
                combined_sequence
            ).molecular_weight()

    except Exception:

        weight = None

    amino_acid_counts = Counter(
        combined_sequence
    )

    return {

        "sequences": sequences,

        "sequence": combined_sequence,

        "residues": residue_count,

        "atoms": atom_count,

        "chains": list(sequences.keys()),

        "elements": dict(element_counts),

        "amino_acids": dict(amino_acid_counts),

        "molecular_weight": weight

    }


# ============================================================
# 10. INTERACTIVE 3D PROTEIN VIEWER
# ============================================================

def show_3d_structure(
    pdb_text,
    style="Cartoon"
):

    view = py3Dmol.view(
        width=1100,
        height=590
    )

    view.addModel(
        pdb_text,
        "pdb"
    )

    if style == "Cartoon":

        view.setStyle({
            "cartoon": {
                "color": "spectrum"
            }
        })

    elif style == "Stick":

        view.setStyle({
            "stick": {
                "radius": 0.16
            }
        })

    elif style == "Sphere":

        view.setStyle({
            "sphere": {
                "scale": 0.28
            }
        })

    elif style == "Surface":

        view.setStyle({
            "cartoon": {
                "color": "spectrum"
            }
        })

        view.addSurface(
            py3Dmol.VDW,
            {
                "opacity": 0.68,
                "color": "lightblue"
            }
        )

    view.setBackgroundColor(
        "#fbfdfc"
    )

    view.zoomTo()

    components.html(
        view._make_html(),
        height=620,
        scrolling=False
    )


# ============================================================
# 11. CREATE PDF ANALYSIS REPORT
# ============================================================

def create_pdf(
    filename,
    prediction,
    probability,
    info,
    outside,
    total
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#087f78"),
            fontSize=21,
            leading=26,
            spaceAfter=7
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#075e59"),
            fontSize=13,
            leading=17,
            spaceBefore=12,
            spaceAfter=7
        )
    )

    story = [

        Paragraph(
            "AllergenAI",
            styles["ReportTitle"]
        ),

        Paragraph(
            "Protein Structure Analysis Report",
            styles["BodyText"]
        ),

        Spacer(1, 8 * mm),

        Paragraph(
            "Prediction Results",
            styles["SectionHeading"]
        )

    ]

    # Prediction information

    result_data = [

        ["Protein file", filename],

        ["Classification", prediction],

        [
            "Allergen model score",
            f"{probability * 100:.2f}%"
        ],

        [
            "Non-allergen model score",
            f"{(1-probability) * 100:.2f}%"
        ]

    ]

    result_table = Table(
        result_data,
        colWidths=[
            52 * mm,
            112 * mm
        ]
    )

    result_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#edf7f3")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#17343b")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#dce9e5")
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )

    story.append(
        result_table
    )

    # Protein information

    story.append(
        Paragraph(
            "Protein Information",
            styles["SectionHeading"]
        )
    )

    protein_rows = [

        ["Residues", str(info["residues"])],

        ["Atoms", str(info["atoms"])],

        [
            "Chains",
            ", ".join(info["chains"]) or "Not available"
        ],

        [
            "Supported atoms",
            str(total)
        ],

        [
            "Atoms outside voxel grid",
            str(outside)
        ],

        [
            "Voxel grid",
            f"{GRID_SIZE} × {GRID_SIZE} × {GRID_SIZE}"
        ],

        [
            "Voxel size",
            f"{VOXEL_SIZE} Å"
        ]

    ]

    if info["molecular_weight"] is not None:

        protein_rows.append([

            "Molecular weight",

            f"{info['molecular_weight']:,.2f} Da"

        ])

    protein_table = Table(
        protein_rows,
        colWidths=[
            75 * mm,
            89 * mm
        ]
    )

    protein_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#edf7f3")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#dce9e5")
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])
    )

    story.append(
        protein_table
    )

    # Atom composition

    story.append(
        Paragraph(
            "Atom Composition",
            styles["SectionHeading"]
        )
    )

    atom_rows = [
        ["Element", "Count"]
    ]

    for element, count in sorted(
        info["elements"].items()
    ):

        atom_rows.append([
            element,
            str(count)
        ])

    atom_table = Table(
        atom_rows,
        colWidths=[
            55 * mm,
            55 * mm
        ]
    )

    atom_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#e8f5f1")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#dce9e5")
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])
    )

    story.append(
        atom_table
    )

    story.append(
        Spacer(1, 9 * mm)
    )

    story.append(
        Paragraph(
            "<b>Research limitation:</b> AllergenAI is an "
            "experimental research prototype. Model scores "
            "are not necessarily calibrated probabilities "
            "of real-world allergenicity. This report is not "
            "a clinical or food-safety determination.",
            styles["BodyText"]
        )
    )

    doc.build(story)

    return buffer.getvalue()

# ============================================================
# 12. SHARED HEADER COMPONENT
# ============================================================

def show_header(eyebrow, title, subtitle):

    st.markdown(
        f"""
        <div class="hero">

            <div class="eyebrow">
                {eyebrow}
            </div>

            <h1>
                {title}
            </h1>

            <p>
                {subtitle}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 13. FILE SIGNATURE
# ============================================================

def get_file_signature(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# 14. PDB FILE UPLOAD COMPONENT
# ============================================================

def render_upload(label, key):

    uploaded = st.file_uploader(
        label,
        type=["pdb"],
        key=key,
        help="Upload a protein structure in PDB format."
    )

    if uploaded is None:

        return None

    file_bytes = uploaded.getvalue()

    if not file_bytes:

        st.error(
            "The selected file is empty."
        )

        return None

    if len(file_bytes) > 200 * 1024 * 1024:

        st.error(
            "File exceeds the 200 MB application limit."
        )

        return None

    try:

        pdb_text = file_bytes.decode(
            "utf-8",
            errors="replace"
        )

        parse_pdb_text(
            pdb_text
        )

    except Exception as exc:

        st.error(
            f"Invalid PDB structure: {exc}"
        )

        return None

    st.success(
        f"Structure ready: {uploaded.name}"
    )

    st.caption(
        f"File size: {len(file_bytes) / 1024:.2f} KB"
    )

    return {

        "name": uploaded.name,

        "bytes": file_bytes,

        "text": pdb_text,

        "signature": get_file_signature(
            file_bytes
        )

    }


# ============================================================
# 15. PROTEIN SUMMARY COMPONENT
# ============================================================

def render_protein_summary(info):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Residues",
        f"{info['residues']:,}"
    )

    col2.metric(
        "Total Atoms",
        f"{info['atoms']:,}"
    )

    col3.metric(
        "Chains",
        len(info["chains"])
    )

    weight = info["molecular_weight"]

    col4.metric(
        "Molecular Weight",
        f"{weight:,.1f} Da"
        if weight is not None
        else "N/A"
    )


# ============================================================
# 16. AMINO ACID AND ATOM COMPOSITION
# ============================================================

def render_composition(info):

    amino_df = pd.DataFrame(
        sorted(
            info["amino_acids"].items()
        ),
        columns=[
            "Amino Acid",
            "Count"
        ]
    )

    atom_df = pd.DataFrame(
        sorted(
            info["elements"].items()
        ),
        columns=[
            "Element",
            "Atom Count"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Amino Acid Composition"
        )

        if not amino_df.empty:

            st.bar_chart(
                amino_df.set_index(
                    "Amino Acid"
                ),
                height=260
            )

            st.dataframe(
                amino_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No amino acid sequence available."
            )

    with col2:

        st.markdown(
            "#### Atom Composition"
        )

        if not atom_df.empty:

            st.bar_chart(
                atom_df.set_index(
                    "Element"
                ),
                height=260
            )

            st.dataframe(
                atom_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No atom composition available."
            )


# ============================================================
# 17. PREDICTION RESULT COMPONENT
# ============================================================

def render_result(result):

    prediction = result["prediction"]

    probability = result["probability"]

    allergen_score = probability * 100

    non_allergen_score = (
        1 - probability
    ) * 100

    st.markdown(
        '<div class="section-kicker">MODEL OUTPUT</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        col1, col2 = st.columns([1.3, 1])

        with col1:

            if prediction == "Allergen":

                st.markdown(
                    """
                    <div class="result-title"
                    style="color:#a34c35;">
                        Allergen class predicted
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-title"
                    style="color:#087f78;">
                        Non-allergen class predicted
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write(
                f"The trained 3D CNN assigned this structure "
                f"to the **{prediction}** class using the "
                f"current 0.50 decision threshold."
            )

            st.caption(
                "This is an experimental model output, "
                "not a validated clinical or food-safety conclusion."
            )

        with col2:

            st.metric(
                "Allergen Model Score",
                f"{allergen_score:.2f}%"
            )

            st.progress(
                float(
                    np.clip(
                        probability,
                        0,
                        1
                    )
                )
            )

            st.caption(
                f"Non-allergen model score: "
                f"{non_allergen_score:.2f}%"
            )

    # Structure processing information

    st.markdown(
        "### Structure Processing"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Supported Atoms",
        f"{result['total']:,}"
    )

    col2.metric(
        "Atoms Outside Grid",
        f"{result['outside']:,}"
    )

    retained = (

        100 * (
            result["total"] - result["outside"]
        ) / result["total"]

        if result["total"] > 0

        else 0

    )

    col3.metric(
        "Atoms Retained",
        f"{retained:.1f}%"
    )

    if result["outside"] > 0:

        st.warning(
            f"{result['outside']} supported atoms "
            f"fell outside the {GRID_SIZE}³ voxel grid. "
            "Interpret the prediction cautiously."
        )

    # Protein information

    with st.expander(
        "View Protein Information",
        expanded=False
    ):

        render_protein_summary(
            result["info"]
        )

        st.text_area(
            "Extracted Amino Acid Sequence",
            value=result["info"]["sequence"],
            height=150,
            disabled=True
        )

    # 3D structure preview

    with st.expander(
        "View 3D Protein Structure",
        expanded=False
    ):

        show_3d_structure(
            result["pdb_text"],
            "Cartoon"
        )

    # PDF report

    st.markdown(
        "### Download Analysis Report"
    )

    pdf_bytes = create_pdf(

        result["filename"],

        prediction,

        probability,

        result["info"],

        result["outside"],

        result["total"]

    )

    st.download_button(

        label="Download Protein Analysis Report",

        data=pdf_bytes,

        file_name=(
            "AllergenAI_"
            + result["filename"].rsplit(".", 1)[0]
            + "_report.pdf"
        ),

        mime="application/pdf",

        key="download_prediction_report"

    )

 
# ============================================================
# 18. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-card">

            <div class="brand-name">
                🧬 AllergenAI
            </div>

            <div class="brand-caption">
                PROTEIN INTELLIGENCE WORKSPACE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-label">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Prediction Lab",
            "Structure Explorer",
            "Protein Insights",
            "Model Evaluation",
            "About the Research"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="small-label">MODEL CONFIGURATION</div>',
        unsafe_allow_html=True
    )

    st.caption("Architecture · 3D CNN")

    st.caption("Input · PDB Structure")

    st.caption(
        f"Voxel Grid · {GRID_SIZE} × {GRID_SIZE} × {GRID_SIZE}"
    )

    st.caption(
        f"Compute · {'CUDA GPU' if DEVICE.type == 'cuda' else 'CPU'}"
    )

    st.divider()

    st.caption(
        "Experimental research prototype. "
        "Not for clinical or food-safety decisions."
    )


# ============================================================
# 19. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    show_header(
        "PROTEIN STRUCTURE INTELLIGENCE",
        "Explore protein allergenicity through 3D structure.",
        "AllergenAI is an experimental BioAI workspace "
        "for exploring protein structures, extracting "
        "molecular information and generating experimental "
        "allergen-class predictions using a 3D CNN."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="soft-card">

                <div class="section-kicker">
                    01 · INPUT
                </div>

                <h3>Protein Structure</h3>

                <p>
                    Upload a PDB file and validate
                    the structure before analysis.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="soft-card">

                <div class="section-kicker">
                    02 · REPRESENTATION
                </div>

                <h3>3D Voxel Encoding</h3>

                <p>
                    Represent supported atom coordinates
                    in a five-channel 3D grid.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="soft-card">

                <div class="section-kicker">
                    03 · OUTPUT
                </div>

                <h3>Model Prediction</h3>

                <p>
                    Explore the binary model output
                    and supporting structural information.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown(
        "### Explore the workspace"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="soft-card">

                <h3>🔬 Prediction Lab</h3>

                <p>
                    Run the trained 3D CNN on a PDB
                    structure, review its output,
                    inspect processing diagnostics
                    and export a PDF report.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="soft-card">

                <h3>🧬 Structure Explorer</h3>

                <p>
                    Inspect uploaded protein structures
                    interactively and switch between
                    molecular representations.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.info(
        "To begin, open Prediction Lab and upload "
        "a valid PDB file. Your trained model file "
        "must be present beside app.py."
    )


# ============================================================
# 20. PREDICTION LAB
# ============================================================

elif page == "Prediction Lab":

    show_header(
        "ANALYSIS WORKSPACE",
        "Prediction Lab",
        "Upload a protein structure, validate its "
        "contents and run the trained 3D CNN classifier."
    )

    uploaded = render_upload(
        "Upload a Protein PDB File",
        "prediction_upload"
    )

    if uploaded is not None:

        st.write("")

        with st.container(border=True):

            st.markdown(
                "#### Ready for Analysis"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "File",
                uploaded["name"]
            )

            col2.metric(
                "File Size",
                f"{len(uploaded['bytes']) / 1024:.1f} KB"
            )

            col3.metric(
                "Structure Status",
                "Validated"
            )

            if st.button(
                "Run Allergen Prediction",
                type="primary",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Encoding structure and running 3D CNN..."
                    ):

                        (
                            prediction,
                            probability,
                            outside,
                            total
                        ) = predict_protein(
                            uploaded["text"]
                        )

                        info = get_protein_information(
                            uploaded["text"]
                        )

                    st.session_state["prediction_result"] = {

                        "signature": uploaded["signature"],

                        "filename": uploaded["name"],

                        "prediction": prediction,

                        "probability": probability,

                        "outside": outside,

                        "total": total,

                        "info": info,

                        "pdb_text": uploaded["text"]

                    }

                    st.success(
                        "Prediction completed successfully."
                    )

                except FileNotFoundError as exc:

                    st.error(str(exc))

                except Exception as exc:

                    st.error(
                        f"Prediction failed: {exc}"
                    )

    result = st.session_state.get(
        "prediction_result"
    )

    if uploaded is not None and result is not None:

        if result.get("signature") == uploaded["signature"]:

            st.write("")

            render_result(result)

        else:

            st.info(
                "A different PDB file is selected. "
                "Run prediction again to generate "
                "a result for this structure."
            )


# ============================================================
# 21. STRUCTURE EXPLORER
# ============================================================

elif page == "Structure Explorer":

    show_header(
        "MOLECULAR VIEWER",
        "Structure Explorer",
        "Explore the three-dimensional arrangement "
        "of atoms and chains in a PDB structure."
    )

    uploaded = render_upload(
        "Choose a PDB Structure",
        "viewer_upload"
    )

    if uploaded is not None:

        col1, col2 = st.columns([1, 2])

        with col1:

            style = st.selectbox(
                "Molecular Representation",
                [
                    "Cartoon",
                    "Stick",
                    "Sphere",
                    "Surface"
                ]
            )

        with col2:

            st.caption(
                "Drag to rotate, scroll to zoom, "
                "and use the representation selector "
                "to change the molecular view."
            )

        with st.container(border=True):

            show_3d_structure(
                uploaded["text"],
                style
            )

        info = get_protein_information(
            uploaded["text"]
        )

        st.markdown(
            "### Structure Overview"
        )

        render_protein_summary(info)


# ============================================================
# 22. PROTEIN INSIGHTS
# ============================================================

elif page == "Protein Insights":

    show_header(
        "STRUCTURAL CHARACTERIZATION",
        "Protein Insights",
        "Extract protein sequence information, "
        "chain counts, molecular weight and "
        "elemental composition from a PDB structure."
    )

    uploaded = render_upload(
        "Upload a PDB File for Analysis",
        "analysis_upload"
    )

    if uploaded is not None:

        try:

            info = get_protein_information(
                uploaded["text"]
            )

            st.markdown(
                "### Protein Overview"
            )

            render_protein_summary(info)

            st.write("")

            st.markdown(
                "### Amino Acid Sequence"
            )

            st.text_area(
                "Extracted Protein Sequence",
                value=info["sequence"],
                height=170,
                disabled=True
            )

            st.caption(
                f"Sequence Length: {len(info['sequence']):,} residues"
            )

            st.caption(
                "Chains: "
                + (
                    ", ".join(info["chains"])
                    if info["chains"]
                    else "None"
                )
            )

            st.divider()

            st.markdown(
                "### Molecular Composition"
            )

            render_composition(info)

            st.divider()

            st.markdown(
                "### Chain-Level Sequences"
            )

            for chain_id, sequence in info["sequences"].items():

                with st.expander(
                    f"Chain {chain_id} · {len(sequence)} residues"
                ):

                    st.code(
                        sequence or "No standard amino-acid sequence."
                    )

        except Exception as exc:

            st.error(
                f"Protein analysis failed: {exc}"
            )


# ============================================================
# 23. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    show_header(
        "MODEL TRANSPARENCY",
        "Model Evaluation",
        "Review the evaluation metrics saved from "
        "the trained 3D CNN. Only values present "
        "in your supplied metrics file are displayed."
    )

    if not METRICS_PATH.exists():

        st.warning(
            f"Evaluation file not found: {METRICS_PATH.name}. "
            "Place the CSV in the same folder as app.py."
        )

    else:

        try:

            metrics = pd.read_csv(
                METRICS_PATH
            )

            if metrics.empty:

                st.info(
                    "The evaluation metrics CSV is empty."
                )

            else:

                st.markdown(
                    "### Evaluation Data"
                )

                st.dataframe(
                    metrics,
                    use_container_width=True,
                    hide_index=True
                )

                normalized = {

                    str(col).strip().lower(): col

                    for col in metrics.columns

                }

                metric_col = next(

                    (
                        normalized[key]

                        for key in (
                            "metric",
                            "metrics",
                            "evaluation_metric",
                            "name"
                        )

                        if key in normalized

                    ),

                    None

                )

                value_col = next(

                    (
                        normalized[key]

                        for key in (
                            "value",
                            "score",
                            "metric_value"
                        )

                        if key in normalized

                    ),

                    None

                )

                if metric_col is not None and value_col is not None:

                    plot_df = metrics[
                        [metric_col, value_col]
                    ].copy()

                    plot_df[value_col] = pd.to_numeric(
                        plot_df[value_col],
                        errors="coerce"
                    )

                    plot_df = plot_df.dropna(
                        subset=[value_col]
                    )

                    if not plot_df.empty:

                        st.markdown(
                            "### Metric Visualization"
                        )

                        st.bar_chart(
                            plot_df.set_index(metric_col),
                            height=320
                        )

                else:

                    st.info(
                        "A chart was not generated because "
                        "the CSV does not contain recognizable "
                        "metric-name and value columns."
                    )

        except Exception as exc:

            st.error(
                f"Could not read evaluation metrics: {exc}"
            )

    st.divider()

    st.markdown(
        "### Model Configuration"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Architecture",
        "3D CNN"
    )

    col2.metric(
        "Input Grid",
        f"{GRID_SIZE}³"
    )

    col3.metric(
        "Input Channels",
        len(ELEMENTS)
    )

    st.caption(
        "Evaluation results are taken directly from "
        "the supplied CSV. Missing values are not assumed."
    )


# ============================================================
# 24. ABOUT THE RESEARCH
# ============================================================

elif page == "About the Research":

    show_header(
        "PROJECT INFORMATION",
        "About AllergenAI",
        "A research prototype exploring three-dimensional "
        "protein structural information for binary "
        "allergen and non-allergen classification."
    )

    st.markdown(
        "### Research Objective"
    )

    st.write(
        """
        The objective of AllergenAI is to explore whether
        three-dimensional protein structural representations
        can support allergen and non-allergen classification
        using deep learning.
        """
    )

    st.divider()

    st.markdown(
        "### Computational Workflow"
    )

    steps = [

        (
            "01",
            "PDB Input",
            "A protein structure is supplied in PDB format."
        ),

        (
            "02",
            "Atom Extraction",
            "Coordinates for C, N, O, S and P atoms are extracted."
        ),

        (
            "03",
            "Voxel Representation",
            "Coordinates are centered and mapped to a 48 × 48 × 48 grid."
        ),

        (
            "04",
            "3D CNN",
            "The voxel tensor is passed through the trained 3D CNN."
        ),

        (
            "05",
            "Binary Output",
            "The sigmoid score is thresholded at 0.50 for the two classes."
        )

    ]

    for number, title, description in steps:

        with st.container(border=True):

            col1, col2 = st.columns([0.16, 1])

            with col1:

                st.markdown(
                    f"### {number}"
                )

            with col2:

                st.markdown(
                    f"**{title}**"
                )

                st.caption(
                    description
                )

    st.divider()

    st.markdown(
        "### Model Architecture"
    )

    st.write(
        """
        The current implementation contains three 3D
        convolutional blocks, batch normalization,
        ReLU activations, max pooling, adaptive average
        pooling and fully connected classification layers.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Voxel Grid",
        "48 × 48 × 48"
    )

    col2.metric(
        "Voxel Size",
        "2.5 Å"
    )

    col3.metric(
        "Atom Channels",
        "5"
    )

    st.caption(
        "Channels: Carbon · Nitrogen · Oxygen · Sulfur · Phosphorus"
    )

    st.divider()

    st.markdown(
        "### Scope and Limitations"
    )

    st.warning(
        """
        The current application performs whole-protein
        binary classification. It does not currently
        identify, localize or visualize allergenic
        binding pockets. Pocket detection would require
        a separate computational method and validation.
        """
    )

    st.info(
        """
        AllergenAI is an experimental research application.
        Its outputs should not be used as a clinical diagnosis,
        food-safety decision, regulatory classification or
        substitute for experimental allergenicity testing.
        """
    )


# ============================================================
# 25. APPLICATION FOOTER
# ============================================================

st.markdown(
    """
    <hr>

    <div style="
        text-align:center;
        padding:12px 0 2px 0;
    ">

        <div style="
            font-family:Manrope,sans-serif;
            color:#087f78;
            font-weight:800;
        ">
            🧬 AllergenAI
        </div>

        <div style="
            font-size:11px;
            color:#809396;
            margin-top:4px;
        ">
            Protein Structure Intelligence
            · Experimental BioAI Research Application
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


