import os
import html
import streamlit as st
import streamlit.components.v1 as components    
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="SmartBin: Smart Waste Detection",
    layout="wide",
    page_icon="♻️",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS (LIGHT MODE DEFAULT)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(76, 175, 80, 0.22), transparent 35%),
            radial-gradient(circle at bottom right, rgba(255, 193, 7, 0.18), transparent 35%),
            linear-gradient(135deg, #f1f8e9 0%, #ffffff 45%, #e8f5e9 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .top-accent {
        height: 7px;
        width: 100%;
        background: linear-gradient(90deg, #43A047 0% 33%, #FFC107 33% 66%, #E53935 66% 100%);
        position: fixed;
        top: 0;
        left: 0;
        z-index: 999999;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dcedc8 0%, #f1f8e9 100%);
        border-right: 1px solid rgba(46, 125, 50, 0.12);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #1B5E20 !important;
    }

    .hero-card {
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(76, 175, 80, 0.18);
        border-radius: 28px;
        padding: 34px 38px;
        box-shadow: 0 18px 45px rgba(46, 125, 50, 0.12);
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        right: -80px;
        top: -80px;
        width: 240px;
        height: 240px;
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.24), rgba(255, 193, 7, 0.18));
        border-radius: 50%;
    }

    .hero-title {
        color: #1B5E20 !important;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #2E7D32 !important;
        font-size: 17px;
        line-height: 1.7;
        max-width: 760px;
    }

    .tag-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .tag {
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid rgba(46, 125, 50, 0.14);
    }

    .tag-green {
        background: #E8F5E9;
        color: #1B5E20;
    }

    .tag-yellow {
        background: #FFF8E1;
        color: #795548;
    }

    .tag-red {
        background: #FFEBEE;
        color: #B71C1C;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(76, 175, 80, 0.16);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 10px 28px rgba(46, 125, 50, 0.09);
        margin-bottom: 18px;
    }

    .section-title {
        color: #1B5E20 !important;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #558B2F !important;
        font-size: 14px;
        margin-top: -6px;
        margin-bottom: 16px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #43A047, #2E7D32) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 22px rgba(46, 125, 50, 0.22) !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 26px rgba(46, 125, 50, 0.28) !important;
        background: linear-gradient(135deg, #388E3C, #1B5E20) !important;
    }

    .stButton > button p,
    .stButton > button span {
        color: white !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(232, 245, 233, 0.55);
        border: 2px dashed rgba(76, 175, 80, 0.35);
        border-radius: 18px;
        padding: 14px;
    }

    /* =======================================
       PERBAIKAN UPLOADER LIGHT MODE (EDITED)
       ======================================= */
    /* Munculkan icon svg (awan/gambar) */
    [data-testid="stFileUploader"] svg,
    [data-testid="stFileUploaderDropzone"] svg {
        color: #FFFFFF !important;
        fill: currentColor !important;
    }
    
    /* Box file setelah upload */
    [data-testid="stUploadedFile"] {
        background-color: #1B5E20 !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* Font nama file dan icon svg jadi hijau */
    [data-testid="stUploadedFile"] *,
    [data-testid="stUploadedFile"] svg {
        color: #FFFFFF !important;
        fill: currentColor !important;
    }
    
    /* Tombol X (hapus file) transparan + efek hover rapi */
    [data-testid="stUploadedFile"] button {
        background-color: #1B5E20 !important;
        color: #FFFFFF !important;
        border: 2px solid #1B5E20 !important;
        border-radius: 50% !important;
    }
    [data-testid="stUploadedFile"] button:hover {
        background-color: rgba(76, 175, 80, 0.15) !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(76, 175, 80, 0.15);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.08);
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div {
        color: #1B5E20 !important;
    }

    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-top: 12px;
    }

    .result-card {
        min-height: 180px;
        border-radius: 22px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 12px 28px rgba(46, 125, 50, 0.10);
        border: 1px solid rgba(76, 175, 80, 0.16);
        background: rgba(255, 255, 255, 0.92);
    }

    .result-card h4 {
        color: #1B5E20 !important;
        font-size: 22px;
        font-weight: 800;
        margin: 0 0 18px 0;
    }

    .result-card p,
    .result-card b {
        color: #1B5E20 !important;
        font-size: 16px;
        line-height: 1.6;
    }

    .result-organik {
        background: linear-gradient(135deg, #E8F5E9, #FFFFFF);
        border-left: 10px solid #43A047;
    }

    .result-anorganik {
        background: linear-gradient(135deg, #FFF8E1, #FFFFFF);
        border-left: 10px solid #FFC107;
    }

    .result-b3 {
        background: linear-gradient(135deg, #FFEBEE, #FFFFFF);
        border-left: 10px solid #E53935;
    }

    .result-unknown {
        background: linear-gradient(135deg, #ECEFF1, #FFFFFF);
        border-left: 10px solid #78909C;
    }

    .trash-container {
        display: flex;
        justify-content: center;
        gap: 36px;
        flex-wrap: wrap;
        margin: 16px 0 4px 0;
        padding: 28px;
        background:
            linear-gradient(135deg, rgba(232,245,233,0.85), rgba(255,255,255,0.95));
        border-radius: 26px;
        border: 1px solid rgba(76, 175, 80, 0.18);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
    }

    .trashcan {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 135px;
        padding: 18px 12px;
        border-radius: 22px;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(76, 175, 80, 0.12);
        transition: all 0.25s ease;
    }

    .trashcan.open {
        transform: translateY(-6px);
        box-shadow: 0 14px 28px rgba(46, 125, 50, 0.15);
    }

    .trashcan svg {
        width: 88px;
        height: 110px;
        overflow: visible;
        filter: drop-shadow(0 8px 10px rgba(0,0,0,0.14));
    }

    .lid {
        transition: transform 0.45s cubic-bezier(0.34, 1.56, 0.64, 1);
        transform-origin: 20px 25px;
    }

    .trashcan.open .lid {
        transform: rotate(-45deg) translate(-5px, -10px);
    }

    .bin-label {
        font-weight: 800;
        font-size: 15px;
        margin-top: 12px;
        text-align: center;
        padding: 7px 14px;
        border-radius: 999px;
        color: white !important;
        min-width: 106px;
    }

    .hijau svg { color: #43A047; }
    .hijau .bin-label { background: #43A047; }

    .kuning svg { color: #FFC107; }
    .kuning .bin-label {
        background: #FFC107;
        color: #3E2723 !important;
    }

    .merah svg { color: #E53935; }
    .merah .bin-label { background: #E53935; }

    .footer-note {
        text-align: center;
        color: #2E7D32 !important;
        font-size: 13px;
        margin-top: 30px;
        opacity: 0.82;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #1B5E20 !important;
    }
            
    /* =========================
    FORCE LIGHT MODE BASE
    Prevent mengikuti dark mode Windows/browser
    ========================= */

    :root {
        color-scheme: light !important;
    }

    html, body, .stApp {
        color-scheme: light !important;
        background-color: #F1F8E9 !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(76, 175, 80, 0.22), transparent 35%),
            radial-gradient(circle at bottom right, rgba(255, 193, 7, 0.18), transparent 35%),
            linear-gradient(135deg, #f1f8e9 0%, #ffffff 45%, #e8f5e9 100%) !important;
    }

    /* Force main text */
    h1, h2, h3, h4, h5, h6,
    p, span, label, li,
    div,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] *,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"] *,
    [data-testid="stSlider"] *,
    [data-testid="stMetric"] * {
        color: #1B5E20 !important;
    }

    /* Sidebar light */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dcedc8 0%, #f1f8e9 100%) !important;
        border-right: 1px solid rgba(46, 125, 50, 0.12) !important;
    }

    /* Cards light */
    .hero-card,
    .section-card,
    .result-card,
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(76, 175, 80, 0.16) !important;
        box-shadow: 0 10px 28px rgba(46, 125, 50, 0.09) !important;
    }

    /* Subtitle / muted text */
    .hero-subtitle,
    .small-muted,
    .footer-note,
    [data-testid="stImageCaption"],
    [data-testid="stImageCaption"] * {
        color: #2E7D32 !important;
    }

    /* File uploader light */
    [data-testid="stFileUploader"] {
        background: rgba(232, 245, 233, 0.55) !important;
        border: 2px dashed rgba(76, 175, 80, 0.35) !important;
        border-radius: 18px !important;
    }

    /* Inputs / radio / slider containers */
    input,
    textarea,
    select,
    [data-baseweb="select"],
    [data-baseweb="input"],
    [data-baseweb="textarea"] {
        color: #1B5E20 !important;
        background-color: #FFFFFF !important;
    }

    /* Alerts light */
    .stAlert {
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(76, 175, 80, 0.18) !important;
        border-radius: 16px !important;
    }

    .stAlert * {
        color: #1B5E20 !important;
    }

    /* Buttons stay white text */
    .stButton > button,
    .stButton > button *,
    .stButton > button p,
    .stButton > button span {
        color: white !important;
    }
            
    /* CATEGORY TEXT - LIGHT MODE */
    .hero-subtitle .cat-text {
        font-weight: 800 !important;
    }

    .hero-subtitle .cat-organik,
    .hero-subtitle .cat-organik *,
    .hero-subtitle .cat-organik b {
        color: #43A047 !important;
    }

    .hero-subtitle .cat-anorganik,
    .hero-subtitle .cat-anorganik *,
    .hero-subtitle .cat-anorganik b {
        color: #F9A825 !important;
    }

    .hero-subtitle .cat-b3,
    .hero-subtitle .cat-b3 *,
    .hero-subtitle .cat-b3 b {
        color: #E53935 !important;
    }
                
</style>

<div class="top-accent"></div>
""", unsafe_allow_html=True)

def apply_dark_theme():
    st.markdown("""
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(76, 175, 80, 0.20), transparent 35%),
                radial-gradient(circle at bottom right, rgba(255, 193, 7, 0.12), transparent 34%),
                linear-gradient(135deg, #07140B 0%, #0F2415 48%, #132B1A 100%) !important;
            color: #DDF8E4 !important;
        }

        .stApp *:not(button):not(button *) {
            color: #DDF8E4 !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07140B 0%, #102B18 100%) !important;
            border-right: 1px solid rgba(165, 214, 167, 0.22) !important;
        }

        .hero-card,
        .section-card,
        .result-card,
        [data-testid="stMetric"],
        [data-testid="stFileUploader"],
        [data-testid="stAlert"] {
            background: rgba(18, 38, 24, 0.92) !important;
            border-color: rgba(165, 214, 167, 0.24) !important;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.34) !important;
        }

        .hero-subtitle,
        .small-muted,
        .footer-note,
        [data-testid="stImageCaption"],
        [data-testid="stImageCaption"] * {
            color: #A5D6A7 !important;
        }

        .tag-green {
            background: rgba(76, 175, 80, 0.20) !important;
            color: #DDF8E4 !important;
        }

        .tag-yellow {
            background: rgba(255, 193, 7, 0.20) !important;
            color: #FFF8E1 !important;
        }

        .tag-red {
            background: rgba(229, 57, 53, 0.20) !important;
            color: #FFCDD2 !important;
        }

        .result-organik {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.24), rgba(18, 38, 24, 0.96)) !important;
            border-left: 10px solid #43A047 !important;
        }

        .result-anorganik {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.24), rgba(18, 38, 24, 0.96)) !important;
            border-left: 10px solid #FFC107 !important;
        }

        .result-b3 {
            background: linear-gradient(135deg, rgba(229, 57, 53, 0.24), rgba(18, 38, 24, 0.96)) !important;
            border-left: 10px solid #E53935 !important;
        }

        .stButton > button,
        .stButton > button *,
        .stButton > button p,
        .stButton > button span {
            color: white !important;
        }

        input,
        textarea,
        select {
            background-color: rgba(18, 38, 24, 0.95) !important;
            color: #DDF8E4 !important;
        }
                
        /* =========================
           FIX STREAMLIT HEADER / TOP BAR
        ========================= */
        header[data-testid="stHeader"],
        [data-testid="stHeader"] {
            background: #07140B !important;
            color: #DDF8E4 !important;
            border-bottom: 1px solid rgba(165, 214, 167, 0.22) !important;
        }

        [data-testid="stToolbar"],
        [data-testid="stToolbar"] *,
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stStatusWidget"] *,
        .stDeployButton,
        .stDeployButton *,
        .stDeployButton button,
        .stDeployButton button * {
            color: #DDF8E4 !important;
            background-color: transparent !important;
        }

        .stDeployButton button {
            border: 1px solid rgba(165, 214, 167, 0.28) !important;
            background: rgba(18, 38, 24, 0.85) !important;
            color: #DDF8E4 !important;
        }
        
        /* =========================
           FIX FILE UPLOADER WHITE AREA
        ========================= */
        [data-testid="stFileUploader"] {
            background: rgba(18, 38, 24, 0.92) !important;
            border: 1px solid rgba(165, 214, 167, 0.24) !important;
            border-radius: 18px !important;
            padding: 14px !important;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(18, 38, 24, 0.95) !important;
            color: #DDF8E4 !important;
            border-color: rgba(165, 214, 167, 0.32) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed rgba(165, 214, 167, 0.32) !important;
            border-radius: 16px !important;
        }

        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {
            color: #DDF8E4 !important;
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button {
            background: rgba(76, 175, 80, 0.18) !important;
            color: #DDF8E4 !important;
            border: 1px solid rgba(165, 214, 167, 0.36) !important;
            border-radius: 10px !important;
        }
        
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploaderDropzone"] button * {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #DDF8E4 !important;
        }
        
        [data-testid="stFileUploader"] button svg,
        [data-testid="stFileUploaderDropzone"] button svg {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #DDF8E4 !important;
        }

        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: rgba(76, 175, 80, 0.32) !important;
            color: #FFFFFF !important;
        }

        /* =======================================
           PERBAIKAN UPLOADER DARK MODE (EDITED)
           ======================================= */
        /* Kotak file dibuat terang agar tulisan hijau gelap terlihat */
        [data-testid="stUploadedFile"] {
            background-color: #E8F5E9 !important; 
            border: 1px solid rgba(165, 214, 167, 0.24) !important;
            border-radius: 12px !important;
        }

        /* Paksa font dan svg jadi hijau gelap (ijo gelap jangan terang) */
        [data-testid="stUploadedFile"] *,
        [data-testid="stUploadedFile"] svg {
            color: #1B5E20 !important;
            fill: currentColor !important;
        }

        /* Tombol silang transparan dengan efek hover rapi */
        [data-testid="stUploadedFile"] button {
            background-color: transparent !important;
            border: none !important;
            border-radius: 50% !important;
        }
        [data-testid="stUploadedFile"] button:hover {
            background-color: rgba(76, 175, 80, 0.2) !important;
        }

        /* =========================
           FIX EMPTY / WHITE STREAMLIT BLOCKS
        ========================= */
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="column"] {
            background: transparent !important;
        }

        /* =========================
           FINAL FORCE TEXT DARK MODE
        ========================= */
        .stApp *:not(button):not(button *) {
            color: #DDF8E4 !important;
        }

        .stButton > button,
        .stButton > button *,
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button * {
            color: #DDF8E4 !important;
        }
                
        /* CATEGORY TEXT - DARK MODE */
        .hero-subtitle .cat-organik,
        .hero-subtitle .cat-organik *,
        .hero-subtitle .cat-organik b {
            color: #81C784 !important;
        }

        .hero-subtitle .cat-anorganik,
        .hero-subtitle .cat-anorganik *,
        .hero-subtitle .cat-anorganik b {
            color: #FFD54F !important;
        }

        .hero-subtitle .cat-b3,
        .hero-subtitle .cat-b3 *,
        .hero-subtitle .cat-b3 b {
            color: #EF9A9A !important;
        }
                
    </style>
    """, unsafe_allow_html=True)

# =========================
# HELPER FUNCTION
# =========================
def render_header():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">♻️ SMARTBIN</div>
        <div class="hero-subtitle">
            <b>AI-Powered Waste Classification System</b><br>
            Mendeteksi jenis sampah menggunakan model YOLO dan memberikan rekomendasi tempat pembuangan:
            <span class="cat-text cat-organik"><b>Organik</b></span>,
            <span class="cat-text cat-anorganik"><b>Anorganik</b></span>,
            <span class="cat-text cat-b3"><b>B3</b></span>.
        </div>
        <div class="tag-row">
            <div class="tag tag-green">🌿 Organic Waste</div>
            <div class="tag tag-yellow">🥫 Inorganic Waste</div>
            <div class="tag tag-red">⚠️ Hazardous Waste</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def normalize_class_name(class_name):
    """
    Menyamakan nama kelas supaya animasi tong tetap jalan
    meskipun label model berbeda kapitalisasi.
    """
    name = str(class_name).lower().strip()

    if "anorganik" in name or "non-organik" in name or "non organic" in name:
        return "anorganik"
    elif "organik" in name or "organic" in name:
        return "organik"
    elif "b3" in name or "hazard" in name or "berbahaya" in name:
        return "b3"
    else:
        return name


def get_disposal_recommendation(class_name):
    normalized = normalize_class_name(class_name)

    if normalized == "organik":
        return "Buang ke tempat sampah organik / bisa dikomposkan."
    elif normalized == "anorganik":
        return "Buang ke tempat sampah anorganik / dapat dipilah untuk daur ulang."
    elif normalized == "b3":
        return "Buang ke tempat khusus B3, jangan dicampur dengan sampah lain."
    else:
        return "Periksa ulang kategori sampah sebelum dibuang."

def get_card_class(class_name):
    normalized = normalize_class_name(class_name)

    if normalized == "organik":
        return "result-organik"
    elif normalized == "anorganik":
        return "result-anorganik"
    elif normalized == "b3":
        return "result-b3"
    else:
        return "result-unknown"

def render_trashcans(detected_classes, theme_mode):
    classes = {normalize_class_name(cls) for cls in detected_classes}

    org_state = "open" if "organik" in classes else ""
    anorg_state = "open" if "anorganik" in classes else ""
    b3_state = "open" if "b3" in classes else ""

    # Logika tema dipindah ke Python agar CSS yang di-generate akurat
    if theme_mode == "Dark":
        bin_container_bg = "linear-gradient(135deg, #07140B, #132B1A)"
        bin_card_bg = "rgba(24, 50, 31, 0.94)"
        bin_border = "rgba(165, 214, 167, 0.24)"
        bin_shadow = "0 14px 28px rgba(0, 0, 0, 0.34)"
        label_text_green = "#F1FFF4"
        label_text_yellow = "#3E2723"
        label_text_red = "#FFF5F5"
    else:
        bin_container_bg = "linear-gradient(135deg, #E8F5E9, #FFFFFF)"
        bin_card_bg = "rgba(255, 255, 255, 0.92)"
        bin_border = "rgba(76, 175, 80, 0.25)"
        bin_shadow = "0 10px 25px rgba(46, 125, 50, 0.10)"
        label_text_green = "#FFFFFF"
        label_text_yellow = "#3E2723"
        label_text_red = "#FFFFFF"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Arial, sans-serif;
        }}

        /* Hanya render satu :root sesuai tema yang sedang aktif */
        :root {{
            --bin-container-bg: {bin_container_bg};
            --bin-card-bg: {bin_card_bg};
            --bin-border: {bin_border};
            --bin-shadow: {bin_shadow};

            --label-text-green: {label_text_green};
            --label-text-yellow: {label_text_yellow};
            --label-text-red: {label_text_red};
        }}

        .trash-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 36px;
            flex-wrap: wrap;
            padding: 28px;
            background: var(--bin-container-bg);
            border-radius: 26px;
            border: 1px solid var(--bin-border);
            box-shadow: var(--bin-shadow);
        }}

        .trashcan {{
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 135px;
            padding: 18px 12px;
            border-radius: 22px;
            background: var(--bin-card-bg);
            border: 1px solid var(--bin-border);
            transition: all 0.25s ease;
        }}

        .trashcan.open {{
            transform: translateY(-6px);
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
        }}

        .trashcan svg {{
            width: 88px;
            height: 110px;
            overflow: visible;
            filter: drop-shadow(0 8px 10px rgba(0, 0, 0, 0.18));
        }}

        .lid {{
            transition: transform 0.45s cubic-bezier(0.34, 1.56, 0.64, 1);
            transform-origin: 20px 25px;
        }}

        .trashcan.open .lid {{
            transform: rotate(-45deg) translate(-5px, -10px);
        }}

        .bin-label {{
            font-weight: 800;
            font-size: 15px;
            margin-top: 12px;
            text-align: center;
            padding: 7px 14px;
            border-radius: 999px;
            min-width: 106px;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.12);
        }}

        .hijau svg {{ color: #43A047; }}
        .kuning svg {{ color: #FFC107; }}
        .merah svg {{ color: #E53935; }}

        .hijau .bin-label {{ background: #43A047; color: var(--label-text-green); }}
        .kuning .bin-label {{ background: #FFC107; color: var(--label-text-yellow); }}
        .merah .bin-label {{ background: #E53935; color: var(--label-text-red); }}
    </style>
    </head>
    <body>
        <div class="trash-container">
            <div class="trashcan hijau {org_state}">
                <svg viewBox="0 0 100 120">
                    <g class="lid" fill="currentColor">
                        <path d="M40 10 h20 v5 h-20 z"></path>
                        <path d="M20 15 h60 v10 h-60 z"></path>
                    </g>
                    <path d="M25 27 h50 l-5 80 h-40 z" fill="currentColor"></path>
                </svg>
                <div class="bin-label">Organik</div>
            </div>

            <div class="trashcan kuning {anorg_state}">
                <svg viewBox="0 0 100 120">
                    <g class="lid" fill="currentColor">
                        <path d="M40 10 h20 v5 h-20 z"></path>
                        <path d="M20 15 h60 v10 h-60 z"></path>
                    </g>
                    <path d="M25 27 h50 l-5 80 h-40 z" fill="currentColor"></path>
                </svg>
                <div class="bin-label">Anorganik</div>
            </div>

            <div class="trashcan merah {b3_state}">
                <svg viewBox="0 0 100 120">
                    <g class="lid" fill="currentColor">
                        <path d="M40 10 h20 v5 h-20 z"></path>
                        <path d="M20 15 h60 v10 h-60 z"></path>
                    </g>
                    <path d="M25 27 h50 l-5 80 h-40 z" fill="currentColor"></path>
                </svg>
                <div class="bin-label">B3</div>
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html_code, height=285, scrolling=False)


def render_detection_cards(detections, theme_mode):
    if not detections:
        st.warning("⚠️ Tidak ada objek sampah yang terdeteksi. Silakan coba gambar lain atau turunkan confidence threshold.")
        return

    is_dark = theme_mode == "Dark"
    max_cols = 3

    for i in range(0, len(detections), max_cols):
        row_detections = detections[i:i + max_cols]
        cols = st.columns(len(row_detections))

        for col, det in zip(cols, row_detections):
            tipe = html.escape(det["Tipe Sampah"])
            akurasi = html.escape(det["Akurasi"])
            rekomendasi = html.escape(det["Rekomendasi"])
            
            normalized = normalize_class_name(tipe)
            
            # Tentukan warna inline berdasarkan tema agar terbebas dari bug caching CSS global
            if is_dark:
                text_color = "#DDF8E4"
                border_col = "rgba(165, 214, 167, 0.24)"
                if normalized == "organik":
                    bg_grad = "linear-gradient(135deg, rgba(76, 175, 80, 0.24), rgba(18, 38, 24, 0.96))"
                    border_left = "#43A047"
                elif normalized == "anorganik":
                    bg_grad = "linear-gradient(135deg, rgba(255, 193, 7, 0.24), rgba(18, 38, 24, 0.96))"
                    border_left = "#FFC107"
                elif normalized == "b3":
                    bg_grad = "linear-gradient(135deg, rgba(229, 57, 53, 0.24), rgba(18, 38, 24, 0.96))"
                    border_left = "#E53935"
                else:
                    bg_grad = "linear-gradient(135deg, rgba(120, 144, 156, 0.22), rgba(18, 38, 24, 0.96))"
                    border_left = "#78909C"
            else:
                text_color = "#1B5E20"
                border_col = "rgba(76, 175, 80, 0.16)"
                if normalized == "organik":
                    bg_grad = "linear-gradient(135deg, #E8F5E9, #FFFFFF)"
                    border_left = "#43A047"
                elif normalized == "anorganik":
                    bg_grad = "linear-gradient(135deg, #FFF8E1, #FFFFFF)"
                    border_left = "#FFC107"
                elif normalized == "b3":
                    bg_grad = "linear-gradient(135deg, #FFEBEE, #FFFFFF)"
                    border_left = "#E53935"
                else:
                    bg_grad = "linear-gradient(135deg, #ECEFF1, #FFFFFF)"
                    border_left = "#78909C"

            # Bangun card dengan style inline agar anti gagal
            card_style = (
                f"background: {bg_grad}; "
                f"border: 1px solid {border_col}; "
                f"border-left: 10px solid {border_left}; "
                "min-height: 180px; border-radius: 22px; padding: 24px; margin-bottom: 18px; "
                "box-shadow: 0 12px 28px rgba(0,0,0,0.05);"
            )

            card_html = (
                f'<div style="{card_style}">'
                f'<h4 style="color: {text_color}; font-size: 22px; font-weight: 800; margin: 0 0 18px 0;">{tipe}</h4>'
                f'<p style="color: {text_color}; font-size: 16px; margin: 0;"><b>Akurasi:</b> {akurasi}</p>'
                f'<p style="color: {text_color}; font-size: 16px; margin-top: 8px;">{rekomendasi}</p>'
                f'</div>'
            )

            with col:
                st.markdown(card_html, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = "best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError("File best.pt tidak ditemukan. Pastikan best.pt berada satu folder dengan app.py.")

    return YOLO(model_path)


def predict_image(image, model, conf_threshold):
    image_np = np.array(image)

    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]

    annotated_img = result.plot()

    detections = []
    detected_classes = set()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        detected_classes.add(class_name)

        detections.append({
            "Tipe Sampah": class_name,
            "Akurasi": f"{conf * 100:.2f}%",
            "Confidence": conf,
            "Rekomendasi": get_disposal_recommendation(class_name)
        })

    detections = sorted(detections, key=lambda x: x["Confidence"], reverse=True)

    for det in detections:
        det.pop("Confidence", None)

    return annotated_img, detections, list(detected_classes)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    theme_mode = st.radio(
        "Tema tampilan:",
        ["Light", "Dark"],
        index=0,
        horizontal=True
    )

    st.markdown("Atur sumber gambar dan batas minimum confidence model.")

    mode = st.radio(
        "Pilih sumber gambar:",
        ["📤 Upload Gambar", "📸 Webcam"],
        index=0
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05
    )

    st.divider()

    st.markdown("### 💡 Tips")
    st.markdown("""
    - Gunakan gambar yang jelas.
    - Usahakan objek sampah terlihat penuh.
    - Background polos akan membantu deteksi.
    - Jika tidak terdeteksi, turunkan threshold.
    """)

    st.divider()

    st.markdown("### 📁 Model")
    st.success("Model `best.pt` berhasil dimuat.")

if theme_mode == "Dark":
    apply_dark_theme()


# =========================
# MAIN APP
# =========================
render_header()

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# =========================
# UPLOAD MODE
# =========================
if mode == "📤 Upload Gambar":
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📷 Input Gambar</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Upload gambar sampah dalam format JPG, JPEG, atau PNG.</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload gambar sampah",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        image = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar Asli", use_container_width=True)

        analyze_button = st.button("🔍 Deteksi Gambar", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Hasil Deteksi</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Hasil bounding box dan klasifikasi akan muncul di sini.</div>', unsafe_allow_html=True)

        if uploaded_file is None:
            st.info("Silakan upload gambar terlebih dahulu.")
        elif analyze_button:
            with st.spinner("Mendeteksi sampah..."):
                annotated_img, detections, detected_classes = predict_image(
                    image=image,
                    model=model,
                    conf_threshold=conf_threshold
                )

            total_detected = len(detections)
            highest_acc = detections[0]["Akurasi"] if detections else "-"

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Objek Terdeteksi", total_detected)
            metric_col2.metric("Akurasi Tertinggi", highest_acc)

            st.image(annotated_img, caption="Gambar Hasil Deteksi", use_container_width=True)
            
        else:
            st.info("Klik tombol **Deteksi Gambar** untuk mulai analisis.")

        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None and analyze_button:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗑️ Rekomendasi Pembuangan</div>', unsafe_allow_html=True)
        render_trashcans(detected_classes, theme_mode)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Detail Klasifikasi</div>', unsafe_allow_html=True)
        render_detection_cards(detections, theme_mode)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================
# WEBCAM MODE
# =========================
elif mode == "📸 Webcam":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Webcam Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Ambil gambar langsung melalui kamera. Pastikan browser mengizinkan akses kamera.</div>', unsafe_allow_html=True)

    camera_image = st.camera_input("Ambil gambar dari webcam")

    st.markdown('</div>', unsafe_allow_html=True)

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

        with st.spinner("Mendeteksi sampah..."):
            annotated_img, detections, detected_classes = predict_image(
                image=image,
                model=model,
                conf_threshold=conf_threshold
            )

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Hasil Deteksi</div>', unsafe_allow_html=True)
            st.image(annotated_img, caption="Gambar Hasil Deteksi", use_container_width=True)

            total_detected = len(detections)
            highest_acc = detections[0]["Akurasi"] if detections else "-"

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Objek Terdeteksi", total_detected)
            metric_col2.metric("Akurasi Tertinggi", highest_acc)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Detail Klasifikasi</div>', unsafe_allow_html=True)
            render_detection_cards(detections, theme_mode)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗑️ Rekomendasi Pembuangan</div>', unsafe_allow_html=True)
        render_trashcans(detected_classes, theme_mode)
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
<div class="footer-note">
    Dibuat untuk klasifikasi sampah berbasis YOLO: Organik, Anorganik, dan B3.
</div>
""", unsafe_allow_html=True)