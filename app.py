# =============================================================
#  SkinScan: phân tích da, routine & gợi ý sản phẩm
#  Cần các file nằm chung 1 thư mục:
#    app.py, skin_model.keras, class_names.json, products.csv,
#    requirements.txt, .streamlit/config.toml
# =============================================================
import html
import json
import unicodedata
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# -------------------------------------------------------------
# 👉 SỬA THÔNG TIN CỦA BẠN Ở ĐÂY
# -------------------------------------------------------------
APP_NAME = "SkinScan"
TAGLINE = "Phân tích da · Gợi ý sản phẩm có bán tại Guardian"
CONTACT = {
    "name": "4 PINK HAT HACKERS",
    "school": "TỔ CHỨC PINK HAT HACKER",
    "email": "pinkhathacker@example.com",
    "phone": "09xx xxx xxx",
    "facebook": "https://facebook.com/",
    "address": "TP. Hồ Chí Minh, Việt Nam",
}
DATA_DATE = "11/2026"      # thời điểm lấy dữ liệu sản phẩm
FONT_NAME = "Plus Jakarta Sans"      # tên font trên Google Fonts, vd "Roboto", "Open Sans", "Montserrat"

# Màu lấy từ trang Guardian
ORANGE = "#F6831F"
ORANGE_DARK = "#D96A0B"
INK = "#333333"
GREY = "#545453"
LINE = "#E6E6E6"
BG = "#FAFAFA"

IMG_SIZE = 224
THRESHOLD = 0.20
TOP_N = 8
MAX_PER_TYPE = 2

st.set_page_config(page_title=f"{APP_NAME} | Phân tích da", page_icon=":material/spa:", layout="wide")

# -------------------------------------------------------------
# Dữ liệu vấn đề da
# -------------------------------------------------------------
VI_NAMES = {
    "inflammatory acne": "Mụn viêm",
    "non inflammatory acne black heads": "Mụn đầu đen",
    "non inflammatory acne white heads": "Mụn đầu trắng",
    "pores": "Lỗ chân lông to",
    "pigmentation": "Sắc tố / nám",
    "dark spots": "Đốm thâm",
    "wrinkles": "Nếp nhăn",
    "redness": "Da ửng đỏ",
}
VI_TO_KEY = {v: k for k, v in VI_NAMES.items()}

KEYWORDS = {
    "inflammatory acne": ["salicylic", "bha", "benzoyl", "tea tree", "tràm trà", "azelaic",
                          "trị mụn", "giảm mụn", "ngừa mụn", "acne", "mụn"],
    "non inflammatory acne black heads": ["salicylic", "bha", "đất sét", "clay", "than hoạt tính",
                                          "charcoal", "mụn đầu đen", "blackhead", "lỗ chân lông"],
    "non inflammatory acne white heads": ["salicylic", "bha", "aha", "glycolic", "lactic",
                                          "tẩy tế bào chết", "tẩy da chết", "tế bào chết", "exfoliat", "mụn"],
    "pores": ["niacinamide", "bha", "salicylic", "đất sét", "clay", "lỗ chân lông", "pore", "da dầu",
              "kiềm dầu", "se khít"],
    "pigmentation": ["vitamin c", "niacinamide", "arbutin", "tranexamic", "sáng da", "dưỡng trắng",
                     "mờ nám", "nám", "brightening", "whitening", "dưỡng sáng", "nghệ", "chống nắng", "spf"],
    "dark spots": ["vitamin c", "niacinamide", "arbutin", "tranexamic", "mờ thâm", "thâm",
                   "sáng da", "dark spot", "chống nắng", "spf"],
    "wrinkles": ["retinol", "retinal", "peptide", "collagen", "bakuchiol", "lão hóa", "lão hoá",
                 "vitamin a", "vita-a", "săn chắc", "nâng cơ", "lifting", "firming",
                 "nếp nhăn", "anti-aging", "anti aging", "chống nắng", "spf"],
    "redness": ["rau má", "centella", "cica", "panthenol", "b5", "ceramide", "làm dịu", "dịu da",
                "barrier", "repair", "xịt khoáng", "phục hồi", "nhạy cảm", "soothing", "calming"],
}

ADVICE = {
    "inflammatory acne": "Ưu tiên BHA (salicylic acid) hoặc benzoyl peroxide, rửa mặt dịu nhẹ 2 lần/ngày. Mụn viêm nặng, đau nhức nên đi khám bác sĩ da liễu.",
    "non inflammatory acne black heads": "BHA giúp làm sạch sâu lỗ chân lông; có thể dùng mặt nạ đất sét 1–2 lần/tuần. Không nặn mụn bằng tay.",
    "non inflammatory acne white heads": "Tẩy tế bào chết hóa học (AHA/BHA) 2–3 lần/tuần giúp thông thoáng lỗ chân lông.",
    "pores": "Niacinamide và BHA giúp kiềm dầu và làm lỗ chân lông trông nhỏ hơn. Tẩy trang kỹ mỗi tối.",
    "pigmentation": "Vitamin C, niacinamide giúp làm sáng da. Kem chống nắng mỗi ngày là bước quan trọng nhất.",
    "dark spots": "Vitamin C, niacinamide, arbutin giúp mờ thâm dần sau 4–8 tuần. Luôn dùng kem chống nắng.",
    "wrinkles": "Retinol/retinal và peptide hỗ trợ chống lão hóa; dùng retinol buổi tối, chống nắng ban ngày.",
    "redness": "Chọn sản phẩm làm dịu, phục hồi hàng rào da (rau má, panthenol, ceramide); tránh cồn và hương liệu.",
}

SKIN_TYPES = {
    "Da dầu": ["da dầu", "kiềm dầu", "dầu thừa", "bã nhờn", "oil", "sebum", "gel", "foam", "bọt"],
    "Da khô": ["da khô", "dưỡng ẩm", "cấp ẩm", "hydrat", "moist", "ceramide", "hyaluronic", "sữa rửa mặt dịu"],
    "Da hỗn hợp": ["hỗn hợp", "cân bằng", "balanc", "kiềm dầu", "dưỡng ẩm"],
    "Da nhạy cảm": ["nhạy cảm", "sensitive", "dịu", "rau má", "centella", "cica", "b5", "panthenol"],
    "Da thường": [],
}

# Routine: (loại sản phẩm trong products.csv, tên bước)
ROUTINE = {
    "Đầy đủ": {
        "Sáng": [("Sữa rửa mặt", "Làm sạch"), ("Toner", "Cân bằng"), ("Serum", "Đặc trị"),
                 ("Kem dưỡng", "Dưỡng ẩm"), ("Chống nắng", "Chống nắng")],
        "Tối": [("Tẩy trang", "Tẩy trang"), ("Sữa rửa mặt", "Làm sạch"), ("Toner", "Cân bằng"),
                ("Serum", "Đặc trị"), ("Kem dưỡng", "Dưỡng ẩm")],
        "Hằng tuần": [("Tẩy tế bào chết", "Tẩy tế bào chết · 2–3 tối/tuần"), ("Mặt nạ", "Mặt nạ · 1–2 lần/tuần")],
    },
    "Cơ bản": {
        "Sáng": [("Sữa rửa mặt", "Làm sạch"), ("Kem dưỡng", "Dưỡng ẩm"), ("Chống nắng", "Chống nắng")],
        "Tối": [("Tẩy trang", "Tẩy trang"), ("Sữa rửa mặt", "Làm sạch"), ("Kem dưỡng", "Dưỡng ẩm")],
        "Hằng tuần": [],
    },
}
BUDGET_WEIGHT = {"Tẩy trang": .11, "Sữa rửa mặt": .14, "Toner": .12, "Serum": .25, "Kem dưỡng": .16,
                 "Chống nắng": .17, "Tẩy tế bào chết": .12, "Mặt nạ": .05}
BUDGETS = {"Tiết kiệm · dưới 500.000đ": 500_000, "Vừa phải · dưới 1.000.000đ": 1_000_000,
           "Thoải mái · dưới 2.000.000đ": 2_000_000}


def norm(text):
    return unicodedata.normalize("NFC", str(text)).lower()


def esc(text):
    return html.escape(str(text))


def fmt_price(x):
    try:
        return f"{int(float(x)):,}".replace(",", ".") + "đ"
    except (ValueError, TypeError):
        return ""


# -------------------------------------------------------------
# Biểu tượng nét mảnh (SVG), thay cho emoji
# -------------------------------------------------------------
_ICONS = {
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "scan": '<path d="M4 8V5h3M17 5h3v3M20 16v3h-3M7 19H4v-3"/><path d="M8 12h8"/>',
    "bag": '<path d="M5 8h14l-1 12H6z"/><path d="M9 8V6a3 3 0 0 1 6 0v2"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    "calendar": '<rect x="4" y="5" width="16" height="15" rx="1"/><path d="M4 10h16M9 3v4M15 3v4"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "mail": '<rect x="3" y="5" width="18" height="14" rx="1"/><path d="m3 7 9 6 9-6"/>',
    "phone": '<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a1 1 0 0 1-1 1A16 16 0 0 1 4 5a1 1 0 0 1 1-1z"/>',
    "pin": '<path d="M12 21s-7-6.5-7-12a7 7 0 0 1 14 0c0 5.5-7 12-7 12z"/><circle cx="12" cy="9" r="2.5"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "school": '<path d="m2 9 10-5 10 5-10 5z"/><path d="M6 11v5c3 2 9 2 12 0v-5"/>',
    "chat": '<path d="M4 5h16v11H9l-5 4z"/>',
    "lock": '<rect x="5" y="11" width="14" height="9" rx="1"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
    "arrow": '<path d="M7 17 17 7M9 7h8v8"/>',
}


def icon(name, size=18):
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
            f'{_ICONS[name]}</svg>')


# -------------------------------------------------------------
# CSS
# -------------------------------------------------------------
FONT_URL = f"https://fonts.googleapis.com/css2?family={FONT_NAME.replace(' ', '+')}:wght@400;500;700&display=swap"
CSS = f"""
<style>
@import url('{FONT_URL}');
html, body, .stApp, .stApp * {{ font-family: '{FONT_NAME}', Arial, sans-serif; }}
.stApp [data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {BG}; color: {INK}; }}
.block-container {{ padding-top: 5.2rem !important; max-width: 1200px; }}
footer, [data-testid="stDecoration"] {{ display: none !important; }}
h1, h2, h3, h4 {{ color: {INK} !important; font-weight: 700 !important; }}
p, label, span, div {{ color: inherit; }}

/* Ô nhập liệu luôn nền trắng, chữ đậm (kể cả khi máy để chế độ tối) */
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="textarea"] {{ background: #fff !important; color: {INK} !important; border-color: #CFCFCF !important; }}
.stApp input::placeholder, .stApp textarea::placeholder {{ color: #9A9A9A !important; }}
.stApp label, .stApp [data-testid="stWidgetLabel"] p {{ color: {INK} !important; font-weight: 500; }}
.stApp [data-testid="stForm"], .stApp [data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-color: {LINE} !important; border-radius: 6px; }}
.stApp [data-testid="stExpander"] details {{ background: #fff; border-color: {LINE}; border-radius: 6px; }}
.stApp [data-testid="stFileUploaderDropzone"] {{ background: #fff; border: 1px dashed #CFCFCF; }}
.stApp [data-testid="stFileUploaderDropzone"] * {{ color: {GREY} !important; }}
.stApp [data-baseweb="tag"] {{ background: #FFF1E4 !important; }}
.stApp [data-baseweb="tag"] span {{ color: {ORANGE_DARK} !important; }}

/* Nút */
.stButton > button, .stFormSubmitButton > button, .stLinkButton > a, .stDownloadButton > button {{
  background: {ORANGE} !important; color: #fff !important; border: none !important; border-radius: 4px !important;
  font-weight: 700 !important; text-transform: uppercase; letter-spacing: .03em; font-size: .85rem !important; padding: .6rem 1.2rem !important; }}
.stButton > button *, .stFormSubmitButton > button *, .stLinkButton > a *, .stDownloadButton > button * {{ color: #fff !important; }}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stLinkButton > a:hover {{ background: {ORANGE_DARK} !important; }}
[data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {LINE}; }}
[data-baseweb="tab-list"] button p {{ font-weight: 500; }}
[data-baseweb="tab-list"] button[aria-selected="true"] p {{ color: {ORANGE} !important; font-weight: 700; }}
[data-baseweb="tab-highlight"] {{ background: {ORANGE} !important; }}

.ic {{ vertical-align: -3px; flex-shrink: 0; }}

/* Thanh thương hiệu */
.topstrip {{ background: {ORANGE}; color: #fff; font-size: .78rem; text-align: center; padding: .35rem; border-radius: 4px 4px 0 0; }}
.brandbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  background:#fff; border:1px solid {LINE}; border-top:none; border-radius:0 0 4px 4px; padding:.85rem 1.2rem; margin-bottom:1.2rem; }}
.brand {{ display:flex; align-items:center; gap:.65rem; }}
.logo {{ width:38px; height:38px; border-radius:50%; border:2px solid {ORANGE}; color:{ORANGE}; display:flex;
  align-items:center; justify-content:center; font-weight:700; font-size:1.05rem; }}
.brand-name {{ font-weight:700; font-size:1.25rem; color:{INK}; line-height:1.15; }}
.brand-name span {{ color:{ORANGE}; }}
.brand-tag {{ font-size:.8rem; color:{GREY}; }}
.note {{ font-size:.78rem; color:{GREY}; border:1px solid {LINE}; border-radius:4px; padding:.3rem .6rem; }}

/* Banner */
.banner {{ background:{ORANGE}; color:#fff; border-radius:4px; padding:1.8rem 2rem; margin-bottom:1.3rem; }}
.banner h1 {{ color:#fff !important; font-size:1.9rem; margin:0 0 .35rem 0; line-height:1.25; }}
.banner p {{ margin:0; max-width:640px; opacity:.95; }}
.steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; margin-top:1.2rem; }}
.step {{ display:flex; gap:.7rem; align-items:flex-start; background:#fff; color:{INK}; border-radius:4px; padding:.75rem .9rem; font-size:.86rem; }}
.step .num {{ width:26px; height:26px; border-radius:50%; background:{ORANGE}; color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:.85rem; }}
.step b {{ display:block; font-size:.92rem; }}

/* Khối */
.panel {{ background:#fff; border:1px solid {LINE}; border-radius:6px; padding:1.1rem 1.25rem; margin-bottom:1rem; }}
.panel h3 {{ margin:0 0 .8rem 0; font-size:1.02rem; display:flex; align-items:center; gap:.45rem; }}
.sec {{ display:flex; align-items:center; gap:.5rem; font-size:1.2rem; font-weight:700; color:{INK};
  margin:1.6rem 0 .25rem 0; padding-left:.65rem; border-left:4px solid {ORANGE}; }}
.sec-sub {{ color:{GREY}; font-size:.9rem; margin:0 0 1rem .9rem; }}
.muted {{ color:{GREY}; font-size:.88rem; }}

.bar-row {{ margin:.5rem 0; }}
.bar-label {{ display:flex; justify-content:space-between; font-size:.88rem; margin-bottom:.22rem; }}
.bar-label span:last-child {{ font-weight:700; }}
.bar {{ height:8px; background:#F0F0F0; border-radius:2px; overflow:hidden; }}
.bar > div {{ height:100%; background:#CFCFCF; }}
.bar.hot > div {{ background:{ORANGE}; }}
.advice {{ border-left:3px solid {ORANGE}; background:#FFF7EF; padding:.65rem .9rem; margin:.55rem 0; font-size:.88rem; border-radius:0 4px 4px 0; }}
.advice b {{ color:{ORANGE_DARK}; }}

/* Lưới sản phẩm */
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:12px; }}
.card {{ position:relative; display:flex; flex-direction:column; background:#fff; border:1px solid {LINE}; border-radius:4px;
  overflow:hidden; text-decoration:none !important; color:{INK} !important; transition:box-shadow .15s ease, border-color .15s ease; }}
.card:hover {{ border-color:{ORANGE}; box-shadow:0 4px 14px rgba(0,0,0,.07); }}
.card .img {{ aspect-ratio:1/1; display:flex; align-items:center; justify-content:center; padding:12px; background:#fff; }}
.card .img img {{ max-width:100%; max-height:100%; object-fit:contain; }}
.card .badge {{ position:absolute; top:8px; right:8px; background:{ORANGE}; color:#fff; font-weight:700; font-size:.72rem; padding:.18rem .45rem; border-radius:2px; }}
.card .body {{ padding:.6rem .8rem .85rem; display:flex; flex-direction:column; gap:.28rem; flex:1; border-top:1px solid #F2F2F2; }}
.card .brandname {{ font-size:.72rem; font-weight:700; text-transform:uppercase; color:{GREY}; letter-spacing:.03em; }}
.card .name {{ font-size:.85rem; line-height:1.38; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; min-height:3.6em; }}
.card .price {{ margin-top:auto; font-weight:700; color:{ORANGE}; font-size:1rem; }}
.card .price s {{ color:#9A9A9A; font-weight:400; font-size:.8rem; margin-left:.35rem; }}
.card .why {{ font-size:.72rem; color:{GREY}; }}
.card .cta {{ margin-top:.3rem; text-align:center; border:1px solid {ORANGE}; color:{ORANGE}; font-weight:700; font-size:.76rem;
  border-radius:3px; padding:.38rem; text-transform:uppercase; letter-spacing:.03em; }}
.card:hover .cta {{ background:{ORANGE}; color:#fff; }}

/* Routine */
.rcol {{ background:#fff; border:1px solid {LINE}; border-radius:6px; overflow:hidden; margin-bottom:1rem; }}
.rhead {{ display:flex; align-items:center; gap:.5rem; padding:.7rem 1rem; font-weight:700; background:#FFF7EF; color:{ORANGE_DARK}; border-bottom:1px solid {LINE}; }}
.ritem {{ display:flex; gap:.8rem; align-items:center; padding:.7rem 1rem; border-bottom:1px solid #F2F2F2; text-decoration:none !important; color:{INK} !important; }}
.ritem:last-child {{ border-bottom:none; }}
.ritem:hover {{ background:#FCFCFC; }}
.ritem .n {{ width:24px; height:24px; border-radius:50%; border:1.5px solid {ORANGE}; color:{ORANGE}; font-size:.78rem; font-weight:700;
  display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.ritem img {{ width:54px; height:54px; object-fit:contain; border:1px solid #F0F0F0; border-radius:3px; background:#fff; flex-shrink:0; }}
.ritem .t {{ flex:1; min-width:0; }}
.ritem .st {{ font-size:.72rem; font-weight:700; text-transform:uppercase; color:{GREY}; letter-spacing:.03em; }}
.ritem .pn {{ font-size:.85rem; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
.ritem .pp {{ font-size:.85rem; font-weight:700; color:{ORANGE}; white-space:nowrap; }}
.total {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:.5rem; background:#fff; border:1px solid {LINE};
  border-left:4px solid {ORANGE}; border-radius:4px; padding:.9rem 1.1rem; margin-bottom:1rem; }}
.total .big {{ font-size:1.35rem; font-weight:700; color:{ORANGE}; }}

/* Liên hệ */
.contact-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:10px; }}
.contact {{ display:flex; gap:.75rem; align-items:flex-start; background:#fff; border:1px solid {LINE}; border-radius:4px; padding:.9rem 1rem; }}
.contact .ic {{ color:{ORANGE}; margin-top:2px; }}
.contact .k {{ font-size:.74rem; color:{GREY}; text-transform:uppercase; letter-spacing:.04em; }}
.contact .v {{ font-size:.95rem; font-weight:500; margin-top:.1rem; word-break:break-word; }}
.contact a {{ color:{ORANGE_DARK} !important; text-decoration:none; }}

.site-footer {{ margin-top:2.5rem; padding:1.3rem 1rem; border-top:1px solid {LINE}; color:{GREY}; font-size:.8rem; text-align:center; line-height:1.65; }}
.site-footer b {{ color:{INK}; }}

@media (max-width: 640px) {{
  .block-container {{ padding-top: 4.2rem !important; }}
  .banner {{ padding:1.3rem 1.1rem; }} .banner h1 {{ font-size:1.45rem; }}
  .grid {{ grid-template-columns:repeat(2,1fr); gap:8px; }}
  .card .name {{ font-size:.8rem; }}
  .note {{ display:none; }}
}}
</style>
"""


def html_block(s):
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def brand_bar():
    html_block(f"""
    <div class="topstrip">Dự án học tập · Gợi ý dựa trên sản phẩm đang bán tại Guardian Việt Nam</div>
    <div class="brandbar">
      <div class="brand"><div class="logo">{esc(APP_NAME[:1].upper())}</div>
        <div><div class="brand-name">{esc(APP_NAME)}</div><div class="brand-tag">{esc(TAGLINE)}</div></div>
      </div>
      <span class="note">Không phải trang chính thức của Guardian</span>
    </div>""")


def section(title, sub="", ic=None):
    i = icon(ic, 20) if ic else ""
    html_block(f'<div class="sec">{i}{esc(title)}</div>' + (f'<div class="sec-sub">{esc(sub)}</div>' if sub else ""))


def footer():
    html_block(f"""
    <div class="site-footer">
      <b>{esc(APP_NAME)}</b> · {esc(CONTACT['name'])} · {esc(CONTACT['email'])}<br>
      Kết quả phân tích chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ da liễu.<br>
      Dự án học tập độc lập, không liên kết với Guardian Việt Nam. Thông tin và giá sản phẩm thu thập tháng
      {esc(DATA_DATE)}, có thể đã thay đổi, vui lòng kiểm tra trên trang bán hàng.
    </div>""")


def discount_pct(r):
    try:
        pct = round(100 * (1 - float(r["price"]) / float(r["old_price"])))
        return pct if pct > 0 else 0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


def product_grid(df, show_why=False):
    cards = []
    for _, r in df.iterrows():
        pct = discount_pct(r)
        badge = f'<span class="badge">-{pct}%</span>' if pct else ""
        old = f"<s>{fmt_price(r['old_price'])}</s>" if r["old_price"] else ""
        why = f'<div class="why">Phù hợp: {esc(r["_why"])}</div>' if show_why and r.get("_why") else ""
        cards.append(
            f'<a class="card" href="{esc(r["url"])}" target="_blank" rel="noopener">{badge}'
            f'<div class="img"><img src="{esc(r["image"])}" loading="lazy" alt=""></div>'
            f'<div class="body"><div class="brandname">{esc(r["brand"])}</div>'
            f'<div class="name">{esc(r["name"])}</div>'
            f'<div class="price">{fmt_price(r["price"])}{old}</div>{why}'
            f'<div class="cta">Xem tại Guardian</div></div></a>'
        )
    st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# Model & dữ liệu
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("skin_model.keras")
    with open("class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


@st.cache_data
def load_products():
    df = pd.read_csv("products.csv", encoding="utf-8-sig", dtype=str).fillna("")
    for col in ["name", "brand", "type", "price", "old_price", "url", "image", "description"]:
        if col not in df.columns:
            df[col] = ""
    df = df[df["type"] != "Dụng cụ"].reset_index(drop=True)
    df.loc[df["type"] == "Xịt khoáng", "type"] = "Toner"
    df["_text"] = (df["name"] + " " + df["description"]).map(norm)
    df["_price"] = pd.to_numeric(df["price"], errors="coerce")
    df["_old"] = pd.to_numeric(df["old_price"], errors="coerce")
    df["_discount"] = ((1 - df["_price"] / df["_old"]) * 100).fillna(0)
    df["_concerns"] = df["_text"].map(
        lambda t: [VI_NAMES[c] for c, ks in KEYWORDS.items() if any(k in t for k in ks)])
    return df


def predict(model, class_names, img):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.asarray(img, dtype=np.float32), 0)
    probs = model.predict(arr, verbose=0)[0]
    return {c: float(p) for c, p in zip(class_names, probs)}


def recommend(df, probs):
    active = {c: p for c, p in probs.items() if p >= THRESHOLD}
    if not active:
        top = max(probs, key=probs.get)
        active = {top: probs[top]}
    scores, reasons = [], []
    for text in df["_text"]:
        score, why = 0.0, []
        for c, p in active.items():
            hits = [k for k in KEYWORDS.get(c.lower(), []) if k in text]
            if hits:
                score += p * (1 + 0.25 * (len(hits) - 1))
                why.append(VI_NAMES.get(c.lower(), c))
        scores.append(score)
        reasons.append(", ".join(why))
    out = df.assign(_score=scores, _why=reasons)
    out = out[out["_score"] > 0].sort_values("_score", ascending=False)
    out = out.groupby("type", sort=False).head(MAX_PER_TYPE).head(TOP_N)
    return out, active


# -------------------------------------------------------------
# TRANG 1: Phân tích da
# -------------------------------------------------------------
def page_analyze():
    brand_bar()
    html_block("""
    <div class="banner">
      <h1>Hiểu làn da của bạn chỉ với một tấm ảnh</h1>
      <p>Chụp ảnh cận mặt để nhận diện các vấn đề da thường gặp và nhận gợi ý sản phẩm phù hợp đang bán tại Guardian.</p>
      <div class="steps">
        <div class="step"><div class="num">1</div><div><b>Chụp ảnh</b>Cận mặt, đủ sáng, không dùng filter</div></div>
        <div class="step"><div class="num">2</div><div><b>Phân tích</b>Nhận diện 8 vấn đề da phổ biến</div></div>
        <div class="step"><div class="num">3</div><div><b>Gợi ý</b>Sản phẩm và routine theo thành phần</div></div>
      </div>
    </div>""")

    left, right = st.columns([5, 7], gap="large")
    with left:
        html_block(f'<div class="sec" style="margin-top:0">{icon("camera", 20)}Ảnh của bạn</div>')
        tab1, tab2 = st.tabs(["Chụp ảnh", "Tải ảnh lên"])
        with tab1:
            shot = st.camera_input("Chụp ảnh", label_visibility="collapsed")
        with tab2:
            upload = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            if upload:
                st.image(upload)
        file = shot or upload
        html_block(f'<div class="muted" style="margin-top:.5rem">{icon("lock", 15)} Ảnh chỉ dùng để phân tích, không được lưu lại.</div>')

    with right:
        if not file:
            html_block(f"""
            <div class="panel"><h3>{icon("info")}Kết quả sẽ hiển thị tại đây</h3>
            <div class="muted">Để kết quả chính xác hơn: chụp nơi có ánh sáng tự nhiên, để mặt mộc,
            giữ máy cách mặt khoảng 25–30 cm và nhìn thẳng vào camera.</div></div>""")
            footer()
            return

        model, class_names = load_model()
        with st.spinner("Đang phân tích..."):
            probs = predict(model, class_names, Image.open(file))

        rows = []
        for c, p in sorted(probs.items(), key=lambda t: -t[1]):
            hot = " hot" if p >= THRESHOLD else ""
            rows.append(
                f'<div class="bar-row"><div class="bar-label"><span>{esc(VI_NAMES.get(c.lower(), c))}</span>'
                f'<span>{p:.0%}</span></div><div class="bar{hot}"><div style="width:{p*100:.1f}%"></div></div></div>')
        html_block(f'<div class="panel"><h3>{icon("scan")}Kết quả phân tích</h3>' + "".join(rows) + "</div>")

        products = load_products()
        recs, active = recommend(products, probs)
        tips = "".join(
            f'<div class="advice"><b>{esc(VI_NAMES.get(c.lower(), c))}:</b> {esc(ADVICE.get(c.lower(), ""))}</div>'
            for c in active)
        html_block(f'<div class="panel"><h3>{icon("info")}Lời khuyên chăm sóc</h3>' + tips + "</div>")

        active_vi = [VI_NAMES.get(c.lower(), c) for c in active]
        if st.button("Tạo routine từ kết quả này"):
            st.session_state["r_concerns"] = active_vi
            st.switch_page(PAGE_ROUTINE)

    section("Sản phẩm gợi ý cho bạn",
            "Chọn dựa trên thành phần phù hợp với kết quả phân tích. Bấm vào sản phẩm để xem và mua trên Guardian.", "bag")
    if recs.empty:
        st.info("Chưa tìm thấy sản phẩm phù hợp trong danh sách.")
    else:
        product_grid(recs, show_why=True)
    st.write("")
    if st.button(f"Xem tất cả sản phẩm cho {active_vi[0]}"):
        st.session_state["f_concern"] = [active_vi[0]]
        st.switch_page(PAGE_PRODUCTS)
    footer()


# -------------------------------------------------------------
# TRANG 2: Routine builder
# -------------------------------------------------------------
def score_for_routine(df, concerns, skin):
    concern_keys = [VI_TO_KEY[c] for c in concerns if c in VI_TO_KEY]
    skin_kw = SKIN_TYPES.get(skin, [])

    def score(text):
        s = sum(1.0 for c in concern_keys if any(k in text for k in KEYWORDS[c]))
        s += 0.8 * any(k in text for k in skin_kw)
        if skin == "Da nhạy cảm" and any(k in text for k in ["retinol", "retinal", "aha", "glycolic"]):
            s -= 1.0
        return s

    return df.assign(_rs=df["_text"].map(score))


def routine_candidates(scored, ptype, cap, lam):
    c = scored[(scored["type"] == ptype) & scored["_price"].notna()].copy()
    if c.empty:
        return c
    # Ưu tiên sản phẩm phù hợp nhất; sản phẩm vượt phần ngân sách của bước này bị trừ điểm (hệ số lam)
    over = ((c["_price"] - cap) / cap).clip(lower=0)
    c["_k"] = c["_rs"] - lam * over
    return c.sort_values(["_k", "_price"], ascending=[False, True]).head(6)


def build_candidates(scored, types_needed, budget):
    """Tìm mức trừ điểm nhỏ nhất sao cho bộ sản phẩm gợi ý mặc định nằm trong ngân sách."""
    wsum = sum(BUDGET_WEIGHT[t] for t in types_needed)
    caps = {t: budget * BUDGET_WEIGHT[t] / wsum for t in types_needed}
    for lam in [0.5, 1, 2, 4, 8, 20, 100]:
        cands = {t: routine_candidates(scored, t, caps[t], lam) for t in types_needed}
        total = sum(c.iloc[0]["_price"] for c in cands.values() if not c.empty)
        if total <= budget:
            break
    return cands


def routine_item(n, step_name, r):
    return (f'<a class="ritem" href="{esc(r["url"])}" target="_blank" rel="noopener">'
            f'<div class="n">{n}</div><img src="{esc(r["image"])}" alt="" loading="lazy">'
            f'<div class="t"><div class="st">{esc(step_name)}</div><div class="pn">{esc(r["name"])}</div></div>'
            f'<div class="pp">{fmt_price(r["price"])}</div></a>')


def page_routine():
    brand_bar()
    section("Tạo routine chăm sóc da", "Chọn tình trạng da và ngân sách, hệ thống sẽ xếp các bước sáng – tối với sản phẩm phù hợp.", "calendar")
    df = load_products()

    left, right = st.columns([4, 8], gap="large")
    with left:
        with st.container(border=True):
            skin = st.radio("Loại da", list(SKIN_TYPES), horizontal=True)
            concerns = st.multiselect("Vấn đề da cần cải thiện", list(VI_NAMES.values()), key="r_concerns",
                                      placeholder="Chọn một hoặc nhiều vấn đề")
            budget_label = st.selectbox("Ngân sách cho cả bộ", list(BUDGETS), index=1)
            level = st.radio("Mức độ", ["Cơ bản", "Đầy đủ"], horizontal=True,
                             captions=["3 bước sáng / 3 bước tối", "5 bước sáng / 5 bước tối + hằng tuần"])

    plan = ROUTINE[level]
    types_needed = list(dict.fromkeys(t for part in plan.values() for t, _ in part))
    budget = BUDGETS[budget_label]
    scored = score_for_routine(df, concerns, skin)
    all_cands = build_candidates(scored, types_needed, budget)

    picks = {}
    with left:
        with st.expander("Đổi sản phẩm cho từng bước"):
            for t in types_needed:
                cands = all_cands[t]
                if cands.empty:
                    continue
                labels = [f"{r['name'][:60]} · {fmt_price(r['price'])}" for _, r in cands.iterrows()]
                i = st.selectbox(t, range(len(labels)), format_func=lambda k, L=labels: L[k], key=f"pick_{t}")
                picks[t] = cands.iloc[i if i is not None and i < len(cands) else 0]

    with right:
        total = sum(float(r["_price"]) for r in picks.values() if pd.notna(r["_price"]))
        status = "Trong ngân sách" if total <= budget else "Vượt ngân sách, thử đổi sản phẩm hoặc chọn mức Cơ bản"
        html_block(f"""<div class="total"><div><div class="muted">Tổng chi phí cả bộ ({len(picks)} sản phẩm)</div>
            <div class="big">{fmt_price(total)}</div></div><div class="muted">Ngân sách {fmt_price(budget)}<br>{status}</div></div>""")

        heads = {"Sáng": icon("sun"), "Tối": icon("moon"), "Hằng tuần": icon("calendar")}
        c1, c2 = st.columns(2)
        for col, part in [(c1, "Sáng"), (c2, "Tối")]:
            items = "".join(routine_item(i + 1, name, picks[t]) for i, (t, name) in enumerate(plan[part]) if t in picks)
            with col:
                html_block(f'<div class="rcol"><div class="rhead">{heads[part]}Buổi {part.lower()}</div>{items}</div>')
        if plan["Hằng tuần"]:
            items = "".join(routine_item(i + 1, name, picks[t]) for i, (t, name) in enumerate(plan["Hằng tuần"]) if t in picks)
            html_block(f'<div class="rcol"><div class="rhead">{heads["Hằng tuần"]}Hằng tuần</div>{items}</div>')

        notes = ["Thử sản phẩm mới trên một vùng da nhỏ (sau tai, quai hàm) trong 24 giờ trước khi dùng toàn mặt.",
                 "Thêm từng sản phẩm mới cách nhau khoảng 1–2 tuần để biết da hợp sản phẩm nào."]
        texts = " ".join(r["_text"] for r in picks.values())
        if any(k in texts for k in ["retinol", "retinal", "vitamin a", "vita-a"]):
            notes.append("Sản phẩm có retinol/vitamin A chỉ dùng buổi tối; tối nào dùng thì bỏ bước tẩy tế bào chết.")
        if any(k in texts for k in ["bha", "aha", "salicylic", "glycolic"]):
            notes.append("Sản phẩm có AHA/BHA: bắt đầu 2–3 lần/tuần rồi tăng dần, luôn dùng kem chống nắng ban ngày.")
        if skin == "Da nhạy cảm":
            notes.append("Da nhạy cảm nên tránh dùng nhiều hoạt chất mạnh cùng lúc.")
        html_block(f'<div class="panel"><h3>{icon("info")}Lưu ý khi dùng</h3>'
                   + "".join(f'<div class="advice">{esc(n)}</div>' for n in notes) + "</div>")

        lines = [f"ROUTINE CHĂM SÓC DA · {APP_NAME}", f"Loại da: {skin}",
                 f"Vấn đề: {', '.join(concerns) or 'không chọn'}", ""]
        for part, steps in plan.items():
            if not steps:
                continue
            lines.append(part.upper())
            for i, (t, name) in enumerate(steps):
                if t in picks:
                    r = picks[t]
                    lines.append(f"{i + 1}. {name}: {r['name']} ({fmt_price(r['price'])})\n   {r['url']}")
            lines.append("")
        lines.append(f"Tổng: {fmt_price(total)}")
        st.download_button("Tải routine về máy", "\n".join(lines), file_name="routine.txt", mime="text/plain")
    footer()


# -------------------------------------------------------------
# TRANG 3: Tất cả sản phẩm
# -------------------------------------------------------------
def page_products():
    brand_bar()
    df = load_products()
    section("Tất cả sản phẩm chăm sóc da mặt", f"{len(df)} sản phẩm · Dùng bộ lọc để tìm nhanh sản phẩm phù hợp", "bag")

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 2])
        q = c1.text_input("Tìm kiếm", placeholder="Tên sản phẩm, thương hiệu, thành phần...")
        types = c2.multiselect("Loại sản phẩm", sorted(t for t in df["type"].unique() if t), placeholder="Tất cả")
        concerns = c3.multiselect("Vấn đề da", list(VI_NAMES.values()), key="f_concern", placeholder="Tất cả")
        c4, c5, c6 = st.columns([3, 2, 2])
        lo, hi = int(df["_price"].min()), int(df["_price"].max())
        price = c4.slider("Khoảng giá (đ)", lo, hi, (lo, hi), step=10000, format="%d")
        brands = c5.multiselect("Thương hiệu", sorted(b for b in df["brand"].unique() if b), placeholder="Tất cả")
        sort = c6.selectbox("Sắp xếp", ["Phù hợp nhất", "Giá thấp đến cao", "Giá cao đến thấp", "Giảm giá nhiều nhất", "Tên A–Z"])
        sale_only = st.toggle("Chỉ hiện sản phẩm đang giảm giá")

    f = df.copy()
    if q:
        qn = norm(q)
        f = f[f["_text"].str.contains(qn, regex=False) | f["brand"].map(norm).str.contains(qn, regex=False)]
    if types:
        f = f[f["type"].isin(types)]
    if brands:
        f = f[f["brand"].isin(brands)]
    if concerns:
        f = f[f["_concerns"].map(lambda cs: any(c in cs for c in concerns))]
    f = f[f["_price"].between(price[0], price[1]) | f["_price"].isna()]
    if sale_only:
        f = f[f["_discount"] > 0]
    order = {"Giá thấp đến cao": ("_price", True), "Giá cao đến thấp": ("_price", False),
             "Giảm giá nhiều nhất": ("_discount", False), "Tên A–Z": ("name", True)}
    if sort in order:
        f = f.sort_values(order[sort][0], ascending=order[sort][1])

    html_block(f'<div class="muted" style="margin:.4rem 0 .8rem">Hiển thị <b>{len(f)}</b> sản phẩm</div>')
    if f.empty:
        st.info("Không có sản phẩm nào khớp bộ lọc. Thử bỏ bớt điều kiện.")
    else:
        product_grid(f.assign(_why=f["_concerns"].map(", ".join)), show_why=True)
    footer()


# -------------------------------------------------------------
# TRANG 4: Giới thiệu & Liên hệ
# -------------------------------------------------------------
def page_contact():
    brand_bar()
    section("Về dự án", ic="info")
    html_block(f"""
    <div class="panel">
      <p><b>{esc(APP_NAME)}</b> ứng dụng học sâu vào chăm sóc da. Mô hình MobileNetV2 (transfer learning) được huấn luyện
      trên hơn 4.000 ảnh da mặt để nhận diện 8 vấn đề: mụn viêm, mụn đầu đen, mụn đầu trắng, lỗ chân lông to,
      sắc tố/nám, đốm thâm, nếp nhăn và da ửng đỏ.</p>
      <p>Từ kết quả, hệ thống đối chiếu thành phần trong tên sản phẩm (BHA, niacinamide, vitamin C, retinol, rau má...)
      để gợi ý sản phẩm và xây dựng routine từ danh mục chăm sóc da mặt của Guardian Việt Nam.</p>
      <div class="advice"><b>Lưu ý:</b> Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ da liễu.
      Nếu da viêm nặng, đau, chảy dịch hoặc kéo dài, hãy đến cơ sở y tế.</div>
    </div>""")

    section("Liên hệ", "Góp ý, báo lỗi hoặc hợp tác, vui lòng liên hệ qua các kênh sau.", "chat")
    html_block(f"""
    <div class="contact-grid">
      <div class="contact">{icon("user", 20)}<div><div class="k">Người thực hiện</div><div class="v">{esc(CONTACT['name'])}</div></div></div>
      <div class="contact">{icon("school", 20)}<div><div class="k">Đơn vị</div><div class="v">{esc(CONTACT['school'])}</div></div></div>
      <div class="contact">{icon("mail", 20)}<div><div class="k">Email</div><div class="v"><a href="mailto:{esc(CONTACT['email'])}">{esc(CONTACT['email'])}</a></div></div></div>
      <div class="contact">{icon("phone", 20)}<div><div class="k">Điện thoại</div><div class="v">{esc(CONTACT['phone'])}</div></div></div>
      <div class="contact">{icon("chat", 20)}<div><div class="k">Facebook</div><div class="v"><a href="{esc(CONTACT['facebook'])}" target="_blank">Nhắn tin trên Facebook</a></div></div></div>
      <div class="contact">{icon("pin", 20)}<div><div class="k">Địa chỉ</div><div class="v">{esc(CONTACT['address'])}</div></div></div>
    </div>""")

    section("Gửi góp ý", ic="mail")
    with st.form("feedback"):
        name = st.text_input("Họ tên", placeholder="Nguyễn Văn A")
        msg = st.text_area("Nội dung góp ý", placeholder="Bạn thấy kết quả phân tích thế nào? Có lỗi gì cần sửa?")
        sent = st.form_submit_button("Tạo email góp ý")
    if sent:
        if not msg.strip():
            st.warning("Vui lòng nhập nội dung góp ý.")
        else:
            body = msg + "\n\n— " + name
            link = f"mailto:{CONTACT['email']}?subject={quote('Góp ý ' + APP_NAME)}&body={quote(body)}"
            st.link_button("Mở ứng dụng email để gửi", link)
    footer()


# -------------------------------------------------------------
# Điều hướng
# -------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
PAGE_ANALYZE = st.Page(page_analyze, title="Phân tích da", icon=":material/face_retouching_natural:", default=True, url_path="phan-tich")
PAGE_ROUTINE = st.Page(page_routine, title="Routine", icon=":material/event_note:", url_path="routine")
PAGE_PRODUCTS = st.Page(page_products, title="Sản phẩm", icon=":material/shopping_bag:", url_path="san-pham")
PAGE_CONTACT = st.Page(page_contact, title="Liên hệ", icon=":material/mail:", url_path="lien-he")
st.navigation([PAGE_ANALYZE, PAGE_ROUTINE, PAGE_PRODUCTS, PAGE_CONTACT], position="top").run()
