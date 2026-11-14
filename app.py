# =============================================================
#  SkinScan: phân tích da & gợi ý sản phẩm
#  Cần các file nằm chung 1 thư mục:
#    app.py, skin_model.keras, class_names.json, products.csv, requirements.txt
# =============================================================
import html
import json
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# -------------------------------------------------------------
# 👉 SỬA THÔNG TIN CỦA BẠN Ở ĐÂY
# -------------------------------------------------------------
APP_NAME = "SkinScan-Guardian"
TAGLINE = "Phân tích da bằng AI · Gợi ý sản phẩm có bán tại Guardian"
CONTACT = {
    "name": "4 HACKER MŨ HỒNG",
    "school": "TỔ CHỨC HACKER MŨ HỒNG",
    "email": "pinkhathacker@example.com",
    "phone": "09xx xxx xxx",
    "facebook": "https://facebook.com/",
    "address": "TP. Hồ Chí Minh, Việt Nam",
}
DATA_DATE = "11/2026"   # thời điểm lấy dữ liệu sản phẩm

# Màu chủ đạo
ORANGE = "#F6831F"
ORANGE_DARK = "#D96A0B"
INK = "#2F2F2E"
GREY = "#545453"
BG = "#FAFAFA"

IMG_SIZE = 224
THRESHOLD = 0.20     # chỉ tính vấn đề da có xác suất từ 20% trở lên
TOP_N = 8            # số sản phẩm gợi ý
MAX_PER_TYPE = 2     # tối đa 2 sản phẩm cùng loại để gợi ý đa dạng

st.set_page_config(page_title=f"{APP_NAME} | Phân tích da", page_icon="🧡", layout="wide")

# -------------------------------------------------------------
# Dữ liệu về các vấn đề da
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
# Giao diện chung (CSS)
# -------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {{ font-family: 'Be Vietnam Pro', sans-serif !important; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1180px; }}
footer, [data-testid="stDecoration"] {{ display: none !important; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

/* Nút bấm */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {{
  background: {ORANGE}; color: #fff !important; border: none; border-radius: 999px;
  font-weight: 600; padding: .55rem 1.3rem; }}
.stButton > button:hover, .stLinkButton > a:hover {{ background: {ORANGE_DARK}; color: #fff !important; }}
[data-baseweb="tab-list"] button[aria-selected="true"] {{ color: {ORANGE} !important; }}
[data-baseweb="tab-highlight"] {{ background: {ORANGE} !important; }}

/* Thanh thương hiệu */
.brandbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  background:#fff; border-radius:16px; padding:.9rem 1.3rem; box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:1.2rem; }}
.brand {{ display:flex; align-items:center; gap:.7rem; }}
.logo {{ width:40px; height:40px; border-radius:12px; background:{ORANGE}; color:#fff; display:flex;
  align-items:center; justify-content:center; font-weight:800; font-size:1.2rem; }}
.brand-name {{ font-weight:800; font-size:1.25rem; color:{INK}; line-height:1.1; }}
.brand-tag {{ font-size:.8rem; color:{GREY}; }}
.pill {{ display:inline-block; background:#FFF1E4; color:{ORANGE_DARK}; font-weight:600; font-size:.78rem;
  padding:.3rem .75rem; border-radius:999px; }}

/* Hero */
.hero {{ background:linear-gradient(120deg, {ORANGE} 0%, #FF9F4A 100%); color:#fff; border-radius:22px;
  padding:2.2rem 2rem; margin-bottom:1.4rem; }}
.hero h1 {{ color:#fff; font-size:2.1rem; font-weight:800; margin:0 0 .4rem 0; line-height:1.2; }}
.hero p {{ font-size:1.02rem; opacity:.95; margin:0; max-width:640px; }}
.steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:.8rem; margin-top:1.4rem; }}
.step {{ background:rgba(255,255,255,.16); border-radius:14px; padding:.8rem 1rem; font-size:.9rem; }}
.step b {{ display:block; font-size:1rem; margin-bottom:.15rem; }}

/* Thẻ chung */
.panel {{ background:#fff; border-radius:18px; padding:1.3rem 1.4rem; box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:1rem; }}
.panel h3 {{ margin:0 0 .8rem 0; font-size:1.1rem; }}
.section-title {{ font-size:1.35rem; font-weight:800; color:{INK}; margin:1.4rem 0 .2rem 0; }}
.section-sub {{ color:{GREY}; font-size:.92rem; margin-bottom:1rem; }}

/* Thanh kết quả */
.bar-row {{ margin:.55rem 0; }}
.bar-label {{ display:flex; justify-content:space-between; font-size:.9rem; color:{INK}; margin-bottom:.25rem; }}
.bar-label span:last-child {{ font-weight:700; }}
.bar {{ height:9px; background:#F1F1F1; border-radius:99px; overflow:hidden; }}
.bar > div {{ height:100%; background:#D9D9D9; border-radius:99px; }}
.bar.hot > div {{ background:{ORANGE}; }}
.advice {{ border-left:4px solid {ORANGE}; background:#FFF8F1; border-radius:10px; padding:.75rem 1rem; margin:.6rem 0; font-size:.92rem; color:{INK}; }}
.advice b {{ color:{ORANGE_DARK}; }}

/* Lưới sản phẩm */
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }}
.card {{ position:relative; display:flex; flex-direction:column; background:#fff; border-radius:16px; overflow:hidden;
  text-decoration:none !important; color:{INK} !important; border:1px solid #EEE; transition:all .18s ease; }}
.card:hover {{ transform:translateY(-3px); box-shadow:0 8px 22px rgba(0,0,0,.08); border-color:{ORANGE}; }}
.card .img {{ aspect-ratio:1/1; background:#fff; display:flex; align-items:center; justify-content:center; padding:10px; }}
.card .img img {{ max-width:100%; max-height:100%; object-fit:contain; }}
.card .badge {{ position:absolute; top:10px; left:10px; background:#E53935; color:#fff; font-weight:700;
  font-size:.75rem; padding:.2rem .5rem; border-radius:6px; }}
.card .body {{ padding:.7rem .85rem .9rem; display:flex; flex-direction:column; gap:.3rem; flex:1; }}
.card .meta {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; color:{ORANGE_DARK}; font-weight:700; }}
.card .name {{ font-size:.86rem; line-height:1.35; font-weight:500; display:-webkit-box; -webkit-line-clamp:3;
  -webkit-box-orient:vertical; overflow:hidden; min-height:3.5em; }}
.card .price {{ margin-top:auto; font-weight:800; color:{INK}; font-size:1rem; }}
.card .price s {{ color:#9A9A9A; font-weight:400; font-size:.8rem; margin-left:.35rem; }}
.card .why {{ font-size:.72rem; color:{GREY}; }}
.card .cta {{ margin-top:.35rem; text-align:center; background:#FFF1E4; color:{ORANGE_DARK}; font-weight:700;
  font-size:.8rem; border-radius:999px; padding:.4rem; }}
.card:hover .cta {{ background:{ORANGE}; color:#fff; }}

/* Liên hệ */
.contact-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
.contact {{ background:#fff; border-radius:16px; padding:1rem 1.1rem; border:1px solid #EEE; }}
.contact .k {{ font-size:.75rem; color:{GREY}; text-transform:uppercase; letter-spacing:.05em; font-weight:600; }}
.contact .v {{ font-size:1rem; font-weight:600; color:{INK}; margin-top:.2rem; word-break:break-word; }}
.contact a {{ color:{ORANGE_DARK} !important; text-decoration:none; }}

/* Chân trang */
.site-footer {{ margin-top:2.5rem; padding:1.4rem 1.2rem; border-top:1px solid #E7E7E7; color:{GREY};
  font-size:.8rem; text-align:center; line-height:1.6; }}
.site-footer b {{ color:{INK}; }}

@media (max-width: 640px) {{
  .hero {{ padding:1.5rem 1.2rem; }} .hero h1 {{ font-size:1.55rem; }}
  .grid {{ grid-template-columns:repeat(2,1fr); gap:10px; }}
  .card .name {{ font-size:.8rem; }}
}}
</style>
"""


def html_block(s):
    # Gộp thành 1 dòng để Streamlit không hiểu nhầm thành khối code
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def brand_bar():
    html_block(f"""
    <div class="brandbar">
      <div class="brand"><div class="logo">S</div>
        <div><div class="brand-name">{esc(APP_NAME)}</div><div class="brand-tag">{esc(TAGLINE)}</div></div>
      </div>
      <span class="pill">Dự án học tập · Không phải trang chính thức của Guardian</span>
    </div>""")


def footer():
    html_block(f"""
    <div class="site-footer">
      <b>{esc(APP_NAME)}</b> · {esc(CONTACT['name'])} · {esc(CONTACT['email'])}<br>
      Kết quả phân tích chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ da liễu.<br>
      Đây là dự án học tập độc lập, không liên kết với Guardian Việt Nam. Thông tin và giá sản phẩm
      được thu thập tháng {esc(DATA_DATE)}, có thể đã thay đổi; vui lòng kiểm tra trên trang bán hàng.
    </div>""")


def product_grid(df, show_why=False):
    cards = []
    for _, r in df.iterrows():
        badge = ""
        if r["old_price"] and r["price"]:
            try:
                pct = round(100 * (1 - float(r["price"]) / float(r["old_price"])))
                if pct > 0:
                    badge = f'<span class="badge">-{pct}%</span>'
            except ValueError:
                pass
        old = f"<s>{fmt_price(r['old_price'])}</s>" if r["old_price"] else ""
        why = f'<div class="why">Phù hợp: {esc(r["_why"])}</div>' if show_why and r.get("_why") else ""
        meta = " · ".join(x for x in [r["type"], r["brand"]] if x)
        cards.append(
            f'<a class="card" href="{esc(r["url"])}" target="_blank" rel="noopener">{badge}'
            f'<div class="img"><img src="{esc(r["image"])}" loading="lazy" alt=""></div>'
            f'<div class="body"><div class="meta">{esc(meta)}</div>'
            f'<div class="name">{esc(r["name"])}</div>'
            f'<div class="price">{fmt_price(r["price"])}{old}</div>{why}'
            f'<div class="cta">Xem tại Guardian ↗</div></div></a>'
        )
    st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# Tải model & dữ liệu
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải model AI...")
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
    html_block(f"""
    <div class="hero">
      <h1>Hiểu làn da của bạn trong 10 giây</h1>
      <p>Chụp một tấm ảnh cận mặt, AI sẽ nhận diện các vấn đề da thường gặp và gợi ý sản phẩm phù hợp đang bán tại Guardian.</p>
      <div class="steps">
        <div class="step"><b>① Chụp ảnh</b>Cận mặt, đủ sáng, không filter</div>
        <div class="step"><b>② AI phân tích</b>Nhận diện 8 vấn đề da phổ biến</div>
        <div class="step"><b>③ Gợi ý sản phẩm</b>Chọn theo thành phần phù hợp</div>
      </div>
    </div>""")

    left, right = st.columns([5, 7], gap="large")
    with left:
        st.markdown("#### Ảnh của bạn")
        tab1, tab2 = st.tabs(["📷 Chụp ảnh", "🖼️ Tải ảnh lên"])
        with tab1:
            shot = st.camera_input("Chụp ảnh", label_visibility="collapsed")
        with tab2:
            upload = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"],
                                      label_visibility="collapsed")
        file = shot or upload
        if file and upload and not shot:
            st.image(upload)
        st.caption("🔒 Ảnh chỉ được dùng để phân tích ngay lúc này, không được lưu lại.")

    with right:
        if not file:
            html_block("""
            <div class="panel"><h3>Kết quả sẽ hiện ở đây</h3>
            <div class="section-sub">Mẹo để có kết quả chính xác hơn: chụp ở nơi đủ ánh sáng tự nhiên, để mặt mộc,
            giữ máy cách mặt khoảng 25–30 cm.</div></div>""")
            return

        model, class_names = load_model()
        with st.spinner("AI đang phân tích làn da..."):
            probs = predict(model, class_names, Image.open(file))

        ranked = sorted(probs.items(), key=lambda t: -t[1])
        rows = []
        for c, p in ranked:
            hot = " hot" if p >= THRESHOLD else ""
            rows.append(
                f'<div class="bar-row"><div class="bar-label"><span>{esc(VI_NAMES.get(c.lower(), c))}</span>'
                f'<span>{p:.0%}</span></div><div class="bar{hot}"><div style="width:{p*100:.1f}%"></div></div></div>')
        html_block('<div class="panel"><h3>Kết quả phân tích</h3>' + "".join(rows) + "</div>")

        products = load_products()
        recs, active = recommend(products, probs)
        tips = "".join(
            f'<div class="advice"><b>{esc(VI_NAMES.get(c.lower(), c))}:</b> {esc(ADVICE.get(c.lower(), ""))}</div>'
            for c in active)
        html_block('<div class="panel"><h3>Lời khuyên chăm sóc</h3>' + tips + "</div>")

    if file:
        html_block('<div class="section-title">Sản phẩm gợi ý cho bạn</div>'
                   '<div class="section-sub">Chọn dựa trên thành phần phù hợp với kết quả phân tích. '
                   'Bấm vào sản phẩm để xem chi tiết và mua trên Guardian.</div>')
        if recs.empty:
            st.info("Chưa tìm thấy sản phẩm phù hợp trong danh sách.")
        else:
            product_grid(recs, show_why=True)
        top = max(active, key=active.get)
        top_vi = VI_NAMES.get(top.lower(), top)
        st.write("")
        if st.button(f"Xem tất cả sản phẩm cho “{top_vi}” →"):
            st.session_state["f_concern"] = [top_vi]
            st.switch_page(PAGE_PRODUCTS)
    footer()


# -------------------------------------------------------------
# TRANG 2: Tất cả sản phẩm
# -------------------------------------------------------------
def page_products():
    brand_bar()
    df = load_products()
    html_block(f'<div class="section-title">Tất cả sản phẩm chăm sóc da mặt</div>'
               f'<div class="section-sub">{len(df)} sản phẩm · Dùng bộ lọc để tìm nhanh sản phẩm phù hợp</div>')

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 2])
        q = c1.text_input("Tìm kiếm", placeholder="Tên sản phẩm, thương hiệu, thành phần...")
        types = c2.multiselect("Loại sản phẩm", sorted(t for t in df["type"].unique() if t))
        concerns = c3.multiselect("Vấn đề da", list(VI_NAMES.values()), key="f_concern")
        c4, c5, c6 = st.columns([3, 2, 2])
        lo, hi = int(df["_price"].min()), int(df["_price"].max())
        price = c4.slider("Khoảng giá (đ)", lo, hi, (lo, hi), step=10000, format="%d")
        brands = c5.multiselect("Thương hiệu", sorted(b for b in df["brand"].unique() if b))
        sort = c6.selectbox("Sắp xếp", ["Phù hợp nhất", "Giá thấp → cao", "Giá cao → thấp", "Giảm giá nhiều nhất", "Tên A → Z"])
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
    if sort == "Giá thấp → cao":
        f = f.sort_values("_price")
    elif sort == "Giá cao → thấp":
        f = f.sort_values("_price", ascending=False)
    elif sort == "Giảm giá nhiều nhất":
        f = f.sort_values("_discount", ascending=False)
    elif sort == "Tên A → Z":
        f = f.sort_values("name")

    st.markdown(f"**{len(f)}** sản phẩm")
    if f.empty:
        st.info("Không có sản phẩm nào khớp bộ lọc. Thử bỏ bớt điều kiện nhé.")
    else:
        f = f.assign(_why=f["_concerns"].map(", ".join))
        product_grid(f, show_why=True)
    footer()


# -------------------------------------------------------------
# TRANG 3: Giới thiệu & Liên hệ
# -------------------------------------------------------------
def page_contact():
    brand_bar()
    html_block(f"""
    <div class="section-title">Về {esc(APP_NAME)}</div>
    <div class="panel">
      <p><b>{esc(APP_NAME)}</b> là dự án ứng dụng trí tuệ nhân tạo vào chăm sóc da. Hệ thống dùng mô hình học sâu
      MobileNetV2 (transfer learning), được huấn luyện trên hơn 4.000 ảnh da mặt, để nhận diện 8 vấn đề da phổ biến:
      mụn viêm, mụn đầu đen, mụn đầu trắng, lỗ chân lông to, sắc tố/nám, đốm thâm, nếp nhăn và da ửng đỏ.</p>
      <p>Dựa trên kết quả, hệ thống đối chiếu thành phần trong tên sản phẩm (BHA, niacinamide, vitamin C, retinol,
      rau má...) để gợi ý sản phẩm phù hợp từ danh mục chăm sóc da mặt của Guardian Việt Nam.</p>
      <div class="advice"><b>Lưu ý:</b> Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ da liễu.
      Nếu da có dấu hiệu viêm nặng, đau, chảy dịch hoặc kéo dài, hãy đến cơ sở y tế.</div>
    </div>

    <div class="section-title">Liên hệ</div>
    <div class="section-sub">Góp ý, báo lỗi hoặc hợp tác, hãy liên hệ với chúng tôi qua:</div>
    <div class="contact-grid">
      <div class="contact"><div class="k">Người thực hiện</div><div class="v">{esc(CONTACT['name'])}</div></div>
      <div class="contact"><div class="k">Đơn vị</div><div class="v">{esc(CONTACT['school'])}</div></div>
      <div class="contact"><div class="k">Email</div><div class="v"><a href="mailto:{esc(CONTACT['email'])}">{esc(CONTACT['email'])}</a></div></div>
      <div class="contact"><div class="k">Điện thoại</div><div class="v">{esc(CONTACT['phone'])}</div></div>
      <div class="contact"><div class="k">Facebook</div><div class="v"><a href="{esc(CONTACT['facebook'])}" target="_blank">Nhắn tin trên Facebook ↗</a></div></div>
      <div class="contact"><div class="k">Địa chỉ</div><div class="v">{esc(CONTACT['address'])}</div></div>
    </div>""")

    st.write("")
    with st.container(border=True):
        st.markdown("#### Gửi góp ý nhanh")
        with st.form("feedback", clear_on_submit=False):
            name = st.text_input("Tên của bạn")
            msg = st.text_area("Nội dung góp ý")
            sent = st.form_submit_button("Tạo email góp ý")
        if sent and msg:
            from urllib.parse import quote
            body = msg + "\n\n— " + name
            link = f"mailto:{CONTACT['email']}?subject={quote('Góp ý ' + APP_NAME)}&body={quote(body)}"
            st.link_button("📧 Mở ứng dụng email để gửi", link)
    footer()


# -------------------------------------------------------------
# Điều hướng
# -------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
PAGE_ANALYZE = st.Page(page_analyze, title="Phân tích da", icon="✨", default=True, url_path="phan-tich")
PAGE_PRODUCTS = st.Page(page_products, title="Sản phẩm", icon="🛍️", url_path="san-pham")
PAGE_CONTACT = st.Page(page_contact, title="Giới thiệu & Liên hệ", icon="📮", url_path="lien-he")
st.navigation([PAGE_ANALYZE, PAGE_PRODUCTS, PAGE_CONTACT], position="top").run()
