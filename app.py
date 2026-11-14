# =============================================================
#  App phân tích da & gợi ý sản phẩm
#  Cần 4 file nằm chung 1 thư mục:
#    app.py, skin_model.keras, class_names.json, products.csv
# =============================================================
import json
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Phân tích da", page_icon="🧴", layout="centered")

IMG_SIZE = 224
THRESHOLD = 0.20   # chỉ tính những vấn đề da có xác suất từ 20% trở lên
TOP_N = 6          # số sản phẩm gợi ý tối đa
MAX_PER_TYPE = 2   # tối đa 2 sản phẩm cùng loại (vd 2 mặt nạ) để gợi ý đa dạng

# Tên tiếng Việt cho 8 class trong dataset
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

# Từ khóa để nhận ra sản phẩm phù hợp với từng vấn đề.
# App tìm các từ này trong TÊN + MÔ TẢ sản phẩm (không phân biệt hoa thường).
# Muốn gợi ý chính xác hơn thì thêm từ khóa vào đây.
KEYWORDS = {
    "inflammatory acne": ["salicylic", "bha", "benzoyl", "tea tree", "tràm trà", "azelaic",
                          "trị mụn", "giảm mụn", "ngừa mụn", "acne", "mụn"],
    "non inflammatory acne black heads": ["salicylic", "bha", "đất sét", "clay", "than hoạt tính",
                                          "charcoal", "mụn đầu đen", "blackhead", "lỗ chân lông"],
    "non inflammatory acne white heads": ["salicylic", "bha", "aha", "glycolic", "lactic",
                                          "tẩy tế bào chết", "tẩy da chết", "exfoliat", "mụn"],
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
                "barrier", "repair", "xịt khoáng",
                "phục hồi", "nhạy cảm", "soothing", "calming"],
}

ADVICE = {
    "inflammatory acne": "Ưu tiên sản phẩm có BHA (salicylic acid) hoặc benzoyl peroxide. Mụn viêm nặng nên đi khám bác sĩ da liễu.",
    "non inflammatory acne black heads": "BHA giúp làm sạch sâu lỗ chân lông; có thể dùng mặt nạ đất sét 1–2 lần/tuần.",
    "non inflammatory acne white heads": "Tẩy tế bào chết hóa học (AHA/BHA) đều đặn giúp giảm mụn đầu trắng.",
    "pores": "Niacinamide và BHA giúp kiềm dầu và làm lỗ chân lông trông nhỏ hơn.",
    "pigmentation": "Vitamin C, niacinamide giúp làm sáng da. Kem chống nắng mỗi ngày là bắt buộc.",
    "dark spots": "Vitamin C, niacinamide, arbutin giúp mờ thâm. Nhớ dùng kem chống nắng.",
    "wrinkles": "Retinol và peptide hỗ trợ chống lão hóa; dùng retinol buổi tối và chống nắng ban ngày.",
    "redness": "Chọn sản phẩm làm dịu, phục hồi (rau má, panthenol, ceramide), tránh sản phẩm có cồn/hương liệu.",
}


@st.cache_resource
def load_model():
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
    df = df[df["type"] != "Dụng cụ"]   # bỏ bông tẩy trang, khăn lau...
    text = df["name"] + " " + df["description"]
    df["_text"] = text.map(lambda t: unicodedata.normalize("NFC", t).lower())
    return df


def predict(model, class_names, img):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.asarray(img, dtype=np.float32), 0)
    probs = model.predict(arr, verbose=0)[0]
    return {c: float(p) for c, p in zip(class_names, probs)}


def recommend(df, probs):
    # Lấy các vấn đề có xác suất >= THRESHOLD; nếu không có thì lấy vấn đề cao nhất
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


def fmt_price(x):
    try:
        return f"{int(float(x)):,}".replace(",", ".") + "đ"
    except (ValueError, TypeError):
        return str(x)


# ------------------------- GIAO DIỆN -------------------------
st.title("🧴 Phân tích da & gợi ý sản phẩm")
st.caption("Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ da liễu.")

model, class_names = load_model()
products = load_products()

tab1, tab2 = st.tabs(["📷 Chụp ảnh", "🖼️ Tải ảnh lên"])
with tab1:
    shot = st.camera_input("Chụp cận mặt, đủ sáng, không filter")
with tab2:
    upload = st.file_uploader("Chọn ảnh mặt", type=["jpg", "jpeg", "png", "webp"])

file = shot or upload
if file:
    with st.spinner("Đang phân tích..."):
        probs = predict(model, class_names, Image.open(file))

    st.subheader("Kết quả phân tích")
    for c, p in sorted(probs.items(), key=lambda t: -t[1]):
        st.progress(min(max(p, 0.0), 1.0), text=f"{VI_NAMES.get(c.lower(), c)}: {p:.0%}")

    recs, active = recommend(products, probs)

    st.subheader("Lời khuyên")
    for c in active:
        st.info(f"**{VI_NAMES.get(c.lower(), c)}:** {ADVICE.get(c.lower(), '')}")

    st.subheader("Sản phẩm gợi ý")
    if recs.empty:
        st.warning("Chưa tìm thấy sản phẩm phù hợp trong products.csv.")
    else:
        cols = st.columns(2)
        for i, (_, r) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    if r["image"]:
                        st.image(r["image"])
                    if r["type"]:
                        st.caption(f"{r['type']} · {r['brand']}")
                    st.markdown(f"**{r['name']}**")
                    if r["price"] != "":
                        line = f"💰 **{fmt_price(r['price'])}**"
                        if r["old_price"] != "":
                            line += f"  ~~{fmt_price(r['old_price'])}~~"
                        st.markdown(line)
                    st.caption(f"Phù hợp với: {r['_why']}")
                    if r["url"]:
                        st.link_button("🛒 Mua tại Guardian", r["url"])
