import io
import re
from typing import List, Dict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import pandas as pd

# ---------- OCR & Parsing Helpers ----------

PHONE_RE = re.compile(r"(?:\+?\d[\s-]?){10,14}\d")

# Noise words to ignore
NOISE_PHRASES = [
    "hindustan petroleum", "corporation", "limited",
    "km from your location", "closed for the day",
    "map", "website", "☎", "📍", "📞", "nearby", "distance",
]

# ✅ Generic address hints (no city names)
ADDRESS_HINTS = [
    "ground floor", "near", "road", "chowk", "square", "mandir", "temple",
    "sector", "phase", "street", "st.", "lane", "plot", "ward",
    "pin", "pincode", "district", "colony", "market", "complex", "building",
    "opp", "opposite", "beside", "above", "block", "area", "city", "state"
]

def preprocess_image(img: Image.Image) -> Image.Image:
    """Light cleanup to help OCR."""
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    w, h = img.size
    if max(w, h) < 1500:
        scale = 1500 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    img = img.filter(ImageFilter.SHARPEN)
    return img

def clean_line(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" ,;:.|-–—\t\n\r").strip()
    return s

def is_noise_line(s: str) -> bool:
    low = s.lower()
    return any(p in low for p in NOISE_PHRASES) or len(low) <= 1

def looks_like_name(s: str) -> bool:
    low = s.lower()
    if any(p in low for p in NOISE_PHRASES): 
        return False
    if re.search(r"\d", s): 
        return False
    words = [w for w in re.split(r"\s+", s) if w]
    if len(words) < 2 or len(words) > 8:
        return False
    # Title-ish casing signal
    caps_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
    return caps_ratio >= 0.5

def normalize_phone(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw)
    # If exactly 10 digits and no country code, assume +91
    if re.fullmatch(r"\d{10}", digits):
        return "+91" + digits
    # Common 11–13 digits formats; keep leading +
    if digits.startswith("+"):
        return digits
    if re.fullmatch(r"\d{11,13}", digits):
        return "+" + digits
    return raw.strip()

def extract_entries_from_text(text: str) -> List[Dict[str, str]]:
    lines = [clean_line(l) for l in text.splitlines()]
    lines = [l for l in lines if l]  # drop blanks

    # Find phone numbers
    phone_positions = []
    for i, l in enumerate(lines):
        if PHONE_RE.search(l):
            phone_positions.append(i)

    entries = []

    for p_idx, line_index in enumerate(phone_positions):
        # Phone
        phone_match = PHONE_RE.search(lines[line_index])
        phone = normalize_phone(phone_match.group(0)) if phone_match else ""

        # Find name above phone
        name = ""
        name_line_index = None
        for j in range(line_index - 1, max(-1, line_index - 10), -1):
            candidate = lines[j]
            if is_noise_line(candidate):
                continue
            if looks_like_name(candidate):
                name = candidate
                name_line_index = j
                break

        # Address = lines between name and phone
        start = (name_line_index + 1) if name_line_index is not None else max(0, line_index - 8)
        address_lines = []
        for k in range(start, line_index):
            c = lines[k]
            if is_noise_line(c):
                continue
            address_lines.append(c)

        # Keep lines that look address-like
        filtered_addr = []
        for l in address_lines:
            low = l.lower()
            if any(h in low for h in ADDRESS_HINTS) or re.search(r"\b\d{3}\s?\d{3}\b|\b\d{6}\b", low):
                filtered_addr.append(l)
            else:
                if len(l.split()) >= 2 and len(l) <= 80:
                    filtered_addr.append(l)

        # Remove duplicates
        seen = set()
        uniq_addr = []
        for l in filtered_addr:
            k = l.lower()
            if k not in seen:
                uniq_addr.append(l)
                seen.add(k)

        address = ", ".join(uniq_addr)
        address = re.sub(r"\s+,", ",", address)
        address = re.sub(r",\s*,", ", ", address).strip(" ,")

        entries.append({
            "Name": name,
            "Address": address,
            "Phone": phone
        })

    return entries

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Image → Excel (Petrol Pumps)", page_icon="⛽", layout="wide")
st.title("⛽ Petrol Pump Info Extractor — Image → Excel")

st.write(
    "Upload screenshots containing petrol pump details. I’ll OCR them and extract **Name, Address, Phone** into a table you can download as **Excel**."
)

with st.sidebar:
    st.header("Upload")
    files = st.file_uploader(
        "Upload screenshots (PNG/JPG/WebP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    show_text = st.checkbox("Show raw OCR text for debugging", value=False)

process_clicked = st.button("🔎 Process Images", type="primary", disabled=not files)

results: List[Dict[str, str]] = []

if process_clicked:
    if not files:
        st.warning("Please upload at least one image.")
    else:
        status = st.status("Running OCR and parsing…", expanded=True)
        try:
            for idx, f in enumerate(files, start=1):
                img = Image.open(f).convert("RGB")
                pimg = preprocess_image(img)

                # OCR
                text = pytesseract.image_to_string(pimg, lang="eng", config="--oem 3 --psm 6")
                if not text or len(text.strip()) < 10:
                    text = pytesseract.image_to_string(img, lang="eng")

                status.write(f"Parsed **{f.name}**")

                if show_text:
                    with st.expander(f"🧾 OCR text: {f.name}"):
                        st.code(text)

                entries = extract_entries_from_text(text)

                # Fallback if no entry found
                if not entries:
                    phones = PHONE_RE.findall(text)
                    phone = normalize_phone(phones[0]) if phones else ""
                    name = ""
                    for line in [clean_line(l) for l in text.splitlines() if clean_line(l)]:
                        if not is_noise_line(line) and looks_like_name(line):
                            name = line
                            break
                    addr_lines = []
                    for l in [clean_line(l) for l in text.splitlines() if clean_line(l)]:
                        if l != name and not is_noise_line(l):
                            addr_lines.append(l)
                    address = ", ".join(addr_lines[:6])
                    entries = [{"Name": name, "Address": address, "Phone": phone}]

                results.extend(entries)

            status.update(label="✅ Done", state="complete")
        except Exception as e:
            status.update(label="❌ Something went wrong", state="error")
            st.exception(e)

if results:
    st.subheader("Preview of Extracted Data")
    df = pd.DataFrame(results, columns=["Name", "Address", "Phone"])

    df = df.replace({"": None}).dropna(how="all")
    df = df.drop_duplicates(subset=["Name", "Phone"], keep="first")

    st.dataframe(df, use_container_width=True)

    # Export to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PetrolPumps")
    output.seek(0)

    st.download_button(
        label="📥 Download Excel",
        data=output.getvalue(),
        file_name="petrol_pumps.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Upload images and click **Process Images** to extract data.")
