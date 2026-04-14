import pytesseract
import cv2
import os
import numpy as np
import re

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

LANG   = 'snd'
CONFIG = '--oem 3 --psm 3'

# ================================================================
# Detect whether the image is a clean screenshot or a physical scan
# ================================================================

def is_scan(image_path):
    """
    Screenshots have near-zero background variance (pure white/black).
    Scans have speckled paper texture — much higher background variance.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Sample corners (far from text) to measure background noise
    h, w = img.shape
    margin = min(h, w) // 8
    corners = [
        img[:margin, :margin],
        img[:margin, w-margin:],
        img[h-margin:, :margin],
        img[h-margin:, w-margin:],
    ]
    variances = [np.var(c.astype(float)) for c in corners]
    avg_var = np.mean(variances)
    # Screenshots: avg_var < 50. Scans: avg_var > 200 typically.
    return avg_var > 80

# ================================================================
# SCAN pipeline — aggressive noise/blur correction
# ================================================================

def deskew(gray):
    """
    Correct slight rotation from physical page placement.
    Uses Hough line detection to find the dominant text angle.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only use near-horizontal lines (text lines)
            if abs(angle) < 15:
                angles.append(angle)

    if not angles:
        return gray

    median_angle = np.median(angles)
    # Don't over-correct — ignore tiny skews below 0.3°
    if abs(median_angle) < 0.3:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    print(f"  Deskew: corrected {median_angle:.2f}°")
    return rotated

def upscale_to_300dpi(gray, estimated_current_dpi=150, target_dpi=300):
    """
    Scans at 150 DPI need 2x upscale to reach 300 DPI.
    300 DPI is the minimum Tesseract needs for reliable RTL OCR.
    Uses INTER_CUBIC which preserves thin strokes better than linear.
    """
    h, w = gray.shape
    # Only upscale if image would benefit (smaller than ~2400px wide)
    if w >= 2400:
        return gray
    scale = target_dpi / estimated_current_dpi
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    print(f"  Upscale: {w}×{h} → {upscaled.shape[1]}×{upscaled.shape[0]}")
    return upscaled

def remove_paper_texture(gray):
    """
    Scanned paper has a gray speckle background.
    Morphological opening removes the texture while preserving text strokes.
    """
    # Small kernel — just big enough to lift the paper grain
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Subtract background to make it cleaner white
    background = cv2.dilate(opened, kernel, iterations=3)
    diff = cv2.subtract(background, opened)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def sharpen_for_diacritics(gray):
    """
    Unsharp masking — gentler than a Laplacian kernel.
    Critical for Sindhi: restores clarity on small dots (diacritics)
    that get blurred in scans. Laplacian kernels over-sharpen and
    create halos around thin strokes.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    return sharpened

def binarize(gray):
    """
    Otsu binarization after all the above cleaning steps.
    Works better on scans than adaptive threshold because
    the paper texture has already been removed.
    """
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def preprocess_scan(image_path):
    """Full scan pipeline: deskew → upscale → denoise → sharpen → binarize"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = deskew(gray)
    gray = upscale_to_300dpi(gray)
    gray = remove_paper_texture(gray)
    gray = sharpen_for_diacritics(gray)
    gray = binarize(gray)
    return gray

# ================================================================
# SCREENSHOT pipeline — same as before (already 93%)
# ================================================================

def preprocess_screenshot(image_path):
    """Minimal processing for clean digital screenshots."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1600:
        scale = 1600 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(denoised, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    return thresh

# ================================================================
# Quality metrics
# ================================================================

def is_sindhi_char(c):
    return (
        '\u0600' <= c <= '\u06ff' or
        '\u0750' <= c <= '\u077f' or
        '\ufb50' <= c <= '\ufdff'
    )

def quality_report(text):
    words = text.split()
    if not words:
        return {}
    sindhi_words, junk_words, total_chars, sindhi_chars = 0, 0, 0, 0
    for word in words:
        sc = sum(1 for c in word if is_sindhi_char(c))
        total_chars += len(word)
        sindhi_chars += sc
        ratio = sc / max(len(word), 1)
        if ratio >= 0.7:
            sindhi_words += 1
        elif ratio == 0 and not word.isdigit():
            junk_words += 1
    return {
        'total_words':  len(words),
        'sindhi_words': sindhi_words,
        'junk_words':   junk_words,
        'sindhi_ratio': sindhi_chars / max(total_chars, 1),
        'avg_word_len': sum(len(w) for w in words) / len(words),
    }

# ================================================================
# Main — auto-detects scan vs screenshot and applies correct pipeline
# ================================================================

def run(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"\n{'='*60}")
    print("SINDHI OCR — AUTO SCAN / SCREENSHOT DETECTION")
    print('='*60)
    print(f"Image: {image_path}")

    scan = is_scan(image_path)
    print(f"Type detected: {'SCAN (physical)' if scan else 'SCREENSHOT (digital)'}")
    print(f"Pipeline: {'scan (deskew+denoise+sharpen)' if scan else 'screenshot (adaptive threshold)'}")
    print('-'*60)

    if scan:
        processed = preprocess_scan(image_path)
    else:
        processed = preprocess_screenshot(image_path)

    temp_path = "_temp_sindhi_auto.png"
    cv2.imwrite(temp_path, processed)

    raw = pytesseract.image_to_string(temp_path, lang=LANG, config=CONFIG)
    text = re.sub(r'\s+', ' ', raw).strip()

    if os.path.exists(temp_path):
        os.remove(temp_path)

    q = quality_report(text)
    print(f"\nResults:")
    print(f"  Total words   : {q.get('total_words', 0)}")
    print(f"  Sindhi words  : {q.get('sindhi_words', 0)}")
    print(f"  Junk words    : {q.get('junk_words', 0)}")
    print(f"  Sindhi ratio  : {q.get('sindhi_ratio', 0):.1%}")
    print(f"  Avg word len  : {q.get('avg_word_len', 0):.1f}")

    out_file = "output_sindhi_auto.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSaved → {out_file}")
    print('='*60)
    return text


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ").strip()
    run(path)