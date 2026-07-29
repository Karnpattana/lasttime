# -*- coding: utf-8 -*-
"""
ตัดพื้นหลัง -> วัดขนาด -> แยกสี -> เซฟ CSV -> แสดงหน้าต่างดูภาพ

หน้าต่าง
  3 Colors - ภาพใบผักที่ตัดพื้นหลังแล้ว ระบายสีตามกลุ่มแบบโปร่งใส

ปุ่ม
  D = รูปถัดไป   A = รูปก่อนหน้า   S = เซฟ CSV ซ้ำ   ESC = ออก

การสลับแกน
  A0, A180, B0, B180   -> วัดปกติ
  A90, A270, B90, B270 -> สลับ height <-> length
"""

import cv2
import numpy as np
import glob
import os
import re
import csv

# ---------------- ตั้งค่าเส้นทาง ----------------
IN_FOLDER = r"E:\CV\D1"
OUT_CSV   = r"E:\CV\fntest\features.csv"

# ---------------- ค่าตัดพื้นหลัง (ล็อกไว้) ----------------
CHROMA_MIN = 4      # ความอิ่มสีขั้นต่ำ
L_MIN      = 7      # ความสว่างขั้นต่ำ
KERNEL     = 11     # ขนาดตัวกวาด (เลขคี่)
CLOSE_ITER = 4      # จำนวนรอบปิดรู
MIN_AREA   = 500    # ตัดชิ้นเล็กกว่านี้ทิ้ง
ROOT_A     = 126    # ค่า a ที่ถือว่าเป็นน้ำตาล = ราก
ROOT_GROW  = 1      # ขยายขอบเขตที่ตัดราก
PAD        = 10     # เว้นขอบรอบวัตถุ

# ---------------- ค่าแยกสี (ล็อกไว้) ----------------
BLACK_V_LO = 5      # ดำ: V ต่ำสุด
BLACK_V_HI = 27     # ดำ: V สูงสุด
WHITE_S    = 96     # S ต่ำกว่านี้ = ขาว
BROWN_LO   = 0      # Hue น้ำตาล เริ่ม
BROWN_HI   = 30     # Hue น้ำตาล จบ
ALPHA      = 12     # ความทึบของสีที่ระบายทับ (%)

VIEW_W = 640        # ความกว้างหน้าต่าง (ย่อเพื่อดูเท่านั้น ไม่กระทบค่า)

# ---------------- สีที่ใช้ระบาย (BGR) ----------------
COL_GREEN = (0, 200,   0)
COL_BLACK = (0,   0, 255)     # ดำ -> โชว์เป็นแดง จะได้เห็นบนพื้นมืด
COL_WHITE = (255, 255, 255)
COL_BROWN = (19,  69, 139)

COLUMNS = ["filename", "number", "day", "height_px", "length_px", "area_px",
           "green_area_px", "black_area_px", "white_area_px", "brown_area_px",
           "green_intensity"]

WIN = "3 Colors"


# ================= ตัดพื้นหลัง =================
def make_mask(img):
    """คืน mask: ขาว = ใบผัก, ดำ = พื้นหลัง/ราก"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = cv2.split(lab)
    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

    mask = ((chroma > CHROMA_MIN) & (L > L_MIN)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)                          # ลบจุดรบกวน
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITER)  # ปิดรูในใบ

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean[lbl == i] = 255
    mask = clean

    # ตัดก้อนราก
    rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    root = ((A >= ROOT_A) & (mask > 0)).astype(np.uint8) * 255
    root = cv2.morphologyEx(root, cv2.MORPH_OPEN, rk)
    root = cv2.morphologyEx(root, cv2.MORPH_CLOSE, rk, iterations=2)

    rn, rlbl, rst, _ = cv2.connectedComponentsWithStats(root, 8)
    if rn > 1:
        big = 1 + int(np.argmax(rst[1:, cv2.CC_STAT_AREA]))
        root = np.zeros_like(mask)
        root[rlbl == big] = 255
        if ROOT_GROW > 0:
            root = cv2.dilate(root, rk, iterations=ROOT_GROW)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(root))

    return mask


def crop_to_object(img, mask):
    """crop ตามขอบวัตถุ คืน (ภาพสี, mask) หรือ None"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    y0 = max(int(ys.min()) - PAD, 0)
    y1 = min(int(ys.max()) + 1 + PAD, mask.shape[0])
    x0 = max(int(xs.min()) - PAD, 0)
    x1 = min(int(xs.max()) + 1 + PAD, mask.shape[1])
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


# ================= วัดขนาด =================
def measure(mask):
    """
    height_px : แกน Y - ทุกคอลัมน์ x หาระยะบนสุด-ล่างสุด -> เอาค่ามากสุด (x เดียวกันแน่นอน)
    length_px : แกน X - ทุกแถว y หาระยะซ้ายสุด-ขวาสุด -> เอาค่ามากสุด (y เดียวกันแน่นอน)
    """
    m = mask > 0
    if not m.any():
        return None

    has_col  = m.any(axis=0)
    top      = np.argmax(m, axis=0)
    bottom   = m.shape[0] - 1 - np.argmax(m[::-1, :], axis=0)
    span_col = np.where(has_col, bottom - top + 1, 0)
    hx = int(span_col.argmax())

    has_row  = m.any(axis=1)
    left     = np.argmax(m, axis=1)
    right    = m.shape[1] - 1 - np.argmax(m[:, ::-1], axis=1)
    span_row = np.where(has_row, right - left + 1, 0)
    ly = int(span_row.argmax())

    return dict(height_px=int(span_col[hx]), hx=hx,
                length_px=int(span_row[ly]), ly=ly)


# ================= แยกสี =================
def split_colors(crop_img, crop_mask):
    """คืน (dict ค่าตัวเลข, dict มาสก์แต่ละสี)"""
    m = crop_mask > 0
    area = int(m.sum())
    if area == 0:
        return None, None

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # แบ่งเป็นชั้น ไม่ทับกัน รวมกันได้ area พอดี
    black = m & (V >= BLACK_V_LO) & (V <= BLACK_V_HI)
    rest  = m & ~black
    white = rest & (S < WHITE_S)
    rest  = rest & ~white
    brown = rest & (Hh >= BROWN_LO) & (Hh <= BROWN_HI)
    green = rest & ~brown                      # ที่เหลือทั้งหมด = เขียว (รวมเขียวอมเหลือง)

    # ความเข้มสีเขียว ExG = 2G - R - B (ยิ่งสูง = เขียวเข้ม)
    b, g, r = cv2.split(crop_img.astype(np.int16))
    exg = 2 * g - r - b

    vals = dict(
        area_px=area,
        black_area_px=int(black.sum()), green_area_px=int(green.sum()),
        white_area_px=int(white.sum()), brown_area_px=int(brown.sum()),
        green_intensity=float(exg[m].mean()),
    )
    masks = dict(black=black, green=green, white=white, brown=brown)
    return vals, masks


# ================= วาดภาพ =================
def draw_colors(crop_img, crop_mask, masks):
    """ระบายสีตามกลุ่มแบบโปร่งใส ทับบนภาพจริง"""
    tint = np.zeros_like(crop_img)
    tint[masks["green"]] = COL_GREEN
    tint[masks["black"]] = COL_BLACK
    tint[masks["white"]] = COL_WHITE
    tint[masks["brown"]] = COL_BROWN

    a = ALPHA / 100.0
    out = cv2.addWeighted(crop_img, 1 - a, tint, a, 0)
    out[crop_mask == 0] = (0, 0, 0)          # นอกใบ = ดำสนิท
    return out


def fit(im, w=VIEW_W):
    """ย่อภาพให้กว้าง w โดยคงอัตราส่วน (แค่เพื่อดู ไม่กระทบค่า)"""
    if im.shape[1] <= w:
        return im
    return cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))


# ================= อ่านชื่อไฟล์ =================
def parse_name(name):
    """
    goak15_D6_A0.png -> (15, 6, 'A0', False)
    swap=True เมื่อมุมเป็น 90/270 (ภาพตะแคง ต้องสลับ height/length)
    """
    mn = re.search(r"goak(\d+)", name, re.I)
    md = re.search(r"_D(\d+)", name, re.I)
    ma = re.search(r"_([AB])(\d+)", name, re.I)

    number = int(mn.group(1)) if mn else -1
    day    = int(md.group(1)) if md else -1
    if ma:
        angle = ma.group(1).upper() + ma.group(2)
        swap  = int(ma.group(2)) in (90, 270)
    else:
        angle, swap = "", False
    return number, day, angle, swap


# ================= ประมวลผล 1 ไฟล์ =================
def process(path):
    """คืน (dict ค่าทั้งหมด, ภาพระบายสี) หรือ (None, None)"""
    name = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        return None, None

    mask = make_mask(img)
    cropped = crop_to_object(img, mask)
    if cropped is None:
        return None, None

    crop_img, crop_mask = cropped
    r = measure(crop_mask)
    v, masks = split_colors(crop_img, crop_mask)
    colors = draw_colors(crop_img, crop_mask, masks)

    number, day, angle, swap = parse_name(name)

    h, l = r["height_px"], r["length_px"]
    if swap:                      # ภาพตะแคง 90/270 -> สลับแกน
        h, l = l, h

    row = dict(
        filename=name, number=number, day=day, angle=angle, swap=swap,
        height_px=h, length_px=l,
        area_px=v["area_px"],
        green_area_px=v["green_area_px"], black_area_px=v["black_area_px"],
        white_area_px=v["white_area_px"], brown_area_px=v["brown_area_px"],
        green_intensity=round(v["green_intensity"], 2),
    )
    return row, colors


# ================= แสดงผล / เซฟ =================
def report(row):
    pct = lambda n: n / row["area_px"] * 100
    tag = "  <- สลับแกน (90/270)" if row["swap"] else ""
    print(f"\n=== {row['filename']} ===  goak{row['number']} D{row['day']} {row['angle']}{tag}")
    print(f"  height_px        {row['height_px']:>9,}")
    print(f"  length_px        {row['length_px']:>9,}")
    print(f"  area_px          {row['area_px']:>9,}")
    print(f"  green_area_px    {row['green_area_px']:>9,}   {pct(row['green_area_px']):5.1f}%")
    print(f"  black_area_px    {row['black_area_px']:>9,}   {pct(row['black_area_px']):5.1f}%")
    print(f"  white_area_px    {row['white_area_px']:>9,}   {pct(row['white_area_px']):5.1f}%")
    print(f"  brown_area_px    {row['brown_area_px']:>9,}   {pct(row['brown_area_px']):5.1f}%")
    print(f"  green_intensity  {row['green_intensity']:>9.2f}")


def save_csv(files):
    """รันทุกไฟล์ -> เขียน CSV"""
    print(f"\nประมวลผล {len(files)} ไฟล์...")
    rows = []
    for i, f in enumerate(files, 1):
        row, _ = process(f)
        if row is None:
            print(f"[{i}/{len(files)}] ข้าม: {os.path.basename(f)}")
            continue
        rows.append(row)
        print(f"[{i}/{len(files)}] {row['filename']:<26} "
              f"H={row['height_px']:>4} L={row['length_px']:>4}"
              f"{'  (สลับแกน)' if row['swap'] else ''}")

    rows.sort(key=lambda x: (x["number"], x["day"], x["filename"]))

    out_dir = os.path.dirname(OUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nเขียน CSV แล้ว: {OUT_CSV}  ({len(rows)} แถว)")


# ================= main =================
def main():
    files = sorted(glob.glob(os.path.join(IN_FOLDER, "*.png")))
    if not files:
        raise SystemExit("ไม่พบไฟล์รูปในโฟลเดอร์ " + IN_FOLDER)

    save_csv(files)          # <<< รันปุ๊บเซฟเลย

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WIN, 60, 60)
    print("\nเปิดหน้าต่างดูภาพ:  A=ก่อนหน้า  D=ถัดไป  S=เซฟ CSV ซ้ำ  ESC=ออก")

    idx = 0
    reload_img = True
    colors = None

    while True:
        if reload_img:
            idx %= len(files)
            row, colors = process(files[idx])
            if row is None:
                print("ข้าม:", os.path.basename(files[idx]))
                idx += 1
                continue
            report(row)
            reload_img = False

        cv2.imshow(WIN, fit(colors))

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k in (ord('d'), ord('D')):
            idx += 1; reload_img = True
        elif k in (ord('a'), ord('A')):
            idx -= 1; reload_img = True
        elif k in (ord('s'), ord('S')):
            save_csv(files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()