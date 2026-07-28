"""
store.py -- ฐานข้อมูลเก็บใบกำกับภาษีที่ตรวจแล้ว (SQLite)
=====================================================================
ก่อนหน้านี้ข้อมูลอยู่ในหน่วยความจำเบราว์เซอร์อย่างเดียว ปิดแท็บแล้วหาย
ไฟล์นี้ทำให้ "กดบันทึก" แล้วเก็บถาวรจริง และเอาไปสรุปยอดรายเดือน/รายปีได้

เลือก SQLite เพราะ:
  - มากับ Python อยู่แล้ว ไม่ต้องติดตั้ง/ดูแล server เพิ่ม
  - ระบบนี้รันบนโน้ตบุ๊กเครื่องเดียว (เครื่องที่มี GPU) ผู้ใช้ไม่กี่คน
  - ได้ไฟล์เดียว (data/invoices.db) ก๊อปไปสำรองได้ตรงๆ

ใช้:
    import store
    store.init()
    store.save_invoice(payload)        # payload = รูปแบบเดียวกับที่ frontend ส่งมา
    store.summary()                    # ยอดรวมรายเดือน/รายปี
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from thaidate import parse_thai_date

DB_PATH = Path(__file__).parent / "data" / "invoices.db"

SCALARS = ["invoice_number", "invoice_date", "seller_name_th", "seller_name_en",
           "seller_tax_id", "seller_address", "buyer_name_th", "buyer_name_en",
           "buyer_tax_id", "buyer_address"]
MONEY = ["subtotal", "discount", "vat", "grand_total"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_at       TEXT NOT NULL,
    filename       TEXT,
    invoice_number TEXT, invoice_date TEXT,
    date_ce        TEXT,              -- 'YYYY-MM-DD' (ค.ศ.) หรือ NULL ถ้าอ่านวันที่ไม่ออก
    seller_name_th TEXT, seller_name_en TEXT, seller_tax_id TEXT, seller_address TEXT,
    buyer_name_th  TEXT, buyer_name_en  TEXT, buyer_tax_id  TEXT, buyer_address  TEXT,
    subtotal REAL, discount REAL, vat REAL, grand_total REAL,
    flags          TEXT               -- JSON array ของข้อสังเกตที่ยังค้างอยู่ตอนบันทึก
);
CREATE TABLE IF NOT EXISTS items (
    invoice_id  INTEGER NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    description TEXT, quantity TEXT,
    unit_price REAL, discount REAL, amount REAL
);
CREATE INDEX IF NOT EXISTS ix_invoices_date ON invoices(date_ce);
CREATE INDEX IF NOT EXISTS ix_invoices_key ON invoices(invoice_number, seller_tax_id);
-- เคยใช้ index กันซ้ำที่ (เลขที่ใบ + ผู้ขาย) แต่ผิด: ในข้อมูลจริงมีคนละใบที่เลขที่ซ้ำกัน
-- (inv_026/035 คนละวันคนละยอด) การกันซ้ำแบบนั้นทำให้ใบจริงหายไป -- ถอดออก
DROP INDEX IF EXISTS ux_invoice_key;
"""


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init():
    with connect() as con:
        con.executescript(SCHEMA)


def _num(v):
    """'1,234.56' -> 1234.56 ; อ่านไม่ออก/ว่าง -> None (ไม่ใช่ 0 เพราะจะทำให้ยอดรวมเพี้ยน)"""
    s = str(v or "").replace(",", "").replace(" ", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def save_invoice(payload: dict) -> dict:
    """
    payload: {"filename":..., "fields": {...}, "items": [...], "flags": [...]}

    *** เรื่องใบซ้ำ ***
    ถ่ายใบเดิมซ้ำ/กดอ่านใหม่ ไม่ควรทำให้ยอดในรายงานเบิ้ลสองเท่า แต่จะกันซ้ำด้วย
    "เลขที่ใบ + ผู้ขาย" เฉยๆ ไม่ได้ -- ในข้อมูลจริงมีคนละใบที่เลขที่ซ้ำกัน
    (inv_026 กับ inv_035 เลขที่ NGH56/020001 เหมือนกัน แต่คนละวัน ยอด 215,200 กับ 2,461)
    ถ้ากันซ้ำแบบนั้นใบจริงจะหายไปเงียบๆ

    จึงเขียนทับเฉพาะเมื่อ "เหมือนกันทั้ง 4 อย่าง": เลขที่ใบ + ผู้ขาย + วันที่ + ยอดรวม
    ถ้าเลขที่ซ้ำแต่ที่เหลือไม่ตรง -> บันทึกเป็นใบใหม่ แล้วคืน dup_number=True
    ให้หน้าเว็บเตือนผู้ใช้ไปตรวจเอง (ยอมมีแถวเกินให้เห็นดีกว่าทำข้อมูลหาย)
    """
    f = payload.get("fields") or {}
    items = payload.get("items") or []

    parsed = parse_thai_date(f.get("invoice_date"))
    date_ce = f"{parsed[0]:04d}-{parsed[1]:02d}-{parsed[2]:02d}" if parsed else None

    cols = ["saved_at", "filename", "date_ce", "flags"] + SCALARS + MONEY
    vals = [datetime.now(timezone.utc).isoformat(timespec="seconds"),
            payload.get("filename") or "", date_ce,
            json.dumps(payload.get("flags") or [], ensure_ascii=False)]
    vals += [str(f.get(k) or "") for k in SCALARS]
    vals += [_num(f.get(k)) for k in MONEY]

    with connect() as con:
        num = str(f.get("invoice_number") or "").strip()
        tax = str(f.get("seller_tax_id") or "").strip()
        total = _num(f.get("grand_total"))
        replaced, dup_number = False, False

        if num:
            same_num = con.execute(
                "SELECT id, date_ce, grand_total FROM invoices"
                " WHERE invoice_number = ? AND seller_tax_id = ?", (num, tax)).fetchall()
            for row in same_num:
                if row["date_ce"] == date_ce and row["grand_total"] == total:
                    con.execute("DELETE FROM items WHERE invoice_id = ?", (row["id"],))
                    con.execute("DELETE FROM invoices WHERE id = ?", (row["id"],))
                    replaced = True
            dup_number = bool(same_num) and not replaced

        cur = con.execute(
            f"INSERT INTO invoices ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        inv_id = cur.lastrowid
        for i, it in enumerate(items, 1):
            con.execute(
                "INSERT INTO items (invoice_id, seq, description, quantity, unit_price, discount, amount)"
                " VALUES (?,?,?,?,?,?,?)",
                (inv_id, i, str(it.get("description") or ""), str(it.get("quantity") or ""),
                 _num(it.get("unit_price")), _num(it.get("discount")), _num(it.get("amount"))))

    return {"id": inv_id, "replaced": replaced, "dup_number": dup_number, "date_ce": date_ce}


def _rows(con, group_expr, where_extra=""):
    return [dict(r) for r in con.execute(f"""
        SELECT {group_expr} AS period,
               COUNT(*)              AS count,
               COALESCE(SUM(subtotal), 0)    AS subtotal,
               COALESCE(SUM(vat), 0)         AS vat,
               COALESCE(SUM(grand_total), 0) AS grand_total
        FROM invoices
        WHERE date_ce IS NOT NULL {where_extra}
        GROUP BY period ORDER BY period
    """)]


def summary() -> dict:
    """ยอดรวมรายเดือน + รายปี สำหรับหน้ากราฟ"""
    with connect() as con:
        monthly = _rows(con, "substr(date_ce, 1, 7)")
        yearly = _rows(con, "substr(date_ce, 1, 4)")
        t = con.execute("""
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(grand_total), 0) AS grand_total,
                   COALESCE(SUM(vat), 0)         AS vat,
                   SUM(CASE WHEN date_ce IS NULL THEN 1 ELSE 0 END) AS undated
            FROM invoices
        """).fetchone()
    return {"monthly": monthly, "yearly": yearly, "totals": dict(t)}


def list_invoices(limit: int = 500) -> list:
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT id, saved_at, filename, invoice_number, invoice_date, date_ce,"
            " seller_name_th, grand_total, vat FROM invoices"
            " ORDER BY COALESCE(date_ce, '') DESC, id DESC LIMIT ?", (limit,))]


def stats() -> dict:
    with connect() as con:
        n = con.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
        m = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    return {"invoices": n, "items": m, "db": str(DB_PATH)}
