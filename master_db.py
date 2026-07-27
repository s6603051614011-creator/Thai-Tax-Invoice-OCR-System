"""
master_db.py -- ที่เก็บถาวร (SQLite) ของ master list ฝั่ง production
=====================================================================
build_master() ใน master_list.py (train-only, ไม่มี DB) ยังใช้แบบเดิมสำหรับ
eval_combined.py / eval_hires_infer.py เพื่อให้ตัวเลขที่รายงานไปแล้ว (42.8% ->
72.3%) reproduce ได้เสมอ ไม่ขึ้นกับข้อมูลจากการใช้งานจริงที่งอกเพิ่มขึ้นเรื่อยๆ

ไฟล์นี้แยกต่างหากสำหรับ api.py (production) เท่านั้น: seed ครั้งแรกจาก train
set ชุดเดียวกับ build_master() แล้วหลังจากนั้นให้พนักงานยืนยัน/แก้ไขข้อมูลจริง
ผ่านปุ่มบันทึกในหน้าเว็บ (POST /confirm) เพิ่มเข้ามาเรื่อยๆ -- ค่าที่ยืนยันล่าสุด
ชนะเสมอ (คนตรวจแล้ว น่าเชื่อกว่าค่าที่นับจาก train)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("dataset/master_list.db")

ANNOTATIONS = Path("dataset/annotations_auto.json")
SPLITS      = Path("dataset/splits.json")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS master_entries (
            side      TEXT NOT NULL,
            norm_name TEXT NOT NULL,
            name_th   TEXT NOT NULL DEFAULT '',
            name_en   TEXT NOT NULL DEFAULT '',
            tax_id    TEXT NOT NULL DEFAULT '',
            address   TEXT NOT NULL DEFAULT '',
            source    TEXT NOT NULL DEFAULT 'train',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (side, norm_name)
        )
    """)
    return conn


def _seed_from_train_if_empty(conn: sqlite3.Connection) -> None:
    """รันครั้งแรกที่ DB ว่าง -- copy จาก train set ชุดเดียวกับ build_master()
    (import ในฟังก์ชันกันวนลูป import กับ master_list.py)"""
    n = conn.execute("SELECT COUNT(*) FROM master_entries").fetchone()[0]
    if n > 0:
        return

    from master_list import build_master  # lazy import, กันวนลูป

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for side in ("seller", "buyer"):
        for norm_name, entry in build_master(side).items():
            rows.append((side, norm_name, entry["name_th"], entry["name_en"],
                        entry["tax_id"], entry["address"], "train", now))
    conn.executemany(
        "INSERT OR IGNORE INTO master_entries "
        "(side, norm_name, name_th, name_en, tax_id, address, source, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    print(f"master_db: seed จาก train เสร็จ ({len(rows)} รายการ)", flush=True)


def load_master(side: str) -> dict:
    """คืน dict รูปแบบเดียวกับ build_master() -- norm_name -> {tax_id, address, name_th, name_en}
    ใช้ใน api.py แทน build_master() ตรงๆ เพื่อให้ได้ข้อมูลที่โตขึ้นจากการยืนยันจริงด้วย"""
    conn = _connect()
    try:
        _seed_from_train_if_empty(conn)
        cur = conn.execute(
            "SELECT norm_name, name_th, name_en, tax_id, address FROM master_entries WHERE side = ?",
            (side,))
        return {
            norm_name: {"name_th": name_th, "name_en": name_en, "tax_id": tax_id, "address": address}
            for norm_name, name_th, name_en, tax_id, address in cur.fetchall()
        }
    finally:
        conn.close()


def save_entry(side: str, norm_name: str, name_th: str, name_en: str,
              tax_id: str, address: str) -> None:
    """บันทึก/อัปเดตข้อมูลที่ผู้ใช้ยืนยันแล้ว -- ค่าล่าสุดชนะเสมอ (คนตรวจแล้ว)"""
    if not norm_name:
        return
    conn = _connect()
    try:
        conn.execute("""
            INSERT INTO master_entries (side, norm_name, name_th, name_en, tax_id, address, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'verified', ?)
            ON CONFLICT(side, norm_name) DO UPDATE SET
                name_th=excluded.name_th, name_en=excluded.name_en,
                tax_id=excluded.tax_id, address=excluded.address,
                source='verified', updated_at=excluded.updated_at
        """, (side, norm_name, name_th, name_en, tax_id, address,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()
    finally:
        conn.close()
