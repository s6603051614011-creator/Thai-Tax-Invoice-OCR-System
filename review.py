"""
review.py -- ตรวจสอบและแก้ไข Ground Truth (ฟอร์มแก้ทีละ field)
=================================================================
รัน:
  python review.py
  python review.py --file dataset/annotations_auto.json

เปิด browser: http://localhost:5000

แก้ข้อมูลในฟอร์ม (ไม่ต้องแตะ JSON เอง) -> กด Verify -> ระบบ normalize +
สร้าง canonical JSON (training target) ให้อัตโนมัติ และบันทึกทันที

Keyboard:
  Ctrl+S            -- Save (ยังไม่ verify)
  Ctrl+Enter        -- Save + Verify + ไปใบถัดไป
  Alt+ArrowLeft/Right -- ใบก่อนหน้า/ถัดไป
"""

import json
import socket
import argparse
import webbrowser
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort

import schema

# ── Config ───────────────────────────────────────────────
DEFAULT_ANNOTATION_FILE = "dataset/annotations_auto.json"
IMAGE_BASE              = Path("dataset/raw")
PORT                    = 5000
# ─────────────────────────────────────────────────────────

app = Flask(__name__)
ANNOTATIONS: list[dict] = []
ANNOTATION_FILE: Path   = Path(DEFAULT_ANNOTATION_FILE)
SAVE_LOCK               = threading.Lock()  # กันไฟล์พังเมื่อหลายคนเซฟพร้อมกัน


def load_annotations(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] ไม่พบไฟล์ {path} -- เริ่มด้วย list ว่าง")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # merge fields บน empty_record เพื่อให้ทุก entry มี key ครบตาม schema
    for e in data:
        e["fields"] = schema.normalize_fields(e.get("fields", {}))
    return data


def save_annotations():
    # เขียนลง temp ก่อนแล้ว replace -> atomic กันไฟล์พังครึ่งทาง
    tmp = ANNOTATION_FILE.with_suffix(ANNOTATION_FILE.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ANNOTATIONS, f, ensure_ascii=False, indent=2)
    tmp.replace(ANNOTATION_FILE)


# ══════════════════════════════════════════════
# API
# ══════════════════════════════════════════════

@app.route("/api/schema")
def api_schema():
    return jsonify({
        "groups":      schema.UI_GROUPS,
        "labels":      schema.LABELS,
        "item_fields": schema.ITEM_FIELDS,
    })


@app.route("/api/count")
def api_count():
    total    = len(ANNOTATIONS)
    verified = sum(1 for a in ANNOTATIONS if a.get("verified"))
    return jsonify({"total": total, "verified": verified})


@app.route("/api/entry/<int:idx>")
def api_entry(idx: int):
    if idx < 0 or idx >= len(ANNOTATIONS):
        abort(404)
    e = ANNOTATIONS[idx]
    return jsonify({
        "idx":          idx,
        "id":           e.get("id", ""),
        "image_path":   e.get("image_path", ""),
        "verified":     e.get("verified", False),
        "auto_labeled": e.get("auto_labeled", False),
        "fields":       e.get("fields", schema.empty_record()),
    })


@app.route("/api/save/<int:idx>", methods=["POST"])
def api_save(idx: int):
    if idx < 0 or idx >= len(ANNOTATIONS):
        abort(404)
    data   = request.get_json()
    fields = schema.normalize_fields(data.get("fields", {}))
    verify = data.get("verify", False)

    with SAVE_LOCK:
        ANNOTATIONS[idx]["fields"]       = fields
        ANNOTATIONS[idx]["ground_truth"] = {"json": schema.build_canonical_json(fields)}
        if verify:
            ANNOTATIONS[idx]["verified"]     = True
            ANNOTATIONS[idx]["auto_labeled"] = False
        save_annotations()
    return jsonify({"ok": True, "verified": ANNOTATIONS[idx]["verified"]})


@app.route("/api/next_unverified/<int:from_idx>")
def api_next_unverified(from_idx: int):
    n = len(ANNOTATIONS)
    for offset in range(1, n + 1):
        idx = (from_idx + offset) % n
        if not ANNOTATIONS[idx].get("verified"):
            return jsonify({"idx": idx})
    return jsonify({"idx": -1})


@app.route("/image/<path:filename>")
def serve_image(filename: str):
    img_path = IMAGE_BASE / filename
    if not img_path.exists():
        abort(404)
    return send_file(img_path, mimetype="image/jpeg")


@app.route("/")
def index():
    return HTML_PAGE


# ══════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>Invoice Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

  #topbar { background: #16213e; padding: 8px 18px; display: flex; align-items: center; gap: 14px; border-bottom: 1px solid #0f3460; flex-shrink: 0; }
  #topbar h1 { font-size: 14px; color: #a0c4ff; white-space: nowrap; }
  #pbar-wrap { flex: 1; background: #0f3460; border-radius: 4px; height: 7px; }
  #pbar { height: 7px; background: #4cc9f0; border-radius: 4px; transition: width .3s; }
  #ptext, #einfo { font-size: 12px; color: #9fb3d1; white-space: nowrap; }
  .badge { font-size: 10px; padding: 2px 7px; border-radius: 9px; display: none; }
  #b-auto { background: #e67e22; color: #fff; }
  #b-ok   { background: #27ae60; color: #fff; }

  #main { display: flex; flex: 1; overflow: hidden; }

  #img-panel { flex: 1.1; background: #0c0c12; display: flex; align-items: flex-start; justify-content: center; overflow: auto; border-right: 1px solid #0f3460; }
  #img { max-width: 100%; cursor: zoom-in; }
  #img.zoom { max-width: 220%; cursor: zoom-out; }

  #edit { width: 460px; flex-shrink: 0; display: flex; flex-direction: column; background: #16213e; }
  #form { flex: 1; overflow-y: auto; padding: 12px 14px; }

  .group { margin-bottom: 14px; }
  .group h3 { font-size: 12px; color: #4cc9f0; border-bottom: 1px solid #0f3460; padding-bottom: 3px; margin-bottom: 7px; letter-spacing: .5px; }
  .field { display: flex; align-items: center; margin-bottom: 5px; gap: 8px; }
  .field label { width: 120px; font-size: 12px; color: #9fb3d1; flex-shrink: 0; text-align: right; }
  .field input { flex: 1; background: #0d1b2a; color: #e8eef5; border: 1px solid #1e3a5f; border-radius: 4px; padding: 5px 8px; font-size: 13px; outline: none; font-family: inherit; }
  .field input:focus { border-color: #4cc9f0; background: #0b1622; }
  .field input.money { text-align: right; font-family: 'Consolas', monospace; }

  /* items table */
  #items-wrap h3 { display: flex; justify-content: space-between; align-items: center; }
  #add-item { background: #2980b9; color: #fff; border: none; border-radius: 4px; font-size: 11px; padding: 2px 9px; cursor: pointer; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { color: #7b96b8; font-weight: 600; padding: 2px; text-align: left; }
  td { padding: 1px; }
  td input { width: 100%; background: #0d1b2a; color: #e8eef5; border: 1px solid #1e3a5f; border-radius: 3px; padding: 3px 4px; font-size: 11px; outline: none; }
  td input:focus { border-color: #4cc9f0; }
  td input.num { text-align: right; font-family: 'Consolas', monospace; }
  td.del { width: 18px; text-align: center; }
  .del-btn { background: none; border: none; color: #c0392b; cursor: pointer; font-size: 14px; font-weight: bold; }
  col.c-desc { width: 40%; } col.c-num { width: 14%; } col.c-del { width: 18px; }

  #sanity { font-size: 11px; padding: 6px 10px; margin-top: 6px; border-radius: 4px; display: none; }
  #sanity.ok   { display: block; background: #14361f; color: #5fd38a; }
  #sanity.warn { display: block; background: #3a2814; color: #f0a04b; }

  #actionbar { display: flex; gap: 7px; padding: 9px 12px; background: #0f2044; border-top: 1px solid #0f3460; flex-shrink: 0; }
  button.act { padding: 8px 12px; border: none; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; }
  button.act:hover { opacity: .88; }
  #b-prev, #b-next { background: #334; color: #aaa; }
  #b-save   { background: #2980b9; color: #fff; flex: 1; }
  #b-verify { background: #27ae60; color: #fff; flex: 1.3; }
  #b-skip   { background: #7d3c98; color: #fff; }

  #toast { position: fixed; bottom: 64px; right: 20px; background: #27ae60; color: #fff; padding: 9px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; opacity: 0; transition: opacity .3s; pointer-events: none; }
  #toast.show { opacity: 1; } #toast.err { background: #c0392b; }
</style>
</head>
<body>

<div id="topbar">
  <h1>Invoice Review</h1>
  <div id="pbar-wrap"><div id="pbar" style="width:0%"></div></div>
  <span id="ptext">0 / 0</span>
  <span id="einfo">--</span>
  <span id="b-auto" class="badge">AUTO</span>
  <span id="b-ok" class="badge">VERIFIED</span>
</div>

<div id="main">
  <div id="img-panel"><img id="img" src="" onclick="this.classList.toggle('zoom')" /></div>
  <div id="edit">
    <div id="form"></div>
    <div id="actionbar">
      <button class="act" id="b-prev" onclick="navigate(-1)">&#8592;</button>
      <button class="act" id="b-save" onclick="saveEntry(false)">Save</button>
      <button class="act" id="b-verify" onclick="saveEntry(true)">Verify &amp; Next</button>
      <button class="act" id="b-skip" onclick="jumpUnverified()">Unverified&#8594;</button>
      <button class="act" id="b-next" onclick="navigate(1)">&#8594;</button>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
let SCHEMA = null;
let currentIdx = 0;
let totalCount = 0;

const MONEY = new Set(['unit_price','discount','amount','subtotal','vat','grand_total']);

async function init() {
  SCHEMA = await (await fetch('/api/schema')).json();
  await loadCount();
  if (totalCount > 0) loadEntry(0);
}

async function loadCount() {
  const d = await (await fetch('/api/count')).json();
  totalCount = d.total;
  const pct = d.total ? (d.verified / d.total * 100) : 0;
  document.getElementById('pbar').style.width = pct + '%';
  document.getElementById('ptext').textContent = `${d.verified} / ${d.total} verified`;
}

function inputRow(key, value, isMoney) {
  const label = SCHEMA.labels[key] || key;
  const cls = isMoney ? 'money' : '';
  return `<div class="field"><label>${label}</label>`
       + `<input id="f-${key}" class="${cls}" value="${esc(value)}" oninput="checkSanity()"></div>`;
}

function buildForm(fields) {
  let html = '';
  for (const g of SCHEMA.groups) {
    html += `<div class="group"><h3>${g.title}</h3>`;
    for (const key of g.fields) html += inputRow(key, fields[key] ?? '', MONEY.has(key));
    html += `</div>`;
  }
  // items table
  html += `<div class="group" id="items-wrap"><h3>รายการสินค้า`
        + `<button id="add-item" onclick="addItem()">+ เพิ่มแถว</button></h3>`
        + `<table><colgroup><col class="c-desc"><col class="c-num"><col class="c-num">`
        + `<col class="c-num"><col class="c-num"><col class="c-del"></colgroup><thead><tr>`;
  for (const c of SCHEMA.item_fields) html += `<th>${SCHEMA.labels[c]||c}</th>`;
  html += `<th></th></tr></thead><tbody id="items-body"></tbody></table>`;
  html += `<div id="sanity"></div></div>`;
  document.getElementById('form').innerHTML = html;

  (fields.items || []).forEach(addItemRow);
  checkSanity();
}

function addItemRow(item) {
  const body = document.getElementById('items-body');
  const tr = document.createElement('tr');
  let cells = '';
  for (const c of SCHEMA.item_fields) {
    const num = ['quantity','unit_price','discount','amount'].includes(c) ? 'num' : '';
    cells += `<td><input class="it-${c} ${num}" value="${esc(item[c]??'')}" oninput="checkSanity()"></td>`;
  }
  cells += `<td class="del"><button class="del-btn" onclick="this.closest('tr').remove();checkSanity()">&times;</button></td>`;
  tr.innerHTML = cells;
  body.appendChild(tr);
}

function addItem() {
  const empty = {}; SCHEMA.item_fields.forEach(c => empty[c] = '');
  addItemRow(empty);
}

function collectFields() {
  const f = {};
  for (const g of SCHEMA.groups)
    for (const key of g.fields) f[key] = document.getElementById('f-' + key).value;
  f.items = [];
  document.querySelectorAll('#items-body tr').forEach(tr => {
    const it = {};
    SCHEMA.item_fields.forEach(c => it[c] = tr.querySelector('.it-' + c).value);
    f.items.push(it);
  });
  return f;
}

function parseNum(s) {
  if (s === undefined || s === null) return NaN;
  const c = String(s).replace(/,/g, '').replace(/บาท/g,'').trim();
  return c === '' ? NaN : parseFloat(c);
}

function checkSanity() {
  const f = collectFields();
  const sub = parseNum(f.subtotal), disc = parseNum(f.discount) || 0;
  const vat = parseNum(f.vat), grand = parseNum(f.grand_total);
  const el = document.getElementById('sanity');
  if (!el) return;
  if ([sub, vat, grand].some(isNaN)) { el.className = ''; return; }
  const expect = sub - disc + vat;
  const vatExpect = sub * 0.07;
  const msgs = [];
  if (Math.abs(expect - grand) > 0.05)
    msgs.push(`ยอดไม่ลงตัว: ${fmt(sub)} - ${fmt(disc)} + ${fmt(vat)} = ${fmt(expect)} ≠ ${fmt(grand)}`);
  if (Math.abs(vatExpect - vat) > Math.max(1, sub*0.005))
    msgs.push(`VAT 7% ควร ≈ ${fmt(vatExpect)} (กรอก ${fmt(vat)})`);
  if (msgs.length) { el.className = 'warn'; el.innerHTML = '⚠ ' + msgs.join('<br>⚠ '); }
  else { el.className = 'ok'; el.textContent = '✓ ยอดเงินลงตัว'; }
}

function fmt(n) { return isNaN(n) ? '?' : n.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}); }
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;'); }

async function loadEntry(idx) {
  if (idx < 0 || idx >= totalCount) return;
  const d = await (await fetch('/api/entry/' + idx)).json();
  currentIdx = idx;
  buildForm(d.fields);
  document.getElementById('einfo').textContent = `${d.id}  [${idx+1}/${totalCount}]`;
  document.getElementById('img').src = '/image/' + d.image_path.split('/').pop();
  document.getElementById('img').classList.remove('zoom');
  document.getElementById('b-auto').style.display = d.auto_labeled ? 'inline' : 'none';
  document.getElementById('b-ok').style.display   = d.verified     ? 'inline' : 'none';
}

async function saveEntry(verify) {
  const r = await fetch('/api/save/' + currentIdx, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ fields: collectFields(), verify })
  });
  const d = await r.json();
  if (d.ok) {
    showToast(verify ? 'Verified!' : 'Saved', false);
    await loadCount();
    if (verify) setTimeout(() => navigate(1), 350);
    else loadEntry(currentIdx);
  } else showToast('Error!', true);
}

function navigate(dir) {
  const n = currentIdx + dir;
  if (n >= 0 && n < totalCount) loadEntry(n);
}

async function jumpUnverified() {
  const d = await (await fetch('/api/next_unverified/' + currentIdx)).json();
  if (d.idx === -1) showToast('Verify ครบทุกใบแล้ว!', false);
  else loadEntry(d.idx);
}

function showToast(msg, err) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'show' + (err ? ' err' : '');
  setTimeout(() => t.className = '', 1600);
}

document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 's')     { e.preventDefault(); saveEntry(false); }
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); saveEntry(true); }
  if (e.altKey && e.key === 'ArrowRight') { e.preventDefault(); navigate(1); }
  if (e.altKey && e.key === 'ArrowLeft')  { e.preventDefault(); navigate(-1); }
});

init();
</script>
</body>
</html>
"""


def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")


def get_lan_ip() -> str:
    """หา IP ของเครื่องในวง LAN (ไว้แชร์ให้เพื่อนเปิด)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # ไม่ได้ส่งจริง แค่ให้ OS เลือก interface
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review & edit invoice annotations (form-based)")
    parser.add_argument("--file", default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    ANNOTATION_FILE = Path(args.file)
    ANNOTATIONS     = load_annotations(ANNOTATION_FILE)

    total    = len(ANNOTATIONS)
    verified = sum(1 for a in ANNOTATIONS if a.get("verified"))
    lan_ip   = get_lan_ip()
    print(f"[INFO] โหลด {total} ใบ ({verified} verified, {total-verified} รอตรวจ)")
    print(f"[INFO] ไฟล์: {ANNOTATION_FILE}")
    print(f"[INFO] เครื่องคุณเปิดที่ : http://localhost:{args.port}")
    print(f"[INFO] ส่งลิงก์นี้ให้เพื่อนในวง Wi-Fi เดียวกัน : http://{lan_ip}:{args.port}")
    print( "[INFO] ถ้าเพื่อนเข้าไม่ได้ -> เปิด Windows Firewall ให้ Python (ดูคำแนะนำที่ผมบอกไว้)")
    print( "[INFO] (Ctrl+C เพื่อปิด)\n")

    threading.Timer(1.0, open_browser).start()
    app.run(host="0.0.0.0", port=args.port, debug=False)
