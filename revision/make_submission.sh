#!/usr/bin/env bash
# ประกอบชุดส่งวารสาร ทุกไฟล์สร้างใหม่จากต้นทาง ไม่คัดลอกของเก่า
set -e
cd "$(dirname "$0")"
TEC=/home/kumwilai/.local/bin/tectonic
OUT=submission
rm -rf $OUT && mkdir -p $OUT

echo "== สร้างตารางและค่าคงที่จากไฟล์ผลจริง =="
( cd .. && /home/kumwilai/osmnx-env/bin/python revision/make_tables.py >/dev/null )

echo "== คอมไพล์บทความหลัก =="
( cd paper && $TEC -X compile main.tex --outdir . --keep-intermediates --keep-logs >/dev/null 2>&1 )
echo "== คอมไพล์ภาคผนวก =="
( cd supp && $TEC -X compile main.tex --outdir . --keep-intermediates --keep-logs >/dev/null 2>&1 )
echo "== สร้างจดหมายตอบ =="
/home/kumwilai/osmnx-env/bin/python build_response_docx.py >/dev/null

pages () { python3 -c "
import re,zlib,sys
d=open(sys.argv[1],'rb').read(); n=0
for m in re.finditer(rb'stream\r?\n',d):
    s=m.end(); e=d.find(b'endstream',s)
    try: t=zlib.decompress(d[s:e])
    except Exception: continue
    cc=[int(x) for x in re.findall(rb'/Count\s+(\d+)',t)]
    if cc: n=max(n,max(cc))
c=[int(x) for x in re.findall(rb'/Count\s+(\d+)',d)]
print(max(n, max(c) if c else 0))" "$1"; }

P=$(pages paper/main.pdf)
S=$(pages supp/main.pdf)
cp paper/main.pdf $OUT/OJCS-2026-04-0386_revised_manuscript.pdf
cp supp/main.pdf  $OUT/OJCS-2026-04-0386_supplementary.pdf
cp Response_to_Reviewers.docx $OUT/OJCS-2026-04-0386_response_to_reviewers.docx
cp CONCERN_TRACKER.md $OUT/concern_tracker.md

echo "== ตรวจก่อนส่ง =="
FAIL=0
[ "$P" -le 12 ] || { echo "ผิด บทความหลัก $P หน้า เกิน 12"; FAIL=1; }
grep -qE "(Reference|Citation).*undefined" paper/main.log && { echo "ผิด มีการอ้างอิงค้างในบทความหลัก"; FAIL=1; } || true
grep -qE "(Reference|Citation).*undefined" supp/main.log && { echo "ผิด มีการอ้างอิงค้างในภาคผนวก"; FAIL=1; } || true
n=$(grep -cE "Overfull \\\\hbox" paper/main.log || true); [ "$n" -eq 0 ] || { echo "เตือน มี Overfull hbox $n จุด"; }
grep -rq "TBDRESULTS\|PLACEHOLDER" sections/*.tex supp/sections/*.tex && { echo "ผิด มีช่องว่างค้าง"; FAIL=1; } || true
grep -q "\[FILL\|\[CHECK\|\[DECISION\|\[TBD" responses_data.json && { echo "ผิด จดหมายตอบมีวงเล็บค้าง"; FAIL=1; } || true
/home/kumwilai/osmnx-env/bin/python style_check.py sections/*.tex supp/sections/*.tex | grep -v "^clean" && { echo "ผิด สไตล์ไม่ผ่าน"; FAIL=1; } || true

echo
echo "บทความหลัก $P หน้า   ภาคผนวก $S หน้า"
ls -1 $OUT
[ "$FAIL" -eq 0 ] && echo "ชุดส่งพร้อม" || { echo "ยังไม่พร้อม"; exit 1; }
