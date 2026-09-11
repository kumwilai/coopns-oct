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
# A hard TeX error used to be swallowed here, leaving the previous PDF in place
# and every later check reading a stale file. Fail loudly instead.
( cd paper && $TEC -X compile main.tex --outdir . --keep-intermediates --keep-logs >/dev/null 2>&1 ) \
  || { echo "ผิด บทความหลักคอมไพล์ไม่ผ่าน"; ( cd paper && $TEC -X compile main.tex --outdir . 2>&1 | grep -E "^error" | head -5 ); exit 1; }
# The supplement prints the main paper's equation numbers, so they are read from
# the aux that the compile above just produced rather than typed in by hand.
/home/kumwilai/osmnx-env/bin/python gen_eqnums.py
echo "== คอมไพล์ภาคผนวก =="
( cd supp && $TEC -X compile main.tex --outdir . --keep-intermediates --keep-logs >/dev/null 2>&1 ) \
  || { echo "ผิด ภาคผนวกคอมไพล์ไม่ผ่าน"; ( cd supp && $TEC -X compile main.tex --outdir . 2>&1 | grep -E "^error" | head -5 ); exit 1; }
echo "== สร้างจดหมายตอบ =="
/home/kumwilai/osmnx-env/bin/python build_response_docx.py >/dev/null

pages () { /home/kumwilai/osmnx-env/bin/python -c "
import pymupdf,sys
print(pymupdf.open(sys.argv[1]).page_count)" "$1"; }

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
n=$(grep -cE "Overfull \\\\hbox" paper/main.log || true); [ "$n" -eq 0 ] || { echo "ผิด มี Overfull hbox $n จุด"; FAIL=1; }
m=$(grep -cE "Overfull \\\\hbox" supp/main.log || true); [ "$m" -eq 0 ] || echo "เตือน ภาคผนวกมี Overfull hbox $m จุด"
grep -rq "TBDRESULTS\|PLACEHOLDER" sections/*.tex supp/sections/*.tex && { echo "ผิด มีช่องว่างค้าง"; FAIL=1; } || true
grep -q "\[FILL\|\[CHECK\|\[DECISION\|\[TBD" responses_data.json && { echo "ผิด จดหมายตอบมีวงเล็บค้าง"; FAIL=1; } || true
AB=$(/home/kumwilai/osmnx-env/bin/python -c "
import pymupdf
t=pymupdf.open('paper/main.pdf')[0].get_text()
i=t.find('ABSTRACT')
i=t.find(' ', i) if i >= 0 else 0
j=t.find('INDEX TERMS')
if j<0: j=t.find('I. INTRODUCTION')
import re; print(len(re.split(r'[\\s-]+', ' '.join(t[i:j].split()).strip())))")
[ "$AB" -le 250 ] || { echo "ผิด บทคัดย่อ $AB คำ เกิน 250 ซึ่งเป็นเพดานของวารสาร"; FAIL=1; }
[ "$AB" -ge 100 ] || { echo "ผิด บทคัดย่อ $AB คำ ต่ำกว่า 100 ซึ่งเป็นขั้นต่ำของวารสาร"; FAIL=1; }
/home/kumwilai/osmnx-env/bin/python style_check.py sections/*.tex supp/sections/*.tex | grep -v "^clean" && { echo "ผิด สไตล์ไม่ผ่าน"; FAIL=1; } || true
/home/kumwilai/osmnx-env/bin/python figure_check.py | grep -v "^clean" && { echo "ผิด มีตัวอักษรทับบล็อกในรูป"; FAIL=1; } || true

echo
echo "บทความหลัก $P หน้า   ภาคผนวก $S หน้า"
ls -1 $OUT
[ "$FAIL" -eq 0 ] && echo "ชุดส่งพร้อม" || { echo "ยังไม่พร้อม"; exit 1; }
