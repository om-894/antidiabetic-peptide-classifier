"""
Build the third draft, clean and tracked, from the second draft.

Both come from the one edit table in revise_ms_3rd.py, so they cannot drift
apart. The tracked file shows second draft to third draft, which is what the
supervisor asked to review.

Usage:  python build_3rd.py <second_draft.docx> <out_dir>

INPUTS   the second-draft .docx
OUTPUTS  ADP_manuscript_JCIM_3rd_draft.docx
         ADP_manuscript_JCIM_3rd_draft_tracked.docx
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import revise_ms_3rd as rev

SKILL = ("/Users/olivermcquillan/Library/Application Support/Claude/"
         "local-agent-mode-sessions/skills-plugin/"
         "8c3daa7b-0fba-4809-855f-18df51b361a2/"
         "2290ba74-5977-4b52-8156-7c3da6ac754b/skills/docx")


def unpack(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)
    # merge adjacent identically formatted runs, so paragraph text is findable
    subprocess.run([sys.executable, f"{SKILL}/scripts/merge_runs.py", dst],
                   check=True, capture_output=True)


def repack(folder, out):
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, folder))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Downloads/ADP_manuscript_JCIM_2nd_draft.docx")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    work = os.path.join(out_dir, "_build")
    os.makedirs(work, exist_ok=True)

    for tracked in (False, True):
        tag = "tracked" if tracked else "clean"
        folder = os.path.join(work, tag)
        unpack(src, folder)
        doc = os.path.join(folder, "word", "document.xml")
        x = open(doc, encoding="utf-8").read()
        x = rev.apply(x, tracked=tracked)
        open(doc, "w", encoding="utf-8").write(x)

        name = ("ADP_manuscript_JCIM_3rd_draft_tracked.docx" if tracked
                else "ADP_manuscript_JCIM_3rd_draft.docx")
        repack(folder, os.path.join(out_dir, name))
        n_ins = len(re.findall(r'<w:ins[ >]', x))
        n_del = len(re.findall(r'<w:del[ >]', x))
        print(f"{name}: {len(rev.REPLACE)} paragraphs revised, "
              f"{n_ins} insertions, {n_del} deletions")


if __name__ == "__main__":
    main()
