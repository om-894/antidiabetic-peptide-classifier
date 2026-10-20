"""
Render paragraph edits as Word tracked changes.

The third draft has to ship as a clean file and as a tracked-changes file that
agree exactly. Deriving one from the other is not possible here: the docx
skill's accept_changes.py needs LibreOffice, which is not installed. So both
are generated from one edit table, this module supplying the tracked renderer
and revise_ms_3rd.py the clean one.

Diffing is at word level rather than paragraph level. Replacing a whole
paragraph would mark every touched paragraph as wholly rewritten, which tells a
reviewer nothing about what actually changed. SequenceMatcher over word tokens
shows only the words that moved.

Formatting survives the diff because each token carries the run properties it
came from, so a superscript citation stays superscript whether it is kept,
deleted or inserted.

INPUTS   paragraph XML from the second draft, plus the replacement text
OUTPUTS  paragraph XML carrying <w:ins> and <w:del> revision marks
"""

import re
from difflib import SequenceMatcher

RUN = re.compile(r'<w:r(?: [^>]*)?>(.*?)</w:r>', re.S)
RPR = re.compile(r'<w:rPr>.*?</w:rPr>', re.S)
TEXT = re.compile(r'<w:t(?: [^>]*)?>(.*?)</w:t>', re.S)
MARKUP = re.compile(r'(\*\*.+?\*\*|\^\{[^}]+\})', re.S)
WORD = re.compile(r'\S+\s*|\s+')

BOLD = "<w:rPr><w:b/><w:bCs/></w:rPr>"
SUPER = '<w:rPr><w:vertAlign w:val="superscript"/></w:rPr>'


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unesc(t):
    return t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def old_tokens(para):
    """Word tokens of an existing paragraph, each with the rPr of its run."""
    out = []
    for body in RUN.findall(para):
        m = RPR.search(body)
        rpr = m.group(0) if m else ""
        for chunk in TEXT.findall(body):
            for tok in WORD.findall(unesc(chunk)):
                out.append((tok, rpr))
    return out


def new_tokens(text):
    """Word tokens of replacement text, reading **bold** and ^{n} superscripts."""
    out = []
    for part in MARKUP.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            rpr, body = BOLD, part[2:-2]
        elif part.startswith("^{"):
            rpr, body = SUPER, part[2:-1]
        else:
            rpr, body = "", part
        for tok in WORD.findall(body):
            out.append((tok, rpr))
    return out


class Ids:
    """Revision ids have to be unique across the document."""

    def __init__(self, start=1000):
        self.n = start

    def next(self):
        self.n += 1
        return self.n


def _runs(tokens, deleted=False):
    """Collapse consecutive tokens sharing rPr into as few runs as possible."""
    out, buf, cur = [], [], None
    for tok, rpr in tokens:
        if rpr != cur and buf:
            out.append((cur, "".join(buf)))
            buf = []
        cur = rpr
        buf.append(tok)
    if buf:
        out.append((cur, "".join(buf)))

    # inside a deletion the text element is w:delText, not w:t
    tag = "w:delText" if deleted else "w:t"
    return "".join(f'<w:r>{rpr}<{tag} xml:space="preserve">{esc(t)}</{tag}></w:r>'
                   for rpr, t in out)


def tracked_paragraph(para, new_text, author, date, ids):
    """Rebuild a paragraph so its difference from new_text shows as revisions."""
    old = old_tokens(para)
    new = new_tokens(new_text)
    if not old:
        return None # nothing to diff against, caller should insert instead

    sm = SequenceMatcher(None, [t for t, _ in old], [t for t, _ in new],
                         autojunk=False)
    parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(_runs(old[i1:i2]))
            continue
        # a replace emits the deletion first so Word shows them in reading order
        if op in ("delete", "replace"):
            parts.append(f'<w:del w:id="{ids.next()}" w:author="{esc(author)}" '
                         f'w:date="{date}">{_runs(old[i1:i2], deleted=True)}'
                         f'</w:del>')
        if op in ("insert", "replace"):
            parts.append(f'<w:ins w:id="{ids.next()}" w:author="{esc(author)}" '
                         f'w:date="{date}">{_runs(new[j1:j2])}</w:ins>')

    m = re.search(r'<w:pPr>.*?</w:pPr>', para, re.S)
    ppr = m.group(0) if m else ""
    open_tag = re.match(r'<w:p(?: [^>]*)?>', para).group(0)
    return f"{open_tag}{ppr}{''.join(parts)}</w:p>"


def inserted_paragraph(new_text, ppr, author, date, ids):
    """A wholly new paragraph, with its paragraph mark marked inserted too."""
    # without the mark on the pPr, accepting the change leaves the text but
    # merges it into the paragraph above
    mark = (f'<w:rPr><w:ins w:id="{ids.next()}" w:author="{esc(author)}" '
            f'w:date="{date}"/></w:rPr>')
    inner = re.sub(r'<w:rPr>.*?</w:rPr>', "", ppr, flags=re.S) if ppr else ""
    inner = inner.replace("</w:pPr>", f"{mark}</w:pPr>") if inner else \
        f"<w:pPr>{mark}</w:pPr>"
    body = (f'<w:ins w:id="{ids.next()}" w:author="{esc(author)}" '
            f'w:date="{date}">{_runs(new_tokens(new_text))}</w:ins>')
    return f"<w:p>{inner}{body}</w:p>"


def deleted_paragraph(para, author, date, ids):
    """Mark an existing paragraph wholly deleted, mark included."""
    mark = (f'<w:rPr><w:del w:id="{ids.next()}" w:author="{esc(author)}" '
            f'w:date="{date}"/></w:rPr>')
    m = re.search(r'<w:pPr>.*?</w:pPr>', para, re.S)
    ppr = m.group(0) if m else "<w:pPr></w:pPr>"
    ppr = re.sub(r'<w:rPr>.*?</w:rPr>', "", ppr, flags=re.S)
    ppr = ppr.replace("</w:pPr>", f"{mark}</w:pPr>")
    body = (f'<w:del w:id="{ids.next()}" w:author="{esc(author)}" '
            f'w:date="{date}">{_runs(old_tokens(para), deleted=True)}</w:del>')
    open_tag = re.match(r'<w:p(?: [^>]*)?>', para).group(0)
    return f"{open_tag}{ppr}{body}</w:p>"
