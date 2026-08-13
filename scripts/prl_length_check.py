#!/usr/bin/env python3
"""PRL length estimate for the UHDM Letter, counted the official APS way.

Usage:
    python3 scripts/prl_length_check.py [path/to/main.tex]

Default target: ignore/overleaf/main.tex (relative to the repo root, i.e. the
parent of the directory holding this script). Exit status is 1 if the Letter
is over the limit, 0 otherwise.

The script:
  1. splits the document into Letter and Supplemental Material at \\appendix,
  2. strips draft markup so the count reflects the SUBMITTED paper,
  3. counts text words, display-equation rows, and figures per the official
     APS formulas (below), measuring true figure aspect ratios from the PDFs,
  4. prints a per-category report and the margin against the 3750-word limit.

================================ OFFICIAL RULES ================================
Source: APS, "Length Limits and Guidelines for Physical Review Article Types",
        https://journals.aps.org/authors/length-guide
        (live page sits behind a Cloudflare bot check; text below taken
        verbatim from the Wayback Machine capture of 2025-08-22,
        https://web.archive.org/web/20250822005554/https://journals.aps.org/authors/length-guide
        accessed 2026-08-11)

  * Limit: "PRL / Letter / 3,750 words".
  * "The general formula for calculating a manuscript's length is:
     Total Word Count = Text + Displayed Math + Figures + Tables"
  * Text -- Include: "Any text in the body of the article; Any text in a
    figure caption or table caption; Any text in a footnote or an endnote."
  * Text -- Exclude: "Title; Author and affiliation listing; Abstract;
    Receipt date, published date, and other publication history; PhySH
    Keywords and DOI; References; Author byline footnotes; Acknowledgments."
  * Displayed Math: "The word equivalent for displayed math is 16 words per
    row for single-column equations. Two-column equations count as 32 words
    per row."
  * Figures: "To estimate the word equivalent for figures use the figure's
    aspect ratio (width / height). The estimate is [(150 / aspect ratio) +
    20 words] for single-column figures, and [300 / (0.5 * aspect ratio)] +
    40 words for double-column figures."
  * Tables: "The word equivalent for tables is 13 words plus 6.5 words per
    line for single-column tables. Double-column tables count as 26 words
    plus 13 words per line."
  * Figure sizes: "GhostScript can be used to determine the bounding box ...
    The units are arbitrary because the guidelines use the aspect ratio,
    which is the width/height."  (We use pdfinfo / the PDF MediaBox.)
  * Tool: APS does NOT endorse texcount. Its official TeX procedure is to
    comment out \\maketitle, use 'nofootinbib', put \\end{document} before the
    bibliography, comment out display equations / table rows / the
    acknowledgment, then count with wordcount.tex from CTAN
    (https://ctan.org/tex-archive/macros/latex/contrib/wordcount).

End Matter (not used in this draft):
Source: PRL Information for Authors, https://journals.aps.org/prl/authors
        (Wayback capture 2025-08-12, accessed 2026-08-11):
  * "Authors can also add up to two pages of appendices or other content --
    called End Matter -- that specialists will want or need to read. End
    Matter does not count against the core length limit."
  * Supplemental Material is a separate deposit and never counts.
================================================================================

Conventions where the official guide is silent (documented so the estimate is
reproducible):
  * Each inline math expression ($...$, \\(...\\), \\ensuremath{...}) counts as
    ONE word (same convention as the community tool texprlcount).
  * Each \\cite{...} (any number of keys) and each \\ref/\\cref/\\eqref counts
    as ONE word (they typeset as "[n]" / "Fig. n").
  * Per-figure word equivalents are rounded UP to the next integer.
  * Rows of display math = 1 + number of \\\\ separators in the environment
    (a trailing \\\\ before \\end is not counted as an extra row).

Draft markup stripped before counting (macros defined in main.tex ~99-119):
  * Deleted with their argument (margin comments / struck text, gone at
    submission):  \\DA \\da \\QJH \\qjh \\THO \\DU \\CTtwo \\CTn \\CTQ \\JWL \\sout
  * Unwrapped, argument kept (colored wrappers around real manuscript text):
    \\CT \\CTB \\rev
  * \\CTnum{old}{new}: the second argument is kept.
  * LaTeX comments (%) removed, \\% respected.
  * The acknowledgment paragraph(s) are excluded per the official rules
    (\\begin{acknowledgments} if present, else the paragraph starting with a
    thanks/funding formula, up to the next section command).

Dependencies: Python 3 stdlib. Optional: pdfinfo (poppler) for figure sizes
(falls back to reading the PDF MediaBox directly), texcount (only used as an
automatic cross-check of the internal word counter).
"""

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

LIMIT = 3750
EQ_WORDS_SINGLE = 16
EQ_WORDS_DOUBLE = 32

DELETE_MACROS = ["DA", "da", "QJH", "qjh", "THO", "DU",
                 "CTtwo", "CTn", "CTQ", "JWL", "sout"]
UNWRAP_MACROS = ["CT", "CTB", "rev"]

MATH_ENVS = ["equation", "align", "alignat", "flalign", "gather",
             "multline", "eqnarray", "displaymath"]


# ----------------------------------------------------------------- utilities
def die(msg):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(2)


def strip_comments(text):
    """Remove LaTeX %-comments, respecting escaped \\%."""
    out_lines = []
    for line in text.split("\n"):
        cut = None
        i = 0
        while i < len(line):
            if line[i] == "\\":
                i += 2  # skip escaped char (covers \% and \\)
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        out_lines.append(line if cut is None else line[:cut])
    return "\n".join(out_lines)


def match_group(text, i):
    """text[i] == '{'; return (content, index_after_closing_brace)."""
    assert text[i] == "{"
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    raise ValueError("Unbalanced braces at offset %d: %r" % (i, text[i:i + 60]))


def strip_draft_macros(text):
    """Apply the draft-markup rules. Iterates until stable (handles nesting)."""
    pat = re.compile(r"\\(CTnum|" + "|".join(DELETE_MACROS + UNWRAP_MACROS) +
                     r")(?![a-zA-Z])")
    while True:
        m = pat.search(text)
        if m is None:
            return text
        name = m.group(1)
        i = m.end()
        while i < len(text) and text[i] in " \t\n":
            i += 1
        if i >= len(text) or text[i] != "{":
            # bare use without argument: just drop the macro token
            text = text[:m.start()] + text[m.end():]
            continue
        arg1, after = match_group(text, i)
        if name == "CTnum":
            j = after
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if j < len(text) and text[j] == "{":
                arg2, after = match_group(text, j)
                repl = arg2  # keep the replacement number
            else:
                repl = arg1
        elif name in DELETE_MACROS:
            repl = " "
        else:  # unwrap
            repl = arg1
        text = text[:m.start()] + repl + text[after:]


def count_words(plain):
    """Count whitespace-separated tokens containing at least one alphanumeric."""
    return sum(1 for tok in plain.split() if re.search(r"[A-Za-z0-9]", tok))


def pdf_size(path):
    """Return (width, height) in points, via pdfinfo or the PDF MediaBox."""
    if shutil.which("pdfinfo"):
        try:
            out = subprocess.run(["pdfinfo", path], capture_output=True,
                                 text=True, timeout=30).stdout
            m = re.search(r"Page size:\s*([\d.]+)\s*x\s*([\d.]+)\s*pts", out)
            if m:
                return float(m.group(1)), float(m.group(2))
        except Exception:
            pass
    with open(path, "rb") as fh:
        data = fh.read()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+"
                  rb"([\d.+-]+)\s+([\d.+-]+)", data)
    if m:
        x0, y0, x1, y1 = (float(m.group(k)) for k in range(1, 5))
        return abs(x1 - x0), abs(y1 - y0)
    raise RuntimeError("cannot determine page size of %s" % path)


# ----------------------------------------------------- text -> plain words
def latex_to_plain(text):
    """Reduce cleaned LaTeX (no display math / floats left) to countable text."""
    t = text
    # labels and graphics/layout commands whose arguments are not prose
    for cmd in ["label", "bibliography", "bibliographystyle", "hypersetup",
                "vspace", "hspace", "includegraphics", "setcounter",
                "counterwithout", "counterwithin", "pagestyle", "captionsetup",
                "renewcommand", "phantomsection", "setlength", "addtocontents",
                "titlecontents", "startcontents", "printcontents"]:
        t = re.sub(r"\\" + cmd + r"\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}", " ", t)
    # citations and cross-references -> one word each
    t = re.sub(r"\\(?:cite|onlinecite|citep|citet)\s*(?:\[[^\]]*\])?\s*"
               r"\{[^{}]*\}", " CITEREF ", t)
    t = re.sub(r"\\(?:cref|Cref|ref|eqref|autoref|pageref)\s*\{[^{}]*\}",
               " XREF ", t)
    # \texorpdfstring{tex}{pdf} -> tex ; \href{url}{text} -> text
    while True:
        m = re.search(r"\\(texorpdfstring|href|MYhref)(?![a-zA-Z])", t)
        if not m:
            break
        i = m.end()
        if t[i] == "[":  # optional arg (MYhref color)
            i = t.index("]", i) + 1
        a1, after = match_group(t, i)
        j = after
        while j < len(t) and t[j] in " \t\n":
            j += 1
        a2, after2 = (None, after)
        if j < len(t) and t[j] == "{":
            a2, after2 = match_group(t, j)
        keep = a1 if m.group(1) == "texorpdfstring" else (a2 or " URL ")
        t = t[:m.start()] + " " + keep + " " + t[after2:]
    t = re.sub(r"\\url\s*\{[^{}]*\}", " URL ", t)
    return t


def replace_inline_math(text):
    """$...$, \\(...\\), \\ensuremath{...}  -> single MATHX token."""
    # \ensuremath{...}
    while True:
        m = re.search(r"\\ensuremath(?![a-zA-Z])\s*\{", text)
        if not m:
            break
        _, after = match_group(text, m.end() - 1)
        text = text[:m.start()] + " MATHX " + text[after:]
    # \( ... \)
    text = re.sub(r"\\\(.*?\\\)", " MATHX ", text, flags=re.S)
    # $ ... $ (unescaped)
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        if c == "$":
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "$":
                    break
                j += 1
            out.append(" MATHX ")
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def plain_words(text):
    """Full pipeline: cleaned LaTeX fragment -> word count."""
    t = latex_to_plain(text)
    t = replace_inline_math(t)
    # \footnote{...} content counts: unwrap
    while True:
        m = re.search(r"\\footnote(?![a-zA-Z])\s*\{", t)
        if not m:
            break
        arg, after = match_group(t, m.end() - 1)
        t = t[:m.start()] + " " + arg + " " + t[after:]
    # leftover environments and commands: drop the command token, keep braced text
    t = re.sub(r"\\begin\s*\{[^{}]*\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\end\s*\{[^{}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?", " ", t)  # any other \cmd[opt]
    t = re.sub(r"\\\\|\\[^a-zA-Z]", " ", t)                # \\, \%, \_, ...
    t = t.replace("~", " ").replace("{", " ").replace("}", " ")
    return count_words(t)


# ------------------------------------------------- structure extraction
def extract_math(text):
    """Remove display math; return (text_without_math, [(env, rows, wide, line)])."""
    # widetext spans mark two-column equations
    wide_spans = [(m.start(), m.end()) for m in
                  re.finditer(r"\\begin\{widetext\}.*?\\end\{widetext\}",
                              text, re.S)]

    def is_wide(pos):
        return any(a <= pos < b for a, b in wide_spans)

    eqs = []
    env_re = re.compile(r"\\begin\{(" + "|".join(MATH_ENVS) +
                        r")(\*?)\}(.*?)\\end\{\1\2\}", re.S)

    def env_sub(m):
        body = m.group(3)
        seps = len(re.findall(r"\\\\", body))
        if re.search(r"\\\\\s*$", body.rstrip()):
            seps -= 1
        rows = 1 + max(seps, 0)
        snip = re.sub(r"\s+", " ", body).strip()[:34]
        eqs.append((m.group(1) + m.group(2), rows, is_wide(m.start()), snip))
        return " "

    out = env_re.sub(env_sub, text)

    def disp_sub(m):
        snip = re.sub(r"\s+", " ", m.group(1)).strip()[:34]
        eqs.append(("\\[..\\]", 1, is_wide(m.start()), snip))
        return " "

    out = re.sub(r"\\\[(.*?)\\\]", disp_sub, out, flags=re.S)
    out = re.sub(r"\\begin\{widetext\}|\\end\{widetext\}", " ", out)
    return out, eqs


def extract_floats(text, env):
    """Remove figure/table envs (keeping captions as text).

    Returns (new_text, [ {star, caption, graphics, body, line} ])."""
    found = []
    pat = re.compile(r"\\begin\{" + env + r"(\*?)\}(?:\[[^\]]*\])?(.*?)"
                     r"\\end\{" + env + r"\1\}", re.S)

    def sub(m):
        body = m.group(2)
        caption = ""
        cm = re.search(r"\\caption(?![a-zA-Z])\s*(?:\[[^\]]*\])?\s*\{", body)
        if cm:
            caption, _ = match_group(body, cm.end() - 1)
        graphics = re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*"
                              r"\{([^{}]+)\}", body)
        found.append({"star": m.group(1) == "*", "caption": caption,
                      "graphics": graphics, "body": body,
                      "line": text[:m.start()].count("\n") + 1})
        return " " + caption + " "   # caption text stays in the body count

    return pat.sub(sub, text), found


def remove_acknowledgments(text):
    """Drop the acknowledgment block; return (text, words_removed, how)."""
    m = re.search(r"\\begin\{acknowledgments\}(.*?)\\end\{acknowledgments\}",
                  text, re.S)
    if m:
        return (text[:m.start()] + " " + text[m.end():],
                plain_words(strip_draft_macros(m.group(1))), "acknowledgments env")
    m = re.search(r"(?:We (?:would like to )?thank|We acknowledge|"
                  r"We are grateful|The authors? (?:thank|acknowledge)|"
                  r"This work was supported)", text)
    if m:
        end = re.search(r"\\(?:mysection|section|bibliography)(?![a-zA-Z])",
                        text[m.start():])
        stop = m.start() + (end.start() if end else len(text) - m.start())
        block = text[m.start():stop]
        return (text[:m.start()] + " " + text[stop:],
                plain_words(strip_draft_macros(block)),
                "heuristic (thanks/funding paragraph up to next section)")
    return text, 0, None


# ------------------------------------------------------------ formulas
def figure_words(width, height, star):
    ar = width / height
    if star:
        raw = 300.0 / (0.5 * ar) + 40.0
    else:
        raw = 150.0 / ar + 20.0
    return ar, raw, math.ceil(raw)


def table_words(n_lines, star):
    return (26 + 13.0 * n_lines) if star else (13 + 6.5 * n_lines)


# ------------------------------------------------------------ texcount
def texcount_check(fragment):
    """Cross-check word count of a cleaned fragment with texcount, if present."""
    if not shutil.which("texcount"):
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(fragment)
        name = fh.name
    try:
        out = subprocess.run(["texcount", "-brief", "-total", "-sum",
                              "-utf8", name],
                             capture_output=True, text=True, timeout=60).stdout
        m = re.search(r"(\d+)", out)
        return int(m.group(1)) if m else None
    finally:
        os.unlink(name)


# ------------------------------------------------------------ main
def analyze_part(text, tex_dir):
    """Run the full pipeline on one document part. Returns a result dict."""
    res = {}
    t = strip_draft_macros(text)
    t, eqs = extract_math(t)
    t, figs = extract_floats(t, "figure")
    t, tabs = extract_floats(t, "table")

    res["eqs"] = eqs
    res["eq_words"] = sum(r * (EQ_WORDS_DOUBLE if w else EQ_WORDS_SINGLE)
                          for _, r, w, _ in eqs)

    fig_rows = []
    fig_total = 0
    for f in figs:
        for g in f["graphics"]:
            path = os.path.join(tex_dir, g)
            if not os.path.exists(path) and os.path.exists(path + ".pdf"):
                path += ".pdf"
            try:
                w, h = pdf_size(path)
                ar, raw, words = figure_words(w, h, f["star"])
                fig_rows.append((g, f["star"], w, h, ar, raw, words))
                fig_total += words
            except Exception as e:
                fig_rows.append((g, f["star"], None, None, None, None, None))
                res.setdefault("errors", []).append(
                    "figure %s: %s" % (g, e))
    res["figs"] = fig_rows
    res["fig_words"] = fig_total

    tab_rows = []
    tab_total = 0.0
    for tb in tabs:
        n_lines = len(re.findall(r"\\\\", tb["body"])) or 1
        wds = table_words(n_lines, tb["star"])
        tab_rows.append((tb["star"], n_lines, wds, tb["line"]))
        tab_total += wds
    res["tables"] = tab_rows
    res["table_words"] = math.ceil(tab_total)

    res["cleaned_text"] = t
    res["text_words"] = plain_words(t)
    res["total"] = (res["text_words"] + res["eq_words"] +
                    res["fig_words"] + res["table_words"])
    return res


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(os.path.dirname(here), "ignore", "overleaf",
                           "main.tex")
    tex_path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(tex_path):
        die("no such file: %s" % tex_path)
    tex_dir = os.path.dirname(os.path.abspath(tex_path))

    with open(tex_path, encoding="utf-8") as fh:
        raw = fh.read()
    text = strip_comments(raw)

    m = re.search(r"\\begin\{document\}", text)
    e = re.search(r"\\end\{document\}", text)
    doc = text[m.end():e.start() if e else len(text)] if m else text

    # ---- Letter / Supplemental Material boundary
    # End Matter (PRL, after the references) is outside the 3750-word count,
    # so it ends the Letter just as \appendix does -- whichever comes first.
    bpos, how = len(doc), "none found -- treating whole document as Letter"
    m_app = re.search(r"\\appendix(?![a-zA-Z])", doc)
    m_em = re.search(r"\\mysection\{End Matter", doc)
    if m_em and (not m_app or m_em.start() < m_app.start()):
        bpos, how = m_em.start(), "End Matter heading"
    elif m_app:
        bpos, how = m_app.start(), r"\appendix"
    else:
        bib = re.search(r"\\bibliography(?![a-zA-Z])", doc)
        start = bib.end() if bib else 0
        m_ocg = re.search(r"\\onecolumngrid(?![a-zA-Z])", doc[start:])
        m_sm = re.search(r"Supplemental Material", doc)
        if m_ocg:
            bpos, how = start + m_ocg.start(), r"\onecolumngrid"
        elif m_sm:
            bpos, how = m_sm.start(), "'Supplemental Material' heading"

    letter, sm = doc[:bpos], doc[bpos:]

    # ---- Letter: excluded front matter (title/authors/abstract) ends at \maketitle
    mk = re.search(r"\\maketitle(?![a-zA-Z])", letter)
    body = letter[mk.end():] if mk else letter
    # references excluded: cut at \bibliography
    bib = re.search(r"\\bibliography(?![a-zA-Z])", body)
    if bib:
        body = body[:bib.start()]
    body, ack_words, ack_how = remove_acknowledgments(body)

    L = analyze_part(body, tex_dir)
    S = analyze_part(sm, tex_dir)

    # AI-usage statement: currently in the Letter body -> counted. Report size.
    ai = re.search(r"\\mysection\{AI-Usage Statement\}(.*)$",
                   strip_comments(body), re.S)
    ai_words = plain_words(strip_draft_macros(ai.group(1))) if ai else 0

    tc = texcount_check(L["cleaned_text"])

    # ------------------------------------------------------------- report
    line = "-" * 72
    print(line)
    print("PRL LENGTH ESTIMATE (official APS counting) -- %s" %
          os.path.relpath(tex_path))
    print(line)
    print("Letter/SM boundary detected at: %s (offset %d)" % (how, bpos))
    if ack_how:
        print("Acknowledgments excluded (%d words) via: %s" % (ack_words, ack_how))
    else:
        print("WARNING: no acknowledgment block found -- nothing excluded.")
    print()
    print("LETTER (counts toward the %d-word limit)" % LIMIT)
    print("  Text (body + captions + section heads, draft markup stripped):")
    print("      %5d words   [internal counter]" % L["text_words"])
    if tc is not None:
        print("      %5d words   [texcount cross-check on the same cleaned "
              "text, delta %+d]" % (tc, tc - L["text_words"]))
    print("  Display math (16 w/row single-col, 32 w/row two-col):")
    for env, rows, wide, snip in L["eqs"]:
        print("      %-10s %d row(s)%s -> %3d words   [%s...]" %
              (env, rows, " [two-col]" if wide else "",
               rows * (EQ_WORDS_DOUBLE if wide else EQ_WORDS_SINGLE), snip))
    print("      subtotal: %d words" % L["eq_words"])
    print("  Figures (single: 150/AR + 20;  double [figure*]: 300/(0.5*AR) + 40):")
    for g, star, w, h, ar, raw, words in L["figs"]:
        if words is None:
            print("      %-28s  SIZE UNKNOWN -- fix path" % g)
        else:
            print("      %-28s %s  %3.0fx%3.0f pt  AR=%.3f  %7.1f -> %3d words"
                  % (g, "double" if star else "single", w, h, ar, raw, words))
    print("      subtotal: %d words" % L["fig_words"])
    if L["tables"]:
        print("  Tables (single: 13 + 6.5/line; double: 26 + 13/line):")
        for star, n, wds, ln in L["tables"]:
            print("      line %4d  %s  %d line(s) -> %.1f words" %
                  (ln, "double" if star else "single", n, wds))
        print("      subtotal: %d words" % L["table_words"])
    else:
        print("  Tables: none")
    print()
    print("  LETTER TOTAL: %d word-equivalents"
          % L["total"])
    print("  OFFICIAL LIMIT: %d" % LIMIT)
    margin = LIMIT - L["total"]
    if margin >= 0:
        print("  MARGIN: %d words UNDER the limit" % margin)
    else:
        print("  MARGIN: %d words OVER the limit" % -margin)
    print()
    print("SUPPLEMENTAL MATERIAL (informational only -- does not count)")
    print("  Text %d words; %d display equations (%d eq-words if counted); "
          "%d figure graphic(s) (%d fig-words if counted)" %
          (S["text_words"], len(S["eqs"]), S["eq_words"],
           len(S["figs"]), S["fig_words"]))
    print()
    print("NOTES / CAVEATS")
    if ai_words:
        print("  * The AI-Usage Statement (%d words) is counted in the Letter "
              "text; if it moves to acknowledgments/End Matter, subtract it."
              % ai_words)
    print("  * Inline math, \\cite and \\cref each count as 1 word "
          "(guide is silent; texprlcount convention).")
    print("  * APS's official tool is wordcount.tex on the compiled TeX, and "
          "final say is the typeset page count (4 pages); this is an estimate.")
    print("  * Rows inside matrices/cases would be overcounted (none present).")
    for err in L.get("errors", []):
        print("  * ERROR: %s" % err)
    print(line)
    sys.exit(0 if margin >= 0 and not L.get("errors") else 1)


if __name__ == "__main__":
    main()
