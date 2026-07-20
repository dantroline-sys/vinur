"""PDF — text layer via **PyMuPDF** (`fitz`), fast and layout-aware, with an
**OCR fallback** for scanned pages.

Sectioning prefers the document outline (TOC) when present; otherwise pages are
grouped under their nearest preceding TOC entry, or emitted page-by-page.  OCR
is the *exception*, not the rule: only pages whose text layer yields almost
nothing are rendered and run through ``ocrmypdf``/``tesseract`` (per-page, via
subprocess on a temp single-page PDF — no network, parser kept constrained).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile

from . import MissingDependency

log = logging.getLogger("knowledgehost.sources.pdf")

_errors_muted = False

_PAGENO = re.compile(r"^\s*(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.I)


def _strip_furniture(pages: list[str]) -> list[str]:
    """Remove running headers/footers and bare page numbers.

    A furniture line is one that (digits normalised away) recurs at the top or
    bottom of ≥30% of pages — the running title/author/journal banner — or is a
    bare page number.  Only the EDGES of each page are touched (first/last two
    non-blank lines), so a body line that happens to match a banner is safe.
    Needs ≥6 pages to establish a pattern; fewer pass through untouched."""
    if len(pages) < 6:
        return pages
    norm = lambda ln: re.sub(r"\d+", "#", ln.strip()).lower()
    seen: dict[str, int] = {}
    edges = []
    for p in pages:
        idx = [i for i, ln in enumerate(p.splitlines()) if ln.strip()]
        edge = set(idx[:2] + idx[-2:])
        edges.append(edge)
        lines = p.splitlines()
        for i in edge:
            k = norm(lines[i])
            if k and k != "#":
                seen[k] = seen.get(k, 0) + 1
    thresh = max(3, int(len(pages) * 0.3))
    banner = {k for k, n in seen.items() if n >= thresh}
    out = []
    for p, edge in zip(pages, edges):
        kept = []
        for i, ln in enumerate(p.splitlines()):
            if i in edge and (_PAGENO.match(ln) or norm(ln) in banner):
                continue
            kept.append(ln)
        out.append("\n".join(kept))
    return out


def _fitz():
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise MissingDependency(f"PyMuPDF (fitz) required for PDFs — "
                                f"run ./install.sh --pdf to add it ({e})")
    # Malformed-but-recoverable PDFs make MuPDF spew "syntax error: …" lines straight to
    # stderr (from C, bypassing logging) — a firehose on a big crawl.  Silence it ONCE,
    # process-wide; we drain the accumulated warnings per-file into our own debug log
    # instead, so they stay visible at log_level=DEBUG without drowning a normal run.
    global _errors_muted
    if not _errors_muted:
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:                      # older/newer API — best-effort, never fatal
            pass
        _errors_muted = True
    return fitz


def _drain_warnings(fitz, path: str) -> None:
    """Clear MuPDF's accumulated warnings for the file just parsed and note the count at
    DEBUG — recoverable syntax warnings mean 'repaired', not 'failed', so they're not
    worth an INFO/WARN line each, but the count is handy when a specific PDF misbehaves."""
    try:
        w = fitz.TOOLS.mupdf_warnings(reset=True)
    except Exception:
        return
    if w:
        n = sum(1 for line in str(w).splitlines() if line.strip())
        log.debug("%s: %d recoverable MuPDF warning(s) (text still extracted)",
                  os.path.basename(path), n)


def _ocr_page(src_path: str, page_no: int) -> str:
    """OCR a single page via ocrmypdf, returning its extracted text (or "")."""
    fitz = _fitz()
    try:
        with tempfile.TemporaryDirectory() as td:
            one = os.path.join(td, "p.pdf")
            out = os.path.join(td, "o.pdf")
            doc = fitz.open(src_path)
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=page_no, to_page=page_no)
            sub.save(one)
            sub.close(); doc.close()
            r = subprocess.run(
                ["ocrmypdf", "--quiet", "--force-ocr", "--optimize", "0", one, out],
                capture_output=True, timeout=180)
            if r.returncode != 0:
                return ""
            d = fitz.open(out)
            text = d[0].get_text("text")
            d.close()
            return text or ""
    except (subprocess.SubprocessError, OSError, RuntimeError) as e:
        log.debug("OCR failed on page %d: %s", page_no, e)
        return ""


def extract(path: str, cfg: dict):
    fitz = _fitz()
    doc = fitz.open(path)                       # a truly unopenable file raises → caught upstream
    blocks = []
    skipped = 0
    try:
        try:
            meta_title = (doc.metadata or {}).get("title") or ""
        except Exception:
            meta_title = ""
        title = meta_title.strip() or os.path.splitext(os.path.basename(path))[0]

        # TOC: list of [level, title, page] -> nearest heading per page
        try:
            toc = doc.get_toc() or []
        except Exception:                      # a broken outline shouldn't cost us the text
            toc = []
        page_heading = {}
        if toc:
            stack = []
            for level, htitle, page in toc:
                stack[:] = stack[:level - 1] + [htitle.strip()]
                page_heading[max(0, page - 1)] = " > ".join([h for h in stack if h])

        cur_heading = ""
        min_chars = cfg.get("ocr_min_chars", 32)
        want_ocr = cfg.get("ocr", True)
        try:
            page_count = doc.page_count
        except Exception:
            page_count = 0
        # Load pages by index and guard each one: a single malformed page (the kind that
        # raises rather than merely warning) is skipped, not allowed to sink the whole doc.
        for i in range(page_count):
            if i in page_heading:
                cur_heading = page_heading[i]
            try:
                text = doc.load_page(i).get_text("text") or ""
            except Exception as e:
                log.debug("%s: unreadable page %d skipped (%s)",
                          os.path.basename(path), i, e)
                skipped += 1
                continue
            if want_ocr and len(text.strip()) < min_chars:
                try:
                    ocr = _ocr_page(path, i)
                    if len(ocr.strip()) > len(text.strip()):
                        text = ocr
                except Exception:
                    pass
            if text.strip():
                blocks.append((cur_heading, text))
        # page furniture (running headers/footers, bare page numbers) is noise
        # to the distiller and pollutes chunk text — strip it corpus-wide here
        stripped = _strip_furniture([t for _, t in blocks])
        blocks = [(h, t) for (h, _), t in zip(blocks, stripped) if t.strip()]
    finally:
        try:
            doc.close()
        except Exception:
            pass
        _drain_warnings(fitz, path)
    if skipped:
        log.info("%s: %d page(s) unreadable, skipped; kept %d block(s)",
                 os.path.basename(path), skipped, len(blocks))
    return title, blocks
