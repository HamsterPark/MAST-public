"""Build full-text search indexes for the PDFs already sitting in the corpus.

``search_fulltext`` searches ``chunks.parquet`` + ``chunk_vectors.npy``, which
only exist for papers that went through ``ingest_pdf``. PDFs an operator dropped
into the corpus by hand have never been through it, so passage search cannot see
them at all — on this machine that was 24 papers and zero searchable chunks.

This script closes that gap: it runs ingest over every PDF in the corpus that has
no chunks yet.

    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/index_local_papers.py --dry-run
    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/index_local_papers.py

**``--promote`` is off by default, on purpose.** Promoting appends a row per
paper to the 205 MB big index, which is an irreversible write to a shared asset
that nothing here can undo. Indexing for passage search needs none of that: the
chunks live beside the PDF. Turn promotion on only when you actually want these
papers to become findable in the 50k corpus search as well.

Costs a DashScope embedding call per batch of chunks (a paper is 17–33 chunks).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MASTV2 = Path(__file__).resolve().parents[1]
if str(_MASTV2) not in sys.path:
    sys.path.insert(0, str(_MASTV2))


def _corpus_pdfs() -> list[Path]:
    """Every PDF the literature tools consider a paper (attachments excluded)."""
    from mast.agents.literature import tools as littools
    return sorted(littools._all_pdfs())


def _already_indexed(pdf: Path) -> bool:
    """True when this PDF's own directory already holds chunks.

    Only meaningful for the ingest layout (``<slug>/source.pdf``); a hand-dropped
    ``Foo.pdf`` has no directory of its own and always reports False.
    """
    return pdf.name.lower() == "source.pdf" and (pdf.parent / "chunks.parquet").is_file()


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be indexed and exit")
    ap.add_argument("--promote", action="store_true",
                    help="ALSO append each paper's abstract to the big 50k index "
                         "(irreversible; off by default)")
    ap.add_argument("--limit", type=int, default=0,
                    help="index at most N papers (0 = no limit)")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip the OCR fallback for PDFs with no text layer")
    args = ap.parse_args(argv)

    from mast.knowledge.ingest import ingest_pdf
    from mast.knowledge.paths import papers_dir

    pdfs = [p for p in _corpus_pdfs() if not _already_indexed(p)]
    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    print(f"papers dir : {papers_dir()}")
    print(f"to index   : {len(pdfs)} PDF(s)")
    print(f"promote    : {'YES — writes into the big index' if args.promote else 'no'}")
    if not pdfs:
        print("nothing to do — every paper in the corpus already has chunks.")
        return 0
    for p in pdfs:
        print(f"  - {p}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    ok = failed = 0
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name} …", flush=True)
        try:
            res = ingest_pdf(str(pdf), promote=args.promote, ocr=not args.no_ocr)
        except Exception as exc:  # noqa: BLE001 — one bad paper must not stop the run
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        status = getattr(res, "status", "")
        if status in ("ingested", "replaced", "noop"):
            ok += 1
            print(f"    {status}: work_id={res.work_id} chunks={res.n_chunks}"
                  + ("  [OCR]" if getattr(res, "ocr_used", False) else ""))
        else:
            failed += 1
            print(f"    {status}: {getattr(res, 'detail', '')}")

    print(f"\ndone: {ok} indexed, {failed} failed.")
    print("search them with the literature agent's search_fulltext tool.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
