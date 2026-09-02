"""PDF truncation marker must advertise the paging tool (2026-09-02 class).

A 35-page PDF was injected with the plain marker "full text available in the
document viewer" — a human-only affordance. The model never learned the full
text was saved as a page-able Document, so it hallucinated content from pages
it never received. The Office path already upgrades its marker with the
doc_id + manage_documents offset recipe; the PDF path (both form and plain
branches) must do the same.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import document_processor


def _run_pdf_attachment(tmp_path, monkeypatch, body_text, doc_id="doc-123"):
    """Drive build_user_content down the plain-PDF branch with stubs."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    upload_id = pdf.name

    long_body = body_text
    monkeypatch.setattr(
        document_processor, "_process_pdf", lambda path, owner=None: long_body
    )
    import src.pdf_forms as pdf_forms
    import src.pdf_form_doc as form_doc

    monkeypatch.setattr(pdf_forms, "has_form_fields", lambda path: False)
    monkeypatch.setattr(form_doc, "create_plain_pdf_document", lambda **kw: doc_id)
    handler = SimpleNamespace(
        resolve_upload=lambda fid, owner=None: {
            "path": str(pdf),
            "mime": "application/pdf",
            "name": upload_id,
        },
        _inside_upload_dir=lambda path: True,
        is_image_file=lambda name, mime: False,
        is_audio_file=lambda name, mime: False,
        is_document_file=lambda name, mime: mime == "application/pdf",
    )
    result = document_processor.build_user_content(
        "see attached", [upload_id], str(tmp_path), handler,
        session_id="s1", auto_opened_docs=[], owner="admin",
    )
    if isinstance(result, list):
        result = result[0]["text"]
    return result


from types import SimpleNamespace


def test_truncated_pdf_marker_names_doc_and_paging_tool(tmp_path, monkeypatch):
    body = "A" * 20000  # over the 15k inline cap
    out = _run_pdf_attachment(tmp_path, monkeypatch, body)
    assert "full text available in the document viewer" not in out, "plain (hint-less) marker leaked"
    assert "saved as document `doc-123`" in out
    assert "manage_documents" in out and "action=read" in out
    assert "document_id=doc-123" in out and "offset=<N>" in out
    assert "20,000 chars" in out  # full length advertised


def test_short_pdf_has_no_marker(tmp_path, monkeypatch):
    out = _run_pdf_attachment(tmp_path, monkeypatch, "short body")
    assert "truncated for inline context" not in out
    assert "doc.pdf" in out or "PDF content" in out


def test_marker_suppressed_when_doc_creation_fails(tmp_path, monkeypatch):
    body = "B" * 20000
    out = _run_pdf_attachment(tmp_path, monkeypatch, body, doc_id=None)
    # Without a doc to point at, never advertise a paging tool for a
    # document that was not created. (With the doc branch skipped entirely,
    # the raw extractor fallback runs — uncapped in this stubbed harness.)
    assert "manage_documents" not in out
    assert "saved as document" not in out
    assert "offset=<N>" not in out
