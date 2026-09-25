from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
from django.test import SimpleTestCase

from qa.ingestion import discover_documents, extract_pdf_pages, validate_document


class IngestionTests(SimpleTestCase):
    def test_discover_documents_finds_pdfs_only(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "keep.pdf").write_bytes(b"pdf")
            (root / "ignore.txt").write_text("not a pdf")
            nested_dir = root / "nested"
            nested_dir.mkdir()
            (nested_dir / "also.pdf").write_bytes(b"pdf")

            docs = discover_documents(root)

            self.assertEqual(len(docs), 2)
            self.assertTrue(all(path.suffix.lower() == ".pdf" for path in docs))

    def test_validate_document_accepts_pdf_and_rejects_unsupported_files(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"pdf")
            txt_path = Path(temp_dir) / "sample.txt"
            txt_path.write_text("plain text")

            self.assertTrue(validate_document(pdf_path))
            self.assertFalse(validate_document(txt_path))

    def test_extract_pdf_pages_keeps_filename_and_page_number(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "This is a test page.")
            document.save(pdf_path)
            document.close()

            pages = extract_pdf_pages(pdf_path)

            self.assertTrue(pages)
            first = pages[0]
            self.assertEqual(first["filename"], pdf_path.name)
            self.assertIsInstance(first["page_number"], int)
            self.assertIn("This is a test page", first["text"])
