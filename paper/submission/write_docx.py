#!/usr/bin/env python3
"""Minimal .docx writer. No python-docx dependency."""

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def _document_xml(paragraphs):
    runs = []
    for p in paragraphs:
        if p is None:
            continue
        if isinstance(p, tuple) and p[0] == "h":
            runs.append(
                f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
                f'<w:r><w:t>{escape(p[1])}</w:t></w:r></w:p>'
            )
        elif isinstance(p, tuple) and p[0] == "b":
            runs.append(
                f'<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
                f'<w:r><w:t>{escape(p[1])}</w:t></w:r></w:p>'
            )
        else:
            runs.append(f'<w:p><w:r><w:t xml:space="preserve">{escape(str(p))}</w:t></w:r></w:p>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>' + "".join(runs) + "</w:body></w:document>"
    )


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def write_docx(path, paragraphs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", _document_xml(paragraphs))
    return path
