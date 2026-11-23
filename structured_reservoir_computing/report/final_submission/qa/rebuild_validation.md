# Extracted package rebuild

The report package was extracted to a fresh temporary directory. SHA-256 verification passed for all listed members. The manuscript assembler regenerated the bundled Markdown to the same checksum. The document builder then compiled the Word and PDF files from the extracted materials, with no raw MATLAB inputs or source-project checkout. The native-document checker passed on the rebuilt files.

All 53 original and rebuilt PDF pages had identical extracted text and raster pixels at 72 dpi in the reference environment. The rebuilt report retained 16 verified contents entries, 226 editable equation objects, 19 editable data tables, 17 technical figures and 59 bibliography entries. The Word file in the project's `outputs` folder is byte-for-byte identical to the canonical submission file.
