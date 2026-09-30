# English colour paper template contract

Reference: `template_zh.docx`, retained unchanged from the Chinese paper. The original has 18 rendered pages and one A4 portrait section. Reference and final file hashes are recorded by `build_english.py` in `template_fidelity.json`. The previously inspected reference render is in the parent workspace `report_output/qa_paper_final`.

## Required fidelity

Keep the source section geometry: 2.5 cm margins, 1.3 cm header/footer distance, a single column, centred PAGE footer. Keep Title 20 pt, Heading 1/2/3 16/14/12 pt, Normal 12 pt, Caption and table text 10.5 pt. Western text uses Times New Roman; equations retain Cambria Math. Body line spacing remains 1.5 with a 24 pt first-line indent. Preserve paragraph properties, heading keep-with-next, three-line table rules, repeated headers, native equations and their numbering, image dimensions, relationships and all styles.

## Editable slots

All Chinese body text becomes English, indexed by direct `word/document.xml/w:body/w:p` order in `english_text.json`. All table cells containing Chinese become English using direct table/row/cell paths in `build_english.py`. The nine main sections, references and two appendices remain. Known-answer results in Table 12 and validation wording in Section 7/Appendix A are updated to the integrated validation evidence. Author and contribution placeholders remain visibly unfilled.

Three existing image parts are replaced with colour plots of the same shape and size. Figure captions are adjusted to the actual colours. Original native OMML equations, footer field, header/footer parts, numbering, styles and other package parts remain byte-identical. Only main-document text, image bytes and title/subject/creator metadata may change. No content controls or text boxes are added. English text may naturally reflow to a different page count; fonts and page geometry must not be shrunk to match the Chinese page count.

The contribution heading begins a new page and its introductory paragraph stays with the table, preserving the assignment's request for a contribution page after English reflow.

## Checks

The builder asserts that no Chinese visible text remains, verifies exactly three images, and audits that only the permitted package parts changed. The source has already been visually inspected on all 18 pages. Render the final English copy through Word and inspect every final page for table wrapping, caption attachment, equation alignment and overflow. The packaged renderer cannot run in this Windows environment because no bundled soffice is available; Word COM export and bundled Poppler provide the render fallback. QA PDF/page images are intermediates, not deliverables.
