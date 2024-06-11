import glob
import pypdfium2 as pdfium
import json
import os

pdfs_path = 'data/pdfs'
paths = glob.glob(f'{pdfs_path}/*.pdf')


for path in paths:
    filename = os.path.basename(path)
    name = filename.replace('.pdf', '')
    json_filename = filename.replace('.pdf', '.json')

    data = []
    with open(f'data/crawled/{json_filename}', 'w') as f:
        pdf = pdfium.PdfDocument(path)
        n_pages = len(pdf)  # get the number of pages in the document
        for i, page in enumerate(pdf):
            # Load a text page helper
            textpage = page.get_textpage()
            # Extract text from the whole page
            text_all = textpage.get_text_range()
            data.append({
                'title': f"{name} - {i:04d}",
                'url': filename,
                'html': text_all
            })

    with open(f'data/crawled/{json_filename}', 'w') as f:
        json.dump(data, f, indent=4)