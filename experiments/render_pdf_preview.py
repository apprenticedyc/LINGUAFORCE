import os, glob
pdf = r'E:\PythonCode\Paper\linguistic_agency_paper\main.pdf'
outdir = r'E:\PythonCode\Paper\linguistic_agency_paper\_preview'
os.makedirs(outdir, exist_ok=True)
for old in glob.glob(os.path.join(outdir, '*.png')):
    os.remove(old)
try:
    import fitz
    doc = fitz.open(pdf)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(os.path.join(outdir, f'p{i+1}.png'))
    print('rendered', len(doc), 'pages with pymupdf')
except Exception as e:
    print('pymupdf failed:', e)
    try:
        from pdf2image import convert_from_path
        imgs = convert_from_path(pdf)
        for i, im in enumerate(imgs):
            im.save(os.path.join(outdir, f'p{i+1}.png'))
        print('rendered', len(imgs), 'pages with pdf2image')
    except Exception as e2:
        print('pdf2image failed:', e2)
