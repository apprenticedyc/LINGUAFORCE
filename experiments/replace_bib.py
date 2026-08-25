# -*- coding: utf-8 -*-
import os
src = r'E:\PythonCode\Paper\linguistic_agency_paper\references.bib'
tmp = r'E:\PythonCode\Paper\linguistic_agency_paper\references_new.bib'
try:
    os.replace(tmp, src)
    print('REPLACED OK, size =', os.path.getsize(src))
except Exception as e:
    print('replace failed:', e)
    try:
        os.remove(tmp)
    except Exception:
        pass