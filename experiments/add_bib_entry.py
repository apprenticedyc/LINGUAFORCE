# -*- coding: utf-8 -*-
import io, os
src = r'E:\PythonCode\Paper\linguistic_agency_paper\references.bib'
tmp = r'E:\PythonCode\Paper\linguistic_agency_paper\references_new.bib'
bib = open(src, encoding='utf-8-sig').read()
entry = '''
@inproceedings{gebru2018datasheets,
  title={Datasheets for Datasets},
  author={Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Iii, Hal Daum\'e and Crawford, Kate},
  journal={Communications of the ACM},
  volume={64},
  number={12},
  pages={86--92},
  year={2021}
}
'''
open(tmp, 'w', encoding='utf-8').write(bib.rstrip() + '\n' + entry + '\n')
print('tmp written OK', os.path.getsize(tmp))