# -*- coding: utf-8 -*-
entry = '''@inproceedings{gebru2018datasheets,
  title={Datasheets for Datasets},
  author={Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Iii, Hal Daum\'e and Crawford, Kate},
  journal={Communications of the ACM},
  volume={64},
  number={12},
  pages={86--92},
  year={2021}
}
'''
open(r'E:\PythonCode\Paper\linguistic_agency_paper\references_extra.bib', 'w', encoding='utf-8').write(entry)
print('references_extra.bib written')