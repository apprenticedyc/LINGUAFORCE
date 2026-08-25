# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\CHANGELOG.md'
add = """
### 归档（2026-08-22）
- `main.tex.bak_iaa`：含 IAA 双人一致性研究 + 全部低成本补强（T/G/B、提示词消融、t-SNE、伦理/Datasheet）的 main.tex 快照。
- `main_iaa_20260822.pdf`：对应 PDF（14 页）。
- 后续页数压缩等修改可在该快照基础上进行，便于回退对比。
"""
io.open(p, 'a', encoding='utf-8').write(add)
print('CHANGELOG updated')