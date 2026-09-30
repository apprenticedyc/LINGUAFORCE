# Analysis Code

The scripts in this directory operate on `../data/` and `../results/` using
paths relative to the submission package.

The parser and annotation-provider runners are not included. They require an
external model service and credentials; the corresponding model outputs used
for the reported evaluation are in `../results/`.

The IAA scripts accept annotation workbooks explicitly with `--a`, `--b`, and
`--sample`. Human annotation workbooks are not bundled here.
