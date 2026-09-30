"""Recompute D5 agreement for two completed annotation workbooks."""

import argparse
from pathlib import Path

from compute_iaa import cohens_kappa, load_xlsx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()

    a = load_xlsx(args.a)
    b = load_xlsx(args.b)
    ids = sorted(set(a) & set(b))
    pairs = [(a[i]["D5"], b[i]["D5"]) for i in ids]
    kappa = cohens_kappa(
        [x for x, _ in pairs], [y for _, y in pairs], maxcat=3, weighted=True
    )
    raw = sum(x == y for x, y in pairs) / len(pairs)
    print(f"n={len(pairs)}")
    print(f"raw_agreement={raw:.3f}")
    print(f"quadratic_weighted_kappa={kappa:.3f}")


if __name__ == "__main__":
    main()
