"""Download the 28 public networks listed in data/networks.csv into data/real/ and check their SHA-256.

The networks are not redistributed with this repository; each is fetched unchanged from the Netzschleuder repository.
Each network keeps its own license and citation (column `citation`; see also the network's page on the repository).
Usage: python data/download_networks.py [--only key1 key2 ...]
"""
import argparse, csv, hashlib, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); a = ap.parse_args()
    out = HERE / "real"; out.mkdir(exist_ok=True)
    bad = 0
    for row in csv.DictReader(open(HERE / "networks.csv")):
        if a.only and row["key"] not in a.only:
            continue
        dst = out / f"{row['key']}.zip"
        if not dst.exists():
            print("downloading", row["url"])
            urllib.request.urlretrieve(row["url"], dst)
        ok = sha256(dst) == row["sha256"]
        print(f"{row['key']:45s} {'ok' if ok else 'SHA-256 MISMATCH'}")
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
