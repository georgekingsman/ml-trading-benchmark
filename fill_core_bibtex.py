#!/usr/bin/env python3
"""
Auto-fill BibTeX metadata for core references using Crossref.
Usage:
  python fill_core_bibtex.py --titles core60_titles.csv --inbib references_eswa_core60.bib --outbib references_eswa_core60_filled.bib

Notes:
- Requires internet access.
- Crossref matching by title is heuristic; always spot-check the results.
"""
import argparse, csv, re, sys, time
from urllib.parse import urlencode
import requests

CROSSREF = "https://api.crossref.org/works"

def http_get_json(url: str, retries: int = 4) -> dict:
    headers = {"User-Agent": "bibtex-filler/1.0 (mailto:your-email@example.com)"}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [RETRY {attempt+1}] SSL/Connection error, waiting {wait}s ...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

def best_work_for_title(title: str) -> dict | None:
    # Query Crossref with a title
    params = {
        "query.title": title,
        "rows": 5,
        "select": "DOI,title,author,issued,container-title,type,page,volume,issue,URL,publisher,ISBN,ISSN"
    }
    url = CROSSREF + "?" + urlencode(params)
    data = http_get_json(url)
    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    # pick item with the highest "title similarity" using simple normalization
    def norm(s): return re.sub(r"\W+", "", (s or "").lower())
    t0 = norm(title)
    def score(it):
        t = norm((it.get("title") or [""])[0])
        # Jaccard-ish score on character bigrams
        def bigrams(x): return {x[i:i+2] for i in range(max(0, len(x)-1))}
        a, b = bigrams(t0), bigrams(t)
        return len(a & b) / max(1, len(a | b))
    items.sort(key=score, reverse=True)
    return items[0]

def work_to_bibtex(key: str, w: dict) -> str:
    # Minimal BibTeX serialization
    wtype = w.get("type", "misc")
    # Map Crossref types
    entry_type = {
        "journal-article": "article",
        "proceedings-article": "inproceedings",
        "book-chapter": "incollection",
        "book": "book",
        "posted-content": "misc",
        "report": "techreport",
    }.get(wtype, "misc")

    title = (w.get("title") or [""])[0]
    doi = w.get("DOI", "")
    url = w.get("URL", "")
    container = (w.get("container-title") or [""])
    container = container[0] if container else ""
    issued = w.get("issued", {}).get("date-parts", [[None]])[0]
    year = issued[0] if issued and issued[0] else None

    authors = w.get("author", []) or []
    def fmt_author(a):
        given = a.get("given","").strip()
        family = a.get("family","").strip()
        if given and family:
            return f"{family}, {given}"
        return family or given or ""
    author_str = " and ".join([fmt_author(a) for a in authors if fmt_author(a)]) or None

    fields = []
    def add_field(k,v):
        if v:
            v = str(v).replace("\n"," ").strip()
            fields.append(f"  {k} = {{{v}}}")
    add_field("title", title)
    add_field("author", author_str)
    add_field("year", year)
    add_field("journal" if entry_type=="article" else "booktitle" if entry_type=="inproceedings" else "howpublished", container)
    add_field("doi", doi)
    add_field("url", url)
    return "@%s{%s,\n%s\n}\n" % (entry_type, key, ",\n".join(fields))

def load_titles(path: str):
    titles = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            titles.append((row["key"].strip(), row["title"].strip()))
    return titles

def strip_existing_entry(bib: str, key: str) -> str:
    # remove any existing entry for key (simple regex)
    pat = re.compile(r'@\w+\{' + re.escape(key) + r',\s*(?:.|\n)*?\n\}\s*', re.M)
    return pat.sub("", bib)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", required=True)
    ap.add_argument("--inbib", required=True)
    ap.add_argument("--outbib", required=True)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    with open(args.inbib, "r", encoding="utf-8") as f:
        bib = f.read()

    filled = []
    titles_list = load_titles(args.titles)
    total = len(titles_list)
    for idx, (key, title) in enumerate(titles_list, 1):
        print(f"[{idx}/{total}] {key}: {title[:60]}...", file=sys.stderr)
        try:
            work = best_work_for_title(title)
            if not work:
                print(f"  [WARN] No Crossref match: {key}", file=sys.stderr)
                continue
            filled.append(work_to_bibtex(key, work))
        except Exception as e:
            print(f"  [ERROR] Skipped {key}: {e}", file=sys.stderr)
        time.sleep(args.sleep)

    # Replace entries
    for entry in filled:
        m = re.match(r'@\w+\{(ref\d+),', entry)
        if not m: 
            continue
        k = m.group(1)
        bib = strip_existing_entry(bib, k)
        bib += "\n" + entry

    with open(args.outbib, "w", encoding="utf-8") as f:
        f.write(bib)

    print(f"Done. Wrote: {args.outbib}")

if __name__ == "__main__":
    main()
