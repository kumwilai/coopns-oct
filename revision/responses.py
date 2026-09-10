# -*- coding: utf-8 -*-
"""Content of the response to reviewers, assembled from three data files.

comments.json       the reviewer comments, quoted verbatim, never edited
responses_data.json our reply and the list of changes, keyed by the same labels
front_matter.json   the opening and the summary of changes

Keeping the text in data files means an edit can never break the module.
Edit the JSON, then run build_response_docx.py.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


COMMENTS = _load("comments.json")
DATA = _load("responses_data.json")
FRONT = _load("front_matter.json")

META = {
    "manuscript": "Manuscript OJCS-2026-04-0386, Minor Revision",
    "title": ("Backbone-Modular Neuro-Symbolic Post-Correction for OCT Denoising "
              "via Explicit Predicate Maps and Stable Fuzzy Allocation"),
    "date": "Submitted 15 September 2026",
}

OPENING = FRONT["opening"]
SUMMARY_OF_CHANGES = FRONT["summary"]

_GROUPS = [
    ("Reviewer 1", "1."),
    ("Reviewer 2", "2."),
    ("Reviewer 3", "3."),
]


def _order(label):
    return [int(x) for x in label.replace("Comment ", "").split(".")]


REVIEWERS = []
for name, prefix in _GROUPS:
    items = sorted([l for l in COMMENTS if l.startswith("Comment " + prefix)], key=_order)
    REVIEWERS.append({
        "name": name,
        "preamble": "",
        "comments": [{
            "label": l,
            "comment": COMMENTS[l],
            "response": DATA.get(l, {}).get("response", "TBD"),
            "changes": DATA.get(l, {}).get("changes", []),
            "table": DATA.get(l, {}).get("table"),
        } for l in items],
    })

CLOSING = FRONT.get("closing", """
We thank the reviewers again. The review process improved this work in a way that a lighter reading
would not have, because the questions about the fuzzy constants and about the release of the code are
what led us to find and repair a defect at the centre of the method. We hope the revised manuscript
meets the standard of the journal.
""")


def status():
    """How many responses are still outstanding."""
    todo = [l for l in COMMENTS if DATA.get(l, {}).get("response", "TBD") == "TBD"]
    return len(COMMENTS) - len(todo), len(COMMENTS), sorted(todo, key=_order)


if __name__ == "__main__":
    done, total, todo = status()
    print(f"{done} of {total} responses written")
    if todo:
        print("outstanding: " + ", ".join(todo))
