# Example: a multi-edition scripture study KB

Build one knowledge base that holds **several Bible editions lined up verse-for-verse**,
with **commentaries layered onto the passages they annotate** and every **cross-reference
turned into a graph edge** — then read any reference across all of it at once.

Everything here uses **public-domain** texts (Project Gutenberg), so there are no licence
problems sharing the resulting KB.

## Why it works

Every edition's verses land on the same **canonical key** (`bible:John.3.16`), regardless
of how that edition spells its books or numbers its chapters. So the King James and the
Douay-Rheims — which name books differently (Douay "3 Kings" *is* the KJV's "1 Kings";
Douay "1 Kings" *is* 1 Samuel) — converge on one node per verse. The Douay-Rheims is
recognised as an **edition** and its complete Vulgate/Latin map is applied automatically,
so you are never asked to hand-map "Kings" or worried about botching the numbering.

## Get some texts

- **King James Version** — Project Gutenberg (e.g. eBook #10), plain text.
- **Douay-Rheims (Challoner)** — Project Gutenberg eBook #8300, plain text.
- Any commentary in the public domain (Matthew Henry, the Challoner notes that ship
  inside the Douay-Rheims file itself, …).

Name each file with its siglum so the translation label is tidy: `kjv.txt`, `drb.txt`.

## Ingest

For each edition, let vinur look at it and confirm:

```
./vinur.sh analyze kjv.txt        # see what it detected; confirm the edition/toggles
./vinur.sh collect kjv.txt --to scripture.kdb --bundle scripture --answers-file kjv.json
./vinur.sh collect drb.txt --to scripture.kdb --bundle scripture --answers-file drb.json
```

or drop them in a source folder and let the **normal crawl** set them aside into the
**"Needs your input"** panel — confirm there, and the next crawl ingests them. Either way
the confirm step is one or two friendly questions:

- *"This looks like the Douay-Rheims (Latin Vulgate) edition. Apply its built-in reference
  map? (recommended)"* — yes.
- *"After ingesting, build the cross-reference graph?"* — yes (default).
- *"This edition has commentary/notes between the verses. Layer them onto the passages
  they annotate?"* — yes (default). (The Douay-Rheims carries the Challoner notes inline.)

The cross-reference graph is built automatically after ingest. To (re)build it by hand:

```
./vinur.sh citations
```

## Read it

```
./vinur.sh read John 3:16
```

```
John 3:16
  [KJV] For God so loved the world, that he gave his only begotten Son; …
  [DRB] For God so loved the world, as to give his only begotten Son.
  ↪ cross-references: 1 John 4:9
  ✎ A note. This verse teaches the charity of God toward mankind; … compare Romans 5:8.
```

`read` accepts a single verse or a range: `read 1 Corinthians 13:4-7`.

## What you get

- **Alignment** — the same reference in every ingested translation, side by side.
- **Cross-references** — a deterministic graph edge for every citation a verse (or a
  note) makes, so you can traverse the connections between disparate passages.
- **Commentary** — each note is a node that *annotates* its verse, and the note's own
  scripture references become links too.
- **Semantic cards** (when you run `distill` with an LM) — per-verse *theme* and
  *parallel* cards; for legal corpora, *definition* / *obligation* / *exception* cards.

## Notes & limits

- **Gospels, Epistles, most of the OT** line up cleanly across editions.
- **Psalms 10–146** diverge between the Vulgate (Douay) and Hebrew (KJV) numbering; those
  are ingested **as printed**, so a Douay Psalm and its KJV counterpart may sit on
  neighbouring keys until a Vulgate↔Hebrew Psalm table is supplied (the edition map's
  `key_aliases` is where it goes).
- **Deuterocanon** (Tobit, Sirach, 1–2 Maccabees, …) is first-class, so a Catholic edition
  ingests whole; a KJV simply has no verses on those keys.
