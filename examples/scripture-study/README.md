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

## The Psalms — reconciled automatically

The Vulgate (Douay) and Hebrew (KJV) traditions number the Psalms differently — Vulgate 9
is Hebrew 9+10, Vulgate 10–112 are Hebrew 11–113, and so on — and the *verse* numbers
drift too, because the Douay counts the Latin titles as verses (Douay 50:3 *"Have mercy
on me"* is KJV 51:1).  vinur does **not** guess any of this from a table: after both
editions are in, it recovers each psalm's exact verse offset **from the two texts
themselves** (matching the wording — they translate the same content), then writes the
Vulgate→Hebrew key aliases into the Douay document's reference map.  This runs
automatically at ingest (config `auto_reconcile`), or by hand:

```
./vinur.sh psalms
```

- Numbered Latin titles move to a **verse-0 superscription slot** (`Psalm 51:0`), so a
  title never masquerades as the Hebrew psalm that shares its number.
- A psalm whose offset can't be established with confidence is **left on its own keys
  and listed** — never guessed.
- On the real Gutenberg files this aligns all 144 divergent psalms (the split Vulgate 9,
  the combined 113, the joined 114+115 and 146+147, and every titled psalm) — so
  `read Psalms 23:1` shows the KJV's *"The LORD is my shepherd"* beside the Douay's
  *"The Lord ruleth me"*, with the Challoner note attached.

## Notes & limits

- Verified against the complete Gutenberg texts: the KJV ingests **31,102 verses in 66
  books** (the exact canonical count) and the Douay-Rheims **35,630 units in 73 books**,
  with no duplicate keys — flowed verse paragraphs, wrapped lines, alternate book titles
  ("Otherwise Called: …") and the Vulgate Psalm 9 restart are all handled.
- **Deuterocanon** (Tobit, Sirach, 1–2 Maccabees, …) is first-class, so a Catholic edition
  ingests whole; a KJV simply has no verses on those keys.
