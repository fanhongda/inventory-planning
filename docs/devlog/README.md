# Development log

One entry per working session, dated, newest last. This is **not** a changelog and not
a substitute for `git log` — those record what changed. This records what `git log`
structurally cannot:

- **Alternatives that were rejected, and why.** A commit shows the design that won. The
  three that lost, and the reason each one lost, are the expensive part to rediscover.
- **What real data turned out to be like.** Every threshold in `config/` is a judgement
  about somebody's export. Six months later the number is still there and the extract
  that justified it is not.
- **Why a constant is the value it is.** `_MIN_DENSITY_FOR_ARIMA = 0.5` has a paragraph
  of reasoning behind it. Most constants do not, and should.
- **Dead ends.** The thing that looked like it would work, didn't, and will look like it
  would work again next time.

## What does not go here

Anything already recorded elsewhere. Code structure is in the code; what changed is in
the commit; the invariants are in `.github/copilot-instructions.md`; the workflow is in
`.claude/skills/inventory-planning/SKILL.md`. An entry that restates one of those is
maintenance cost with no reader.

## Real data

**Entries in this directory are committed, so they must carry no customer data.** That
includes what looks harmless: material numbers, customer names, business-unit names,
revenue figures, SKU counts specific enough to identify an entity. Describe the *shape*
of what was found — "a two-level product hierarchy where the coarser level had two
spellings for one unit" — not the values.

Anything that needs the actual numbers to make sense goes in `docs/devlog/local/`,
which is git-ignored. Write it there and reference it from the committed entry by
filename, so a reader knows the detail exists and where.

## Format

`YYYY-MM-DD.md`, one file per day. Append to the day's file rather than creating a
second. Headings are free-form; a decision, its alternatives, and its consequence are
the three things worth writing down.
