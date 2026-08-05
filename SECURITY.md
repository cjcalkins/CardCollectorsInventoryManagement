# Security

## Reporting

This is a personal, locally-run tool. If you find a security problem, open an issue
describing the impact and the steps to reproduce it. Do not include live credentials in
the issue text.

## Handling secrets in this repo

- Credentials belong in `.env` (ignored) or in the app's settings UI — never in a tracked file.
- `.gitignore` matches key material **by extension** (`*.key`, `*.pem`, `*.crt`, `*.cer`,
  `*.p12`, `*.pfx`) and any `.env.*` except the tracked `.env.sample` placeholder.
- `inventory.db` holds live marketplace tokens and the mailbox password (see the note at
  the top of `models.py`). **Treat the database file itself as a secret** — it is ignored
  by `*.db`, and migration bundles under `uploads/migration_exports/` contain the same
  material in plaintext.

## Incident: credentials committed to git history (2026-07 / 2026-08)

### What was exposed

| Secret | Added | Removed | Status |
|---|---|---|---|
| `certs/cardcollector.key` — 2048-bit RSA private key | `65e455c4` (2026-07-14) | `0787925` (2026-07-29) | **Public forever.** Superseded in code — see below. |
| `certs/cardcollector.crt` — self-signed cert, CN `cardcollector.local`, valid to 2036-07-11 | `65e455c4` | `0787925` | Public forever. |
| `.env` — `JUSTTCG_API_KEY` (`tcg_d9a3…`) | `b831b1bd` (2026-05-18) | `d90abbf7` (2026-05-23) | **Revoked and reissued 2026-08-05.** |

### The part that was missed the first time

Both removals were **delete-commits, not history rewrites**. The blobs remain reachable
from `origin/main` and can be recovered by anyone who can read the repository:

```
git show 65e455c4:certs/cardcollector.key
```

Deleting a file from the tip does not un-publish it. A secret that has been pushed is
compromised permanently, and the only real remedy is to **rotate it at the source**.

Worse, the application actively kept using the exposed key. `_ensure_self_signed_cert()`
returned any existing cert/key pair unconditionally, and the committed certificate is
valid until 2036 — so every clone still holding that pair would have gone on serving a
downloadable private key for another decade, silently.

### What was done

1. **The JustTCG API key was revoked and reissued** at the provider (2026-08-05). This is
   the step that actually ends the exposure; no code change substitutes for it.
2. **The app now refuses the leaked keypair.** `_ensure_self_signed_cert()` fingerprints
   the on-disk private key (SHA-256 over its DER `SubjectPublicKeyInfo`) and, on a match
   against `_COMPROMISED_SPKI_SHA256`, deletes the pair and generates a fresh one instead
   of serving it. Only a *positive* match discards anything — an unreadable key is left
   alone so a transient error cannot churn a legitimate certificate on every boot.
   Delete `certs/cardcollector.*` on any machine to force regeneration immediately.
3. **`.gitignore` now matches key material by extension**, not only the `certs/` folder
   the key happened to live in.

### Still open

Rewriting history to purge the blobs (`git filter-repo` + force-push) has **not** been
done. It rewrites every published commit SHA and breaks every existing clone, so it is an
owner decision. It is also the less important half: once a secret is rotated and the app
stops serving it, what remains in history is dead material. Note that GitHub may retain
unreachable objects even after a force-push, which is a further reason rotation — not
rewriting — is the load-bearing step.

## If a secret is committed again

1. **Rotate it first.** Revoke at the provider and issue a new one. Do this before
   anything else; every other step is cleanup.
2. Remove the file and confirm `.gitignore` covers its *shape*, not just its path.
3. If the secret is a key the app consumes, make the app reject it by identity — a
   deleted file that the code still happily loads from disk is not a fix.
4. Only then decide whether a history rewrite is worth the disruption.
