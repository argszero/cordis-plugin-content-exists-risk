# @argszero/cordis-plugin-content-exists-risk

A rescue for a dsh session the provider refuses to receive.

DeepSeek's content moderation can reject an entire request:

```
HTTP 400  {"message":"Content Exists Risk","code":"INVALID_REQUEST"}
```

The rejection is a property of *content*, and the content that trips it is already
durable. Because dsh re-derives the whole conversation from the session log on
every turn (`GenerateOptions.messages` — "exactly as the provider sees them"),
**one stored record that trips moderation makes every later turn of that session
fail**, whatever you now type. Reported cases: a long session whose history holds
symbols fetched from a web page — a two-character new message failed identically
([#7181](https://github.com/deepseek-ai/deepseek-harness/discussions/7181)); a
brand-new session that failed on its first turn
([#6495](https://github.com/deepseek-ai/deepseek-harness/discussions/6495)); a
session whose failure is inherited by every session that reads it
([#6476](https://github.com/deepseek-ai/deepseek-harness/discussions/6476)).

The log's content is not editable, and the built-in retry policy is right not to
retry a deterministic 400. So the only lever left is **what is sent**, not what is
stored — and that is what this plugin moves.

## What it does

Mounted on the public `llm/stream` waterfall, it inserts a space between every
pair of adjacent characters when neither of them is a space. Moderation matches
character sequences; a spaced-out sequence is not that sequence.

```
FLAG-TOKEN-1234   ->   F L A G - T O K E N - 1 2 3 4
```

Three properties make that usable rather than reckless:

| | |
|---|---|
| **A healthy session pays nothing** | The rewrite is *reactive*. An untouched request takes the identity path — `next()` with your own object — so a session that never trips moderation pays no tokens, no cache invalidation, and no re-dispatch. |
| **Cost is spent newest-first, only as far as needed** | The scope is a ladder (`ladder`, default `[4, 32, 0]`): a rung's *rejection* is the evidence it was not enough, so the next rung rewrites more. Nothing is paid for messages the retry proved irrelevant. Spacing roughly doubles the characters it touches, so scope is the only real currency. |
| **A refused attempt is never shown** | A rejection arrives before the provider streams anything, so the terminal frame of a refused attempt is held back while another rung is available, and only the rung that was actually accepted is streamed. When every rung is refused, the provider's own failure is surfaced unchanged — this plugin never turns a failure into a success it did not observe. |

Once a rung is accepted, the session is latched to it (in memory, for the life of
the process), so later requests skip the untouched probe and start at the rung
already known to work.

## Install

```sh
# from the npm registry, once published there
npm install @argszero/cordis-plugin-content-exists-risk

# from this repository (only this verb attaches a plugin to a profile)
dsh plugin --profile <profile> add github:argszero/cordis-plugin-content-exists-risk
```

`npm install` puts the package in a tree; it does not register the bundle, so a
`dsh` profile never mounts it that way. `dsh plugin add` writes the entry into
`dsh.profile.bundles`.

Then mount it, by adding to your profile's config:

```yaml
- insert:
    - id: content-exists-risk
      name: '@argszero/cordis-plugin-content-exists-risk'
```

The package ships a `cordis.patch.yml` (`dsh.bundle.patch`), so a bundle-aware
profile picks it up without any further wiring. Mounting needs no configuration.

## Configuration

| key | default | meaning |
|---|---|---|
| `mode` | `auto` | `auto` probes untouched and rewrites only after a rejection; `always` rewrites from the first dispatch; `observe` reports what it *would* rewrite and dispatches unchanged; `off` leaves every request alone. |
| `ladder` | `[4, 32, 0]` | Rungs, in newest-message counts. `0` means every message and may only be the last rung. Must escalate. |
| `roles` | `['user', 'tool']` | Roles whose content may be rewritten. User input and tool results are the two places fetched content lands, and they are two message roles from the 0.2.1 line on (before it, a tool result arrived as user-role content). Naming roles explicitly replaces the default, so `roles: [user]` narrows the mount back to user input alone. |
| `includeAuxiliary` | `false` | Whether `purpose`-carrying calls (compaction, session title) are rewritten. Their output is folded back into the session, so a spaced request yields a spaced summary. |
| `maxChars` | `200000` | Characters added per request before a pass stops. `0` disables the budget. |
| `matchPattern` | `content[_ ]?exists[_ ]?risk` | Regex (case-insensitive) matched against the provider's failure message. |
| `matchCodes` | `[]` | Extra failure codes treated as a content rejection. Empty by default — the generic 400 code `INVALID_REQUEST` also covers malformed requests and oversized bodies, so it is not a moderation signal on its own. Add it for a gateway that mangles the message. |
| `statuses` | `[400]` | HTTP statuses a rejection must carry (a failure reporting no status passes, since a gateway may not forward one). |
| `warnLimit` | `5` | Bound on emitted warnings per mount. `0` silences them. |

A session that is already dead when you mount this plugin is rescued on its next
turn: the first request pays one refused probe, the second pays the rung that
works. Set `mode: always` if you would rather skip the probe and rewrite from the
start.

## What it does not fix

- **A trigger in assistant or system content** — out of default scope. Add
  `assistant` to `roles`. Model output is not where fetched content enters, and
  spacing it costs the most tokens while inviting the model to imitate spaced-out
  text.
- **A trigger inside `tool-call` arguments** — never rewritten. `arguments` is a
  raw JSON string; a character-level transform would corrupt the call's syntax,
  and a JSON-aware one is a different feature with different hazards.
- **A trigger inside a tool schema (`options.tools`) or a one-shot `system`
  field** — never rewritten. A schema's string leaves include `enum`, `const` and
  `pattern`, where inserting a space changes the schema's meaning.
- **Persistence** — the latch is an in-process, bounded map keyed by `sessionId`.
  A restarted daemon pays one extra probe per poisoned session; nothing is
  written to disk or to the session log.
- **Fidelity** — the rewritten text is what the model reads and answers. It is
  legible, but it is not the original, and a rung that fires invalidates the
  provider's prefix cache from the first rewritten message onward. Both are real
  costs, paid only by a session whose alternative is failing every turn.

## How it hooks in

`llm/stream` is a waterfall, but the last argument is the innermost `next` and
cordis closes over the listener's own argument list — `next` takes no parameters,
so a listener cannot substitute the options object for the rest of the chain. A
loop-built request additionally arrives deep-frozen (its content is "a pure
function of the session log"), so mutation throws outright. A fresh dispatch
through `ctx.llm.stream()` is therefore the only way to change what the adapter
receives. The plugin registers with `prepend: true`, so every other listener sees
exactly one dispatch per proposed request — the one that is sent — and keeps a
`WeakSet` of its own replacements so the re-dispatch terminates.

The consequence is stated plainly in the module doc: a rewritten request is a new
object and does not carry `dsh-agent-loop`'s process-local marker, so the loop's
request-reconstruction invariant does not inspect it. That invariant exists to
keep a *sent* request equal to the session log's derivation; a request that could
not be sent at all is the one case where that equality is deliberately broken,
knowingly, and with a log line.

## Verification

`npm test` builds and runs 48 tests against a real `@deepseek-ai/cordis` context
and a real `@deepseek-ai/dsh-llm` runtime, with a stub provider that refuses a
request exactly the way moderation does (a terminal `error` finish, no content):

- the unmounted control arm fails on the same request — the defect is real, and
  the plugin is what fixes it;
- a poisoned session is rescued, in three adapter calls, and the consumer never
  sees a failure;
- an attempt that already streamed content is **not** retried;
- a rung that cannot reach the trigger does not retry forever, and the provider's
  own failure is then surfaced unchanged;
- a healthy request is dispatched as the identical object, with one adapter call;
- a differently worded 400 is not treated as moderation;
- `mode: off` / `observe` / `always`, `includeAuxiliary`, a deep-frozen
  loop-style request, and the diagnostic bound.

Nine mutations of the implementation were run against this suite (transform
disabled, hold-back guard removed, attachment protection removed, rung
off-by-one, latch not written, latch not read, rejection gate removed, an
undeclared import, a stale dependency) — each one turned the suite red.

## Supported lines

Each line below was measured separately: its own `npm install`, a build with
`tsc`, then the full suite, with the resolved `@deepseek-ai/dsh-llm` version
checked first. The suite is identical on all of them — 48 tests, 48 passing —
because the walk reads block content structurally rather than against a tag list,
and the default role scope names both spellings of a tool result.

| line | resolved version measured |
|---|---|
| 0.1.2 prerelease | `0.1.2-rc.1` |
| 0.1.3 prerelease | `0.1.3-alpha.2` |
| 0.1.5 prerelease | `0.1.5-alpha.1` |
| 0.1.6 prerelease | `0.1.6-alpha.1` |
| 0.2.1 prerelease | `0.2.1-rc.1`, `0.2.1-rc.2`, `0.2.1-rc.3` |

From `0.2.1-rc.1` on, tool results are first-class `role: 'tool'` messages and a
`tool-result` content block in interpreted content is a hard admission refusal;
before that they were a `tool-result` container inside a user-role message. Both
shapes are in default scope, so one mount covers either line. What does **not**
change: an assistant or system message is still opt-in, and tool-call arguments,
tool schemas and `options.system` are still never rewritten.

## Source

- [#7181 — long session blocked by input moderation (400 Content Exists Risk / INVALID_REQUEST)](https://github.com/deepseek-ai/deepseek-harness/discussions/7181)
- [#6495](https://github.com/deepseek-ai/deepseek-harness/discussions/6495), [#6490](https://github.com/deepseek-ai/deepseek-harness/discussions/6490), [#5445](https://github.com/deepseek-ai/deepseek-harness/discussions/5445), [#6476](https://github.com/deepseek-ai/deepseek-harness/discussions/6476)

## License

MIT
