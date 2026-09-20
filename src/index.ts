/**
 * Content-moderation rescue for a session the provider refuses to receive.
 *
 * ## The gap
 *
 * A provider's content moderation can reject an entire request — DeepSeek
 * answers HTTP 400 `{"message":"Content Exists Risk","code":"INVALID_REQUEST"}`
 * (`httpErrorCode` maps a 400 to `INVALID_REQUEST`; the provider's `message`
 * survives into the failure). The rejection is a property of *content*, and the
 * content that triggered it is already durable: dsh re-derives the whole
 * conversation from the session log on every request — `GenerateOptions.messages`
 * is "ordered conversation messages, exactly as the provider sees them" — so one
 * stored record that trips moderation makes **every later turn of that session
 * fail**, whatever the user now types. Reported cases: a long session whose
 * history holds symbols fetched from a web page (discussion #7181 — a
 * two-character new message fails identically), a session that fails on its very
 * first turn (#6495, #6490), and a session whose failure is inherited by every
 * session that reads it (#6476).
 *
 * Nothing in the harness edits the log's content, and the built-in retry policy
 * is right not to: a deterministic 400 is not a transient failure. The session
 * is dead, and the only lever left is *what is sent*, not *what is stored*.
 *
 * ## What this plugin does
 *
 * It intervenes on the public `llm/stream` waterfall with one deliberately
 * dumb transform: **insert a space between every pair of adjacent characters
 * when neither of them is a space** (`spaceOut`). Moderation matches character
 * sequences; a spaced-out sequence is not that sequence. The transform is
 * idempotent — applying it twice changes nothing — which is also part of what
 * makes the re-dispatch below terminate.
 *
 * Three properties make it usable rather than reckless:
 *
 * 1. **A healthy session pays nothing.** An untouched request takes the identity
 *    path (`next()` with the caller's own object): no tokens, no cache
 *    invalidation, no re-dispatch. The rewrite is *reactive* — it fires only
 *    after a real rejection was observed and, per rung, never twice.
 * 2. **Cost is spent newest-first, and only as far as needed.** The user-visible
 *    rule is a ladder (`ladder`, default `[4, 32, 0]`): a rung's *failure* is
 *    the evidence that it was not enough, so the next rung rewrites more — the
 *    newest 4 messages, then the newest 32, then every message. Nothing is paid
 *    for messages the retry proved irrelevant. Spacing every character roughly
 *    doubles the characters it touches, so scope is the only real currency here.
 * 3. **The failure that triggered a retry never reaches the caller.** A rejected
 *    attempt produced no visible output before its terminal frame, so that frame
 *    is held back while another rung is still available, and only the rung that
 *    was actually accepted is streamed. When every rung is rejected, the
 *    provider's own terminal frame is surfaced unchanged: the plugin never turns
 *    a failure into a success it did not observe.
 *
 * Once a rung works, the session is *latched* to it in memory, so later requests
 * skip the doomed untouched probe and start at the rung already known to work.
 *
 * ## Why the rewrite is a re-dispatch, not an in-place edit
 *
 * `llm/stream` is a waterfall, but the listener's last argument is the innermost
 * `next` and cordis closes over the listener's own argument list: `next` takes
 * **no** parameters, so a listener cannot substitute the options object for the
 * rest of the chain. A loop-built request additionally arrives deep-frozen (its
 * content is "a pure function of the session log"), so mutation throws outright.
 * A fresh dispatch through `ctx.llm.stream()` is therefore the only way to change
 * what the adapter receives. This plugin registers with `prepend: true` so every
 * other listener sees exactly one dispatch per proposed request (the one that is
 * sent), and keeps a `WeakSet` of its own replacements so the re-dispatch
 * terminates.
 *
 * The consequence, stated plainly: a rewritten request is a new object, so it
 * does not carry `dsh-agent-loop`'s process-local marker
 * (`markAgentLoopRequest` is a `WeakSet` — a shallow copy is not a loop
 * request), and the loop's request-reconstruction invariant, which inspects only
 * marked requests, does not inspect it. That invariant exists to keep a *sent*
 * request equal to the session log's derivation; a request that could not be
 * sent at all is the one case where that equality is deliberately broken —
 * knowingly, and with a log line.
 *
 * ## Boundaries (what this plugin does not fix)
 *
 * - **Assistant and system content is opt-in.** `roles` defaults to `['user']`
 *   (which carries tool results, the shape most reported poisons arrive in).
 *   Model output is not where fetched content enters, spacing it costs the most
 *   tokens, and it invites the model to imitate spaced-out text. Add
 *   `'assistant'` only when the poison is known to sit there.
 * - **`tool-call` arguments are never rewritten.** `arguments` is a raw JSON
 *   string; a character-level transform would corrupt the call's syntax, and a
 *   JSON-aware one is a different feature with different hazards.
 * - **`options.tools` and `options.system` are never rewritten.** A tool
 *   schema's string leaves include `enum`, `const` and `pattern`, where inserting
 *   a space changes the schema's meaning; a one-shot caller's `system` field is
 *   the harness's own prompt assembly, not provider-fetched content.
 * - **Auxiliary calls are skipped by default.** Calls carrying a `purpose`
 *   (compaction, session title) are excluded unless `includeAuxiliary` is set:
 *   their *output* is folded back into the session, so a spaced request yields a
 *   spaced summary — a durable degradation bought for a call that is not part of
 *   the conversation.
 * - **The latch is in-process.** A bounded `Map` keyed by `sessionId`, dropped on
 *   restart; a restarted daemon pays one extra probe per poisoned session.
 *
 * ## Cost, stated honestly
 *
 * A rung that fires invalidates the provider's prefix cache from the first
 * rewritten message onward, and character spacing multiplies the affected text's
 * token count (`abc` becomes `a b c`). Both are real. They are paid only by a
 * session whose alternative is failing every turn, and only down to the
 * shallowest rung the provider accepted.
 *
 * @module @argszero/cordis-plugin-content-exists-risk
 */

import z from '@deepseek-ai/schemastery'
import type { Context } from '@deepseek-ai/cordis'
import type { ContentBlock, GenerateOptions, LlmFailure, Message, StreamChunk } from '@deepseek-ai/dsh-llm'

/** How the plugin behaves when it sees a request. */
export type Mode = 'off' | 'observe' | 'auto' | 'always'

/** Message roles whose content this plugin may rewrite. */
export type ScopedRole = 'system' | 'user' | 'assistant'

/** Deepest `tool-result` nesting traversed before a container is left alone. */
const MAX_DEPTH = 8

/** Bound on remembered session rungs; the entry used least recently is evicted. */
const MAX_LATCH_ENTRIES = 256

/**
 * Code points that must stay attached to the character before them, so no space
 * is inserted directly ahead of one: combining marks (`\p{M}`, including
 * variation selectors) belong to their base character, and a zero-width joiner
 * holds an emoji sequence together. Splitting either changes what the text
 * *renders* as, which is a fidelity cost this transform does not need to pay —
 * the character *sequence* is already broken by the spaces inserted around it.
 */
const ATTACHED = /[\p{M}\u200d]/u

/** Plugin configuration. Every field is optional; see {@link resolveConfig}. */
export interface Config {
  /**
   * `auto` (default) probes with the request untouched and rewrites only after a
   * rejection; `always` rewrites from the first dispatch of a request; `observe`
   * reports what would be rewritten and dispatches the request unchanged;
   * `off` leaves every request alone.
   */
  mode?: Mode
  /**
   * Escalating rungs, in newest-message counts: the request's last N messages
   * are rewritten, `0` meaning every message and allowed only as the final rung.
   * A rung is tried only when the previous one was rejected. Default `[4, 32, 0]`.
   */
  ladder?: number[]
  /** Roles whose content may be rewritten. Default `['user']` (carries tool results). */
  roles?: ScopedRole[]
  /** Whether `purpose`-carrying auxiliary calls are rewritten too. Default `false`. */
  includeAuxiliary?: boolean
  /**
   * Stop rewriting a request once this many characters have been added.
   * Counted per request and checked before each message, so one message can
   * overshoot it. `0` disables the budget. Default `200000`.
   */
  maxChars?: number
  /**
   * Regular expression (source, case-insensitive unless it says otherwise)
   * matched against the provider's failure message to recognize a
   * content-moderation rejection. Default `content[_ ]?exists[_ ]?risk`.
   */
  matchPattern?: string
  /**
   * Extra failure codes treated as a content rejection. Empty by default: the
   * generic 400 code `INVALID_REQUEST` also covers malformed requests and
   * oversized bodies, so it is not a moderation signal on its own. Add it for a
   * gateway that mangles the provider's message.
   */
  matchCodes?: string[]
  /** HTTP statuses a rejection must carry, when the failure reports one. Default `[400]`. */
  statuses?: number[]
  /** Bound on emitted warnings per mount. `0` silences logging. Default `5`. */
  warnLimit?: number
}

/** Configuration with every default materialized and validated. */
export interface ResolvedConfig {
  readonly mode: Mode
  readonly ladder: readonly number[]
  readonly roles: readonly ScopedRole[]
  readonly includeAuxiliary: boolean
  readonly maxChars: number
  readonly matchPattern: RegExp
  readonly matchCodes: readonly string[]
  readonly statuses: readonly number[]
  readonly warnLimit: number
}

/** The plugin name used by the mount patch. */
export const name = 'content-exists-risk'

/** This plugin dispatches through `ctx.llm.stream()` and injects its service. */
export const inject = ['llm']

/** Runtime schema for {@link Config}; cordis applies it before {@link apply}. */
export const Config: z<Config> = z.object({
  mode: z.union(['off', 'observe', 'auto', 'always']).default('auto'),
  ladder: z.array(z.number()).default([4, 32, 0]),
  roles: z.array(z.union(['system', 'user', 'assistant'])).default(['user']),
  includeAuxiliary: z.boolean().default(false),
  maxChars: z.number().default(200000),
  matchPattern: z.string().default('content[_ ]?exists[_ ]?risk'),
  matchCodes: z.array(z.string()).default([]),
  statuses: z.array(z.number()).default([400]),
  warnLimit: z.number().default(5),
})

const MODES: readonly Mode[] = ['off', 'observe', 'auto', 'always']

/** Thrown for a configuration {@link Config} cannot express but a caller can still pass. */
function invalid(detail: string): never {
  throw new Error(`${name}: ${detail}`)
}

/**
 * Validate a ladder: non-negative integers, escalating, with `0` (every message)
 * only as the final rung.
 * @param ladder - the requested rungs.
 * @returns the rungs, unchanged.
 */
function validateLadder(ladder: readonly number[]): readonly number[] {
  if (ladder.length === 0) invalid('ladder must hold at least one rung')
  let previous = -1
  ladder.forEach((rung, index) => {
    if (!Number.isInteger(rung) || rung < 0) invalid(`ladder[${index}] must be a non-negative integer, saw ${rung}`)
    if (rung === 0) {
      if (index !== ladder.length - 1) invalid('a ladder rung of 0 means every message and may only be the last rung')
      return
    }
    if (rung <= previous) invalid(`ladder must escalate: ladder[${index}] is ${rung} after ${previous}`)
    previous = rung
  })
  return ladder
}

/**
 * Fill every default so the transform can be driven without Cordis, and reject a
 * configuration the runtime schema cannot express.
 * @param config - mount configuration.
 * @returns the resolved configuration.
 */
export function resolveConfig(config: Config = {}): ResolvedConfig {
  const mode = config.mode ?? 'auto'
  if (!MODES.includes(mode)) invalid(`unknown mode "${String(mode)}"`)
  const statuses = config.statuses ?? [400]
  statuses.forEach((status) => {
    if (!Number.isInteger(status) || status < 100 || status > 599) invalid(`statuses must hold HTTP statuses, saw ${status}`)
  })
  const maxChars = config.maxChars ?? 200000
  if (!Number.isInteger(maxChars) || maxChars < 0) invalid(`maxChars must be a non-negative integer, saw ${maxChars}`)
  const source = config.matchPattern ?? 'content[_ ]?exists[_ ]?risk'
  let matchPattern: RegExp
  try {
    matchPattern = new RegExp(source, 'i')
  } catch (error) {
    invalid(`matchPattern is not a regular expression: ${String(error)}`)
  }
  const warnLimit = config.warnLimit ?? 5
  if (!Number.isInteger(warnLimit) || warnLimit < 0) invalid(`warnLimit must be a non-negative integer, saw ${warnLimit}`)
  return {
    mode,
    ladder: validateLadder(config.ladder ?? [4, 32, 0]),
    roles: config.roles ?? ['user'],
    includeAuxiliary: config.includeAuxiliary ?? false,
    maxChars,
    matchPattern,
    matchCodes: config.matchCodes ?? [],
    statuses,
    warnLimit,
  }
}

/**
 * Whether one code point is already a space, so nothing is inserted beside it.
 * `\s` covers the ASCII and Unicode whitespace classes; a zero-width space is
 * not in `\s` but separates in the same way, so it counts.
 * @param point - a single code point.
 * @returns whether the point separates.
 */
export function isSpace(point: string): boolean {
  return point === '\u200b' || /\s/u.test(point)
}

/**
 * Insert a space between every pair of adjacent characters when neither is a
 * space — the transform this plugin exists for.
 *
 * Iterates code points, not UTF-16 units, so a surrogate pair (an emoji, a flag
 * symbol, a CJK extension character) is never split; and never inserts ahead of
 * a combining mark or a zero-width joiner, so a character keeps what it renders
 * with (see {@link ATTACHED}).
 *
 * Idempotent: the output of one call carries a space in every position a second
 * call would insert one, so `spaceOut(spaceOut(text)) === spaceOut(text)`. The
 * re-dispatch guard relies on the same property through object identity rather
 * than through this one, but the property is what keeps a double application
 * harmless.
 *
 * @param text - the text to space out.
 * @returns the spaced text, or the input unchanged when there is nothing to do.
 */
export function spaceOut(text: string): string {
  const points = Array.from(text)
  const first = points[0]
  if (first === undefined || points.length < 2) return text
  let out = first
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1]
    const current = points[index]
    if (previous === undefined || current === undefined) break
    if (!isSpace(previous) && !isSpace(current) && !ATTACHED.test(current)) out += ' '
    out += current
  }
  return out
}

/** What one rewrite pass did. */
export interface RewriteOutcome {
  /** The rewritten message list; untouched messages keep their identity. */
  readonly messages: Message[]
  /** Messages actually rebuilt — the ladder's unit of progress. */
  readonly messagesRewritten: number
  /** Blocks whose text changed. */
  readonly blocksRewritten: number
  /** Characters added across the whole request. */
  readonly charsAdded: number
  /** Whether the character budget stopped the pass before the rung was covered. */
  readonly budgetExhausted: boolean
}

/** Mutable accounting for one pass. */
interface PassState {
  readonly maxChars: number
  blocks: number
  chars: number
  exhausted: boolean
}

/**
 * Read the model-visible text of a block, if it has any.
 *
 * The vocabulary's convention is a `text` string (`text` and `reasoning` blocks
 * both use it), which is also why this is generic rather than a type list: a
 * block type a plugin adds keeps working. `tool-call` carries `arguments` and
 * `image`/`file` carry a reference, so all three fall out untouched, as does any
 * unknown block without a `text` field.
 *
 * @param block - one content block.
 * @returns the text, or `undefined` when the block carries none.
 */
function textOf(block: ContentBlock): string | undefined {
  const candidate = (block as { text?: unknown }).text
  return typeof candidate === 'string' ? candidate : undefined
}

/**
 * Rewrite one block list.
 * @param blocks - the blocks to walk.
 * @param state - character accounting for the current pass.
 * @param depth - current `tool-result` nesting depth.
 * @returns rebuilt blocks, or `undefined` when nothing changed.
 */
function rewriteBlocks(blocks: readonly ContentBlock[], state: PassState, depth: number): ContentBlock[] | undefined {
  let changed = false
  const next: ContentBlock[] = []
  for (const block of blocks) {
    if (block.type === 'tool-result' && depth < MAX_DEPTH) {
      const nested = rewriteBlocks(block.content, state, depth + 1)
      if (nested === undefined) {
        next.push(block)
        continue
      }
      changed = true
      next.push({ ...block, content: nested })
      continue
    }
    const text = textOf(block)
    if (text === undefined) {
      next.push(block)
      continue
    }
    const rewritten = spaceOut(text)
    if (rewritten === text) {
      next.push(block)
      continue
    }
    changed = true
    state.blocks += 1
    state.chars += rewritten.length - text.length
    next.push({ ...block, text: rewritten } as ContentBlock)
  }
  return changed ? next : undefined
}

/**
 * Rewrite the newest `newest` eligible messages of a request.
 *
 * Walks the message list from the end, so the budget is spent where the trigger
 * is most likely to be, and counts *rewritten* messages rather than visited ones:
 * a message with nothing to space out (an image, a tool call, already-spaced
 * text) does not consume the rung.
 *
 * @param messages - the request's message list.
 * @param config - resolved configuration.
 * @param newest - how many of the newest eligible messages to rewrite; `0` means every message.
 * @returns the outcome, or `undefined` when this transform can change nothing.
 */
export function rewriteMessages(
  messages: readonly Message[],
  config: ResolvedConfig,
  newest: number,
): RewriteOutcome | undefined {
  const state: PassState = { maxChars: config.maxChars, blocks: 0, chars: 0, exhausted: false }
  const next = [...messages]
  let rewritten = 0
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (newest !== 0 && rewritten >= newest) break
    if (state.maxChars > 0 && state.chars >= state.maxChars) {
      state.exhausted = true
      break
    }
    const message = messages[index]
    if (message === undefined) continue
    if (!config.roles.includes(message.role)) continue
    const content = rewriteBlocks(message.content, state, 0)
    if (content === undefined) continue
    next[index] = { ...message, content }
    rewritten += 1
  }
  if (rewritten === 0) return undefined
  return {
    messages: next,
    messagesRewritten: rewritten,
    blocksRewritten: state.blocks,
    charsAdded: state.chars,
    budgetExhausted: state.exhausted,
  }
}

/**
 * Whether one terminal failure is the provider's content-moderation rejection.
 *
 * Gated three ways so an unrelated failure is never retried: the status must be
 * one of `statuses` (a failure that reports no status passes, because a gateway
 * may not forward one), and either one of `matchCodes` matches or `matchPattern`
 * matches the failure message. The default pattern is the provider's own
 * wording, which is why the default `matchCodes` is empty — `INVALID_REQUEST` is
 * the generic 400 code, not a moderation signal.
 *
 * @param failure - the terminal failure, if the stream ended in one.
 * @param config - resolved configuration.
 * @returns whether the request was rejected by content moderation.
 */
export function isContentRejection(failure: LlmFailure | undefined, config: ResolvedConfig): boolean {
  if (failure === undefined) return false
  if (failure.status !== undefined && !config.statuses.includes(failure.status)) return false
  if (config.matchCodes.includes(failure.code)) return true
  return config.matchPattern.test(failure.message)
}

/**
 * Render one outcome as a single bounded line.
 * @param outcome - what a pass did.
 * @param newest - the rung that produced it (`0` = every message).
 * @returns a human-readable summary.
 */
export function describeOutcome(outcome: RewriteOutcome, newest: number): string {
  const scope = newest === 0 ? 'every message' : `the newest ${String(newest)} message(s)`
  const budget = outcome.budgetExhausted ? ', budget exhausted' : ''
  return `${String(outcome.messagesRewritten)} message(s) rewritten across ${scope}`
    + ` (${String(outcome.blocksRewritten)} block(s), +${String(outcome.charsAdded)} chars${budget})`
}

/**
 * Register the rescue on the `llm/stream` waterfall.
 *
 * @param ctx - Cordis context carrying the `llm` service.
 * @param config - mount configuration.
 */

export function apply(ctx: Context, config: Config = {}): void {
  const resolved = resolveConfig(config)
  // Requests this plugin dispatched itself. Membership is what stops the guard
  // from re-entering: a rewritten request is registered before it is sent, so the
  // listener recognizes it on the way back through the waterfall.
  const dispatched = new WeakSet<GenerateOptions>()
  // Session id -> index of the first ladder rung worth trying. Written once the
  // provider has told us which rung it accepts, read to skip a probe already
  // known to fail. Bounded and in-process (see the module note).
  const latches = new Map<string, number>()
  let warned = 0

  const report = (line: string): void => {
    if (resolved.warnLimit <= 0 || warned >= resolved.warnLimit) return
    warned += 1
    ctx.logger.warn(`${name}: ${line}`)
  }

  const sessionKey = (request: GenerateOptions): string | undefined =>
    request.sessionId === undefined ? undefined : String(request.sessionId)

  const remember = (request: GenerateOptions, rung: number): void => {
    const key = sessionKey(request)
    if (key === undefined) return
    latches.delete(key)
    latches.set(key, rung)
    while (latches.size > MAX_LATCH_ENTRIES) {
      const oldest = latches.keys().next()
      if (oldest.done === true) break
      latches.delete(oldest.value)
    }
  }

  /** One rung of a request's plan, with the ladder index it came from. */
  interface Rung {
    readonly newest: number
    readonly rung: number
    readonly outcome: RewriteOutcome
  }

  /** How one dispatch ended. */
  type Attempt = { readonly kind: 'accepted' } | { readonly kind: 'rejected'; readonly frame: StreamChunk; readonly failure: LlmFailure }

  /**
   * Consume one dispatch, holding back the frame that ends a rejected attempt.
   *
   * Every chunk is yielded as it arrives except that frame, which is returned
   * instead. A retried attempt produced no visible output — a moderation
   * rejection is refused before the provider streams anything — so nothing
   * already yielded is at risk, and a discarded attempt's failure is never shown.
   * `produced` guards the assumption rather than trusting it: a stream that
   * emitted anything at all is never retried.
   */
  async function* attempt(inner: AsyncIterable<StreamChunk>): AsyncGenerator<StreamChunk, Attempt> {
    let produced = false
    for await (const chunk of inner) {
      if (chunk.type === 'finish') {
        if (!produced && chunk.reason.kind === 'error' && isContentRejection(chunk.reason.failure, resolved)) {
          return { kind: 'rejected', frame: chunk, failure: chunk.reason.failure }
        }
        yield chunk
        return { kind: 'accepted' }
      }
      produced = true
      yield chunk
    }
    return { kind: 'accepted' }
  }

  async function* rescue(
    request: GenerateOptions,
    identity: () => AsyncIterable<StreamChunk>,
  ): AsyncGenerator<StreamChunk> {
    const key = sessionKey(request)
    const latched = key === undefined ? undefined : latches.get(key)
    // Where to start: a session that already found its rung skips both the probe
    // and the rungs that were proven insufficient.
    const start = latched === undefined ? 0 : Math.min(latched, resolved.ladder.length - 1)
    const levels = latched === undefined ? resolved.ladder : resolved.ladder.slice(start)

    // `undefined` is the untouched probe. A rung that would rewrite nothing is
    // dropped rather than dispatched — it would only repeat the request the
    // previous rung already sent.
    const plan: (Rung | undefined)[] = []
    if (resolved.mode === 'auto' && latched === undefined) plan.push(undefined)
    levels.forEach((newest, offset) => {
      const outcome = rewriteMessages(request.messages, resolved, newest)
      if (outcome !== undefined) plan.push({ newest, rung: start + offset, outcome })
    })
    // Nothing this transform can reach: behave like an unmounted plugin rather
    // than emitting a stream with no frame in it.
    if (plan.length === 0) {
      yield* identity()
      return
    }

    if (resolved.mode === 'observe') {
      if (plan.length === 0 || plan.every(entry => entry === undefined)) {
        report('nothing this transform can rewrite; request dispatched unchanged')
      }
      for (const entry of plan) {
        if (entry !== undefined) report(`${describeOutcome(entry.outcome, entry.newest)} — mode observe, request dispatched unchanged`)
      }
      yield* identity()
      return
    }

    const dispatch = (rung: Rung): AsyncIterable<StreamChunk> => {
      const replacement: GenerateOptions = { ...request, messages: rung.outcome.messages }
      dispatched.add(replacement)
      return ctx.llm.stream(replacement)
    }

    let held: { readonly frame: StreamChunk; readonly failure: LlmFailure } | undefined
    for (let index = 0; index < plan.length; index += 1) {
      const entry = plan[index]
      const isLast = index === plan.length - 1
      // The previous attempt produced no visible output, so retrying here loses
      // nothing — and it is the only way to learn whether a rung is enough.
      const outcome = yield* attempt(entry === undefined ? identity() : dispatch(entry))
      if (outcome.kind === 'accepted') {
        if (entry !== undefined) {
          remember(request, entry.rung)
          report(`${describeOutcome(entry.outcome, entry.newest)} — accepted; this session now starts at that rung`)
        }
        return
      }
      if (isLast) {
        yield outcome.frame
        return
      }
      held = outcome
      report(entry === undefined
        ? `request rejected (HTTP ${String(outcome.failure.status ?? '?')}, ${outcome.failure.code}: ${outcome.failure.message}); escalating`
        : `${describeOutcome(entry.outcome, entry.newest)} — still rejected; escalating`)
    }

    // Every rung was tried and each was refused. Surface the provider's own
    // terminal frame, unchanged: this plugin never turns a failure into a
    // success it did not observe.
    if (held !== undefined) {
      remember(request, resolved.ladder.length - 1)
      report('every rung was rejected; the trigger is outside what this mount rewrites (see the module note for what is out of scope)')
      yield held.frame
    }
  }

  ctx.on('llm/stream', (options: GenerateOptions, next: () => AsyncIterable<StreamChunk>): AsyncIterable<StreamChunk> => {
    // Our own replacement on its way back through the waterfall, or a request
    // this mount is configured to leave alone.
    if (dispatched.has(options)) return next()
    if (resolved.mode === 'off') return next()
    if (!resolved.includeAuxiliary && options.purpose !== undefined) return next()
    return rescue(options, next)
  }, { prepend: true })
}
