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
 * - **Assistant and system content is opt-in.** `roles` defaults to
 *   `['user', 'tool']` — user input and tool results, the two places fetched
 *   content enters, and the shape most reported poisons arrive in. Tool results
 *   were user-role content through the 0.1.x lines and are first-class
 *   `role: 'tool'` messages from the 0.2.1 line on; the default names both so
 *   one mount covers either line. Model output is not where fetched content
 *   enters, spacing it costs the most tokens, and it invites the model to
 *   imitate spaced-out text. Add `'assistant'` only when the poison is known to
 *   sit there.
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
import z from '@deepseek-ai/schemastery';
import type { Context } from '@deepseek-ai/cordis';
import type { ContentBlock, LlmFailure } from '@deepseek-ai/dsh-llm';
/** How the plugin behaves when it sees a request. */
export type Mode = 'off' | 'observe' | 'auto' | 'always';
/** Message roles whose content this plugin may rewrite. */
export type ScopedRole = 'system' | 'user' | 'assistant' | 'tool';
/** Plugin configuration. Every field is optional; see {@link resolveConfig}. */
export interface Config {
    /**
     * `auto` (default) probes with the request untouched and rewrites only after a
     * rejection; `always` rewrites from the first dispatch of a request; `observe`
     * reports what would be rewritten and dispatches the request unchanged;
     * `off` leaves every request alone.
     */
    mode?: Mode;
    /**
     * Escalating rungs, in newest-message counts: the request's last N messages
     * are rewritten, `0` meaning every message and allowed only as the final rung.
     * A rung is tried only when the previous one was rejected. Default `[4, 32, 0]`.
     */
    ladder?: number[];
    /**
     * Roles whose content may be rewritten. Default `['user', 'tool']`: user input
     * and tool results are the two places fetched content enters, and they are two
     * distinct message roles from the line that made tool results first-class.
     */
    roles?: ScopedRole[];
    /** Whether `purpose`-carrying auxiliary calls are rewritten too. Default `false`. */
    includeAuxiliary?: boolean;
    /**
     * Stop rewriting a request once this many characters have been added.
     * Counted per request and checked before each message, so one message can
     * overshoot it. `0` disables the budget. Default `200000`.
     */
    maxChars?: number;
    /**
     * Regular expression (source, case-insensitive unless it says otherwise)
     * matched against the provider's failure message to recognize a
     * content-moderation rejection. Default `content[_ ]?exists[_ ]?risk`.
     */
    matchPattern?: string;
    /**
     * Extra failure codes treated as a content rejection. Empty by default: the
     * generic 400 code `INVALID_REQUEST` also covers malformed requests and
     * oversized bodies, so it is not a moderation signal on its own. Add it for a
     * gateway that mangles the provider's message.
     */
    matchCodes?: string[];
    /** HTTP statuses a rejection must carry, when the failure reports one. Default `[400]`. */
    statuses?: number[];
    /** Bound on emitted warnings per mount. `0` silences logging. Default `5`. */
    warnLimit?: number;
}
/** Configuration with every default materialized and validated. */
export interface ResolvedConfig {
    readonly mode: Mode;
    readonly ladder: readonly number[];
    readonly roles: readonly ScopedRole[];
    readonly includeAuxiliary: boolean;
    readonly maxChars: number;
    readonly matchPattern: RegExp;
    readonly matchCodes: readonly string[];
    readonly statuses: readonly number[];
    readonly warnLimit: number;
}
/** The plugin name used by the mount patch. */
export declare const name = "content-exists-risk";
/** This plugin dispatches through `ctx.llm.stream()` and injects its service. */
export declare const inject: string[];
/** Runtime schema for {@link Config}; cordis applies it before {@link apply}. */
export declare const Config: z<Config>;
/**
 * Fill every default so the transform can be driven without Cordis, and reject a
 * configuration the runtime schema cannot express.
 * @param config - mount configuration.
 * @returns the resolved configuration.
 */
export declare function resolveConfig(config?: Config): ResolvedConfig;
/**
 * Whether one code point is already a space, so nothing is inserted beside it.
 * `\s` covers the ASCII and Unicode whitespace classes; a zero-width space is
 * not in `\s` but separates in the same way, so it counts.
 * @param point - a single code point.
 * @returns whether the point separates.
 */
export declare function isSpace(point: string): boolean;
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
export declare function spaceOut(text: string): string;
/**
 * The part of a request message this walk reads and rebuilds.
 *
 * Named here rather than imported because the published element type of
 * `GenerateOptions.messages` differs across the lines this package supports:
 * `Message` through the 0.1.x lines, and `Message | RequestUserInput` from the
 * line that made tool results first-class. Both are structurally this, and one
 * source has to compile against every line the package claims.
 */
export interface WalkableMessage {
    readonly role: string;
    readonly content: readonly ContentBlock[];
}
/** What one rewrite pass did. */
export interface RewriteOutcome<M extends WalkableMessage = WalkableMessage> {
    /** The rewritten message list; untouched messages keep their identity. */
    readonly messages: M[];
    /** Messages actually rebuilt — the ladder's unit of progress. */
    readonly messagesRewritten: number;
    /** Blocks whose text changed. */
    readonly blocksRewritten: number;
    /** Characters added across the whole request. */
    readonly charsAdded: number;
    /** Whether the character budget stopped the pass before the rung was covered. */
    readonly budgetExhausted: boolean;
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
export declare function rewriteMessages<M extends WalkableMessage>(messages: readonly M[], config: ResolvedConfig, newest: number): RewriteOutcome<M> | undefined;
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
export declare function isContentRejection(failure: LlmFailure | undefined, config: ResolvedConfig): boolean;
/**
 * Render one outcome as a single bounded line.
 * @param outcome - what a pass did.
 * @param newest - the rung that produced it (`0` = every message).
 * @returns a human-readable summary.
 */
export declare function describeOutcome(outcome: RewriteOutcome, newest: number): string;
/**
 * Register the rescue on the `llm/stream` waterfall.
 *
 * @param ctx - Cordis context carrying the `llm` service.
 * @param config - mount configuration.
 */
export declare function apply(ctx: Context, config?: Config): void;
