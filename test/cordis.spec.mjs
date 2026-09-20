/**
 * Wiring tests against a real Cordis context, a real `llm/stream` waterfall and a
 * real `dsh-llm` runtime with a stub provider.
 *
 * The stub provider is the point: it *refuses* a request whose payload still
 * contains a trigger sequence, exactly as moderation does, and it refuses it the
 * way a 400 arrives — as a terminal `error` finish produced before any content.
 * Everything the plugin claims rests on that shape:
 *
 *  - a rejected attempt is retried, and the retried attempt's failure never
 *    reaches the consumer (the held-back terminal frame);
 *  - a rung that is not enough is visible, and the next rung reaches further;
 *  - the rung that works is remembered, so the next request starts there;
 *  - when no rung is enough, the provider's own failure is surfaced unchanged;
 *  - a request that is never rejected is dispatched as the identical object, with
 *    no copy and no extra adapter call.
 *
 * The unmounted control arm is what makes the rest evidence: without the plugin
 * the same request fails.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { LlmAdapter, createUserMessage } from '@deepseek-ai/dsh-llm'
import * as plugin from '../lib/index.js'

/** A character sequence the stub provider refuses to receive. */
const TRIGGER = 'FLAGTRIGGER'
const OK_CHUNKS = [{ type: 'text-delta', index: 0, text: 'ok' }, { type: 'finish', reason: { kind: 'stop' } }]
const REJECTION = {
  type: 'finish',
  reason: {
    kind: 'error',
    failure: { message: 'Content Exists Risk', code: 'INVALID_REQUEST', status: 400 },
  },
}

/**
 * A provider that moderates: it refuses the whole request while the trigger
 * sequence survives anywhere in what it was asked to read, and streams normally
 * otherwise. A rejection yields no content, which is what makes a retry safe.
 */
class ModeratingAdapter extends LlmAdapter {
  seen = []

  async * stream(options) {
    this.seen.push(options)
    if (JSON.stringify(options.messages).includes(TRIGGER)) {
      yield REJECTION
      return
    }
    yield* OK_CHUNKS
  }
}

const userMessage = (text, id) => createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'plugin', plugin: 'test' } })
const assistantMessage = (text, id = 'assistant') => ({
  id,
  role: 'assistant',
  content: [{ type: 'text', text }],
  source: { kind: 'model', provider: 'probe', model: 'm' },
})

const deepFreeze = (value) => {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value)
    for (const nested of Object.values(value)) deepFreeze(nested)
  }
  return value
}

/**
 * Mount the runtime, an observation listener, the plugin and the stub adapter.
 * @param config - mount configuration.
 * @param options - `{ mounted: false }` leaves the plugin out (the control arm),
 *   `{ provider }` overrides the route name.
 */
async function mount(config = {}, options = {}) {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  const warnings = []
  // `levels.default` must admit WARN: cordis drops a message below the
  // exporter's level and the root logger defaults to INFO.
  ctx.logger.exporter({
    levels: { default: 3 },
    export: (message) => { if (message.type === 'warn') warnings.push(message.args.map(String).join(' ')) },
  })
  const observed = []
  // Registered before the plugin, so it sits inside the plugin's position in the
  // chain and observes exactly the request that is finally dispatched.
  ctx.on('llm/stream', (dispatched, next) => {
    observed.push(dispatched)
    return next()
  })
  if (options.mounted !== false) await ctx.plugin({ name: plugin.name, inject: plugin.inject, apply: plugin.apply }, config)
  const adapter = new ModeratingAdapter()
  ctx.llm.registerAdapter([options.provider ?? 'probe'], adapter)
  return { ctx, adapter, observed, warnings }
}

const collect = async (stream) => {
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

const isError = (chunk) => chunk?.type === 'finish' && chunk.reason.kind === 'error'

test('unmounted, the same request fails — the defect is real', async () => {
  const { ctx, adapter } = await mount({}, { mounted: false })
  const request = { provider: 'probe', model: 'm', messages: [userMessage(`page body ${TRIGGER}`, 'old')] }
  const chunks = await collect(ctx.llm.stream(request))
  assert.equal(adapter.seen.length, 1)
  assert.equal(isError(chunks.at(-1)), true)
  assert.equal(chunks.at(-1).reason.failure.message, 'Content Exists Risk')
})

test('a poisoned session is rescued, and the refused attempts are never shown', async () => {
  const { ctx, adapter, observed } = await mount({ ladder: [1, 0] })
  const request = {
    provider: 'probe',
    model: 'm',
    sessionId: 'sess-1',
    messages: [userMessage(`page body ${TRIGGER}`, 'old'), userMessage('newest question', 'new')],
  }
  const chunks = await collect(ctx.llm.stream(request))

  assert.equal(adapter.seen.length, 3, 'probe, then one rung that was not enough, then one that was')
  // The probe and the first rung were refused; only the accepted rung's chunks
  // reach the consumer, and the consumer never sees a failure at all.
  assert.deepEqual(chunks, OK_CHUNKS)
  assert.equal(chunks.some(isError), false, 'a refused attempt\'s terminal frame is held back')

  assert.equal(adapter.seen[0].messages[0], request.messages[0], 'the probe dispatched the caller\'s own objects')
  assert.ok(adapter.seen[1].messages[0].content[0].text.includes(TRIGGER), 'rung 1 rewrote only the newest message')
  assert.equal(adapter.seen[2].messages[0].content[0].text.includes(TRIGGER), false, 'rung 2 reached the old message')
  assert.equal(adapter.seen[2].messages[1].content[0].text, 'n e w e s t q u e s t i o n')

  assert.equal(request.messages[0].content[0].text.endsWith(TRIGGER), true, 'the caller\'s request is untouched')
  assert.equal(observed.length, 3, 'other listeners see one dispatch per attempt, and exactly one per request sent')
})

test('the rung that worked is remembered, so the next request does not re-probe', async () => {
  const { ctx, adapter } = await mount({ ladder: [1, 0] })
  const build = () => ({
    provider: 'probe',
    model: 'm',
    sessionId: 'sess-2',
    messages: [userMessage(`page body ${TRIGGER}`, 'old'), userMessage('next question', 'new')],
  })
  await collect(ctx.llm.stream(build()))
  const callsAfterFirst = adapter.seen.length
  await collect(ctx.llm.stream(build()))
  assert.equal(callsAfterFirst, 3)
  assert.equal(adapter.seen.length, 4, 'the second request paid one adapter call, not three')
  assert.equal(adapter.seen[3].messages[0].content[0].text.includes(TRIGGER), false, 'and started at the rung known to work')
})

test('a rung that cannot reach the trigger does not retry forever', async () => {
  // The trigger sits in assistant content, which the default scope leaves alone.
  const { ctx, adapter } = await mount({ ladder: [1, 0] })
  const request = {
    provider: 'probe',
    model: 'm',
    messages: [assistantMessage(`it said ${TRIGGER}`), userMessage('question', 'new')],
  }
  const chunks = await collect(ctx.llm.stream(request))
  assert.equal(adapter.seen.length, 3, 'the probe and both rungs, then it stops')
  assert.equal(isError(chunks.at(-1)), true, 'the provider\'s own failure reaches the consumer')
  assert.equal(chunks.at(-1).reason.failure.message, 'Content Exists Risk')
  assert.equal(chunks.length, 1, 'and it is the only thing the consumer receives')
})

test('widening the scope reaches a trigger only the assistant side carries', async () => {
  const { ctx, adapter } = await mount({ ladder: [0], roles: ['user', 'assistant'] })
  const request = {
    provider: 'probe',
    model: 'm',
    messages: [assistantMessage(`it said ${TRIGGER}`), userMessage('question', 'new')],
  }
  const chunks = await collect(ctx.llm.stream(request))
  assert.deepEqual(chunks, OK_CHUNKS)
  assert.equal(adapter.seen.length, 2, 'probe, then the widened rung')
  assert.equal(adapter.seen[1].messages[0].content[0].text.includes(TRIGGER), false)
})

test('a healthy request takes the identity path: same object, one call, no copy', async () => {
  const { ctx, adapter, observed } = await mount()
  const request = { provider: 'probe', model: 'm', messages: [userMessage('an ordinary question', 'a')] }
  const chunks = await collect(ctx.llm.stream(request))
  assert.deepEqual(chunks, OK_CHUNKS)
  assert.equal(adapter.seen.length, 1)
  assert.equal(adapter.seen[0], request, 'the adapter receives the very same request object')
  assert.equal(observed[0], request)
})

test('a differently worded 400 is not treated as moderation', async () => {
  const { ctx, adapter } = await mount()
  const ctxRejection = {
    type: 'finish',
    reason: { kind: 'error', failure: { message: 'Bad request: malformed tool schema', code: 'INVALID_REQUEST', status: 400 } },
  }
  const target = new class extends LlmAdapter {
    seen = []
    async * stream(options) { this.seen.push(options); yield ctxRejection }
  }()
  ctx.llm.registerAdapter(['other'], target)
  const chunks = await collect(ctx.llm.stream({ provider: 'other', model: 'm', messages: [userMessage('hello', 'a')] }))
  assert.equal(target.seen.length, 1, 'no retry was attempted')
  assert.deepEqual(chunks, [ctxRejection])
  assert.equal(adapter.seen.length, 0)
})

test('an attempt that already streamed content is never retried', async () => {
  // The hold-back rule is a rule about evidence, not optimism: a rejection that
  // arrives *after* content was already handed to the consumer cannot be retried,
  // because the consumer would see the content twice. The plugin must surface it.
  const { ctx } = await mount({ ladder: [0] })
  const target = new class extends LlmAdapter {
    seen = []
    async * stream(options) {
      this.seen.push(options)
      yield { type: 'text-delta', index: 0, text: 'partial answer' }
      yield REJECTION
    }
  }()
  ctx.llm.registerAdapter(['midstream'], target)
  const chunks = await collect(ctx.llm.stream({
    provider: 'midstream',
    model: 'm',
    messages: [userMessage(`page ${TRIGGER}`, 'a')],
  }))
  assert.equal(target.seen.length, 1, 'a retry here would duplicate content the consumer already saw')
  assert.equal(chunks[0].text, 'partial answer')
  assert.equal(isError(chunks.at(-1)), true)
})

test('mode off and mode observe leave the request alone', async () => {
  for (const mode of ['off', 'observe']) {
    const { ctx, adapter, warnings } = await mount({ mode })
    const request = { provider: 'probe', model: 'm', messages: [userMessage(`page ${TRIGGER}`, 'a')] }
    const chunks = await collect(ctx.llm.stream(request))
    assert.equal(adapter.seen.length, 1, `${mode}: one dispatch`)
    assert.equal(adapter.seen[0], request, `${mode}: the caller's own object`)
    assert.equal(isError(chunks.at(-1)), true, `${mode}: the failure surfaces`)
    if (mode === 'observe') assert.match(warnings[0], /mode observe, request dispatched unchanged/)
    else assert.equal(warnings.length, 0)
  }
})

test('mode always rewrites from the first dispatch', async () => {
  const { ctx, adapter } = await mount({ mode: 'always', ladder: [0] })
  const request = { provider: 'probe', model: 'm', messages: [userMessage(`page ${TRIGGER}`, 'a')] }
  const chunks = await collect(ctx.llm.stream(request))
  assert.deepEqual(chunks, OK_CHUNKS)
  assert.equal(adapter.seen.length, 1, 'no probe was spent')
  assert.notEqual(adapter.seen[0], request)
  assert.equal(adapter.seen[0].messages[0].content[0].text.includes(TRIGGER), false)
})

test('an auxiliary call is left alone unless it is opted in', async () => {
  const { ctx, adapter } = await mount({ ladder: [0] })
  const request = { provider: 'probe', model: 'm', purpose: 'compaction', messages: [userMessage(`summary ${TRIGGER}`, 'a')] }
  const chunks = await collect(ctx.llm.stream(request))
  assert.equal(adapter.seen.length, 1)
  assert.equal(adapter.seen[0], request)
  assert.equal(isError(chunks.at(-1)), true)

  const opted = await mount({ ladder: [0], includeAuxiliary: true })
  const rewritten = await collect(opted.ctx.llm.stream({ ...request, messages: [userMessage(`summary ${TRIGGER}`, 'a')] }))
  assert.deepEqual(rewritten, OK_CHUNKS)
  assert.equal(opted.adapter.seen.length, 2)
})

test('a deep-frozen loop-style request is rewritten rather than mutated', async () => {
  const { ctx, adapter } = await mount({ ladder: [0] })
  const request = deepFreeze({
    provider: 'probe',
    model: 'm',
    messages: [deepFreeze(userMessage(`page ${TRIGGER}`, 'a'))],
  })
  const chunks = await collect(ctx.llm.stream(request))
  assert.deepEqual(chunks, OK_CHUNKS)
  assert.equal(adapter.seen.length, 2)
  assert.equal(request.messages[0].content[0].text.includes(TRIGGER), true, 'the frozen original still holds the trigger')
  assert.equal(adapter.seen[1].messages[0].content[0].text.includes(TRIGGER), false)
})

test('diagnostics name the rejection and the rung, bounded by warnLimit', async () => {
  const { ctx, warnings } = await mount({ ladder: [0], warnLimit: 2 })
  const build = () => ({ provider: 'probe', model: 'm', sessionId: 'sess-3', messages: [userMessage(`page ${TRIGGER}`, 'a')] })
  await collect(ctx.llm.stream(build()))
  assert.equal(warnings.length, 2, 'escalation and acceptance')
  assert.match(warnings[0], /request rejected \(HTTP 400, INVALID_REQUEST: Content Exists Risk\); escalating/)
  assert.match(warnings[1], /1 message\(s\) rewritten across every message .* — accepted; this session now starts at that rung/)
  await collect(ctx.llm.stream(build()))
  assert.equal(warnings.length, 2, 'the bound holds')
})

test('warnLimit 0 silences diagnostics without changing behaviour', async () => {
  const { ctx, adapter, warnings } = await mount({ ladder: [0], warnLimit: 0 })
  const chunks = await collect(ctx.llm.stream({ provider: 'probe', model: 'm', messages: [userMessage(`page ${TRIGGER}`, 'a')] }))
  assert.deepEqual(chunks, OK_CHUNKS)
  assert.equal(adapter.seen.length, 2, 'the rescue still ran')
  assert.equal(warnings.length, 0)
})
