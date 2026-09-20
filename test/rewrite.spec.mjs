/**
 * Message-level rewriting: which messages a rung reaches, what it leaves alone,
 * and how the character budget bounds it.
 *
 * Every assertion here is about scope rather than about spacing (which
 * `space-out.spec.mjs` owns), because scope is the only thing that keeps this
 * plugin affordable: a rung that rewrites more than it must is a rung that costs
 * the operator tokens for nothing.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { describeOutcome, isContentRejection, resolveConfig, rewriteMessages } from '../lib/index.js'

const message = (role, content, id = `${role}-${content.length}`) => ({
  id,
  role,
  content,
  source: role === 'assistant' ? { kind: 'model', provider: 'probe', model: 'm' } : { kind: 'plugin', plugin: 'test' },
})

const user = (text, id) => message('user', [{ type: 'text', text }], id)
const assistant = (text, id) => message('assistant', [{ type: 'text', text }], id)

test('the newest messages are rewritten, the older ones are not', () => {
  const messages = [user('older', 'a'), user('middle', 'b'), user('newest', 'c')]
  const outcome = rewriteMessages(messages, resolveConfig(), 1)
  assert.equal(outcome.messagesRewritten, 1)
  assert.equal(outcome.messages[2].content[0].text, 'n e w e s t')
  assert.equal(outcome.messages[0], messages[0], 'an untouched message keeps its identity')
  assert.equal(outcome.messages[1], messages[1])
})

test('a rung counts rewritten messages, so a message it cannot touch does not consume it', () => {
  const messages = [
    user('older', 'a'),
    { id: 'img', role: 'user', content: [{ type: 'image', attachment: { id: 'x' } }], source: { kind: 'plugin', plugin: 't' } },
    user('newest', 'c'),
  ]
  const outcome = rewriteMessages(messages, resolveConfig(), 2)
  assert.equal(outcome.messagesRewritten, 2, 'the image-only message costs nothing')
  assert.equal(outcome.messages[0].content[0].text, 'o l d e r')
  assert.equal(outcome.messages[1], messages[1], 'and it is returned unchanged')
})

test('roles scope the rewrite, and user is the default', () => {
  const messages = [assistant('spoken', 'a'), user('typed', 'b')]
  const scoped = rewriteMessages(messages, resolveConfig(), 0)
  assert.equal(scoped.messagesRewritten, 1)
  assert.equal(scoped.messages[0], messages[0], 'assistant content is out of default scope')
  assert.equal(scoped.messages[1].content[0].text, 't y p e d')

  const widened = rewriteMessages(messages, resolveConfig({ roles: ['user', 'assistant'] }), 0)
  assert.equal(widened.messagesRewritten, 2)
  assert.equal(widened.messages[0].content[0].text, 's p o k e n')
})

test('nested tool-result content is rewritten, because that is where fetched content lands', () => {
  const messages = [message('user', [{
    type: 'tool-result',
    toolCallId: 'call-1',
    content: [{ type: 'text', text: 'page body' }],
  }])]
  const outcome = rewriteMessages(messages, resolveConfig(), 0)
  assert.equal(outcome.blocksRewritten, 1)
  assert.equal(outcome.messages[0].content[0].type, 'tool-result')
  assert.equal(outcome.messages[0].content[0].toolCallId, 'call-1', 'the container keeps its identity fields')
  assert.equal(outcome.messages[0].content[0].content[0].text, 'p a g e b o d y')
})

test('tool-call arguments are left exactly as the model wrote them', () => {
  const raw = '{"path":"/tmp/a b","content":"hello"}'
  const messages = [message('user', [{ type: 'tool-call', id: 'call-1', name: 'write', arguments: raw }])]
  assert.equal(rewriteMessages(messages, resolveConfig(), 0), undefined, 'a tool call alone is nothing this transform can change')
})

test('blocks without model-visible text are untouched', () => {
  const messages = [message('user', [
    { type: 'image', attachment: { id: 'img' } },
    { type: 'file', attachment: { id: 'file' } },
  ])]
  assert.equal(rewriteMessages(messages, resolveConfig(), 0), undefined)
})

test('a block type this plugin has never seen is rewritten when it carries text', () => {
  const messages = [message('user', [{ type: 'plugin-block', text: 'custom' }])]
  const outcome = rewriteMessages(messages, resolveConfig(), 0)
  assert.equal(outcome.messages[0].content[0].text, 'c u s t o m')
  assert.equal(outcome.messages[0].content[0].type, 'plugin-block', 'the block tag survives')
})

test('text that is already spaced out is not a change', () => {
  const messages = [user('a b c', 'a')]
  assert.equal(rewriteMessages(messages, resolveConfig(), 0), undefined)
})

test('a rewritten message keeps its id, role and source', () => {
  const original = user('text', 'id-1')
  const outcome = rewriteMessages([original], resolveConfig(), 0)
  const rebuilt = outcome.messages[0]
  assert.equal(rebuilt.id, 'id-1')
  assert.equal(rebuilt.role, 'user')
  assert.deepEqual(rebuilt.source, original.source)
  assert.notEqual(rebuilt, original)
})

test('the character budget stops a pass and says so', () => {
  const messages = [user('aaaaa', 'a'), user('bbbbb', 'b'), user('ccccc', 'c')]
  const outcome = rewriteMessages(messages, resolveConfig({ maxChars: 8 }), 0)
  assert.equal(outcome.budgetExhausted, true)
  assert.equal(outcome.charsAdded, 8, 'each rewritten message adds four characters')
  assert.equal(outcome.messagesRewritten, 2, 'the check runs before each message, so the budget stops it on the boundary')
})

test('a budget of zero means no budget', () => {
  const messages = [user('aaaaa', 'a'), user('bbbbb', 'b')]
  const outcome = rewriteMessages(messages, resolveConfig({ maxChars: 0 }), 0)
  assert.equal(outcome.budgetExhausted, false)
  assert.equal(outcome.messagesRewritten, 2)
})

test('a rung of zero rewrites every eligible message', () => {
  const messages = [user('aa', 'a'), assistant('bb', 'b'), user('cc', 'c')]
  const outcome = rewriteMessages(messages, resolveConfig(), 0)
  assert.equal(outcome.messagesRewritten, 2, 'both user messages, not the assistant one')
  assert.equal(outcome.messages[0].content[0].text, 'a a')
})

test('describeOutcome names the scope, the cost and a spent budget', () => {
  const outcome = rewriteMessages([user('abcd', 'a')], resolveConfig(), 4)
  assert.match(describeOutcome(outcome, 4), /^1 message\(s\) rewritten across the newest 4 message\(s\) \(1 block\(s\), \+3 chars\)$/)
  assert.match(describeOutcome(outcome, 0), /across every message/)
  assert.match(describeOutcome({ ...outcome, budgetExhausted: true }, 4), /budget exhausted/)
})

test('resolveConfig materializes the documented defaults', () => {
  const resolved = resolveConfig()
  assert.equal(resolved.mode, 'auto')
  assert.deepEqual(resolved.ladder, [4, 32, 0])
  assert.deepEqual(resolved.roles, ['user'])
  assert.equal(resolved.includeAuxiliary, false)
  assert.equal(resolved.maxChars, 200000)
  assert.deepEqual(resolved.statuses, [400])
  assert.deepEqual(resolved.matchCodes, [])
  assert.equal(resolved.warnLimit, 5)
  assert.equal(resolved.matchPattern.test('Content Exists Risk'), true)
  assert.equal(resolved.matchPattern.test('CONTENT EXISTS RISK'), true, 'matching is case-insensitive')
})

test('resolveConfig rejects a ladder that does not escalate', () => {
  assert.throws(() => resolveConfig({ ladder: [] }), /at least one rung/)
  assert.throws(() => resolveConfig({ ladder: [4, 4] }), /must escalate/)
  assert.throws(() => resolveConfig({ ladder: [8, 4] }), /must escalate/)
  assert.throws(() => resolveConfig({ ladder: [0, 4] }), /only be the last rung/)
  assert.throws(() => resolveConfig({ ladder: [-1] }), /non-negative integer/)
  assert.throws(() => resolveConfig({ ladder: [1.5] }), /non-negative integer/)
  assert.deepEqual(resolveConfig({ ladder: [1, 0] }).ladder, [1, 0])
})

test('resolveConfig rejects configuration the schema cannot express', () => {
  assert.throws(() => resolveConfig({ mode: 'sometimes' }), /unknown mode/)
  assert.throws(() => resolveConfig({ matchPattern: '(' }), /not a regular expression/)
  assert.throws(() => resolveConfig({ statuses: [99] }), /HTTP statuses/)
  assert.throws(() => resolveConfig({ maxChars: -1 }), /non-negative integer/)
  assert.throws(() => resolveConfig({ warnLimit: 1.5 }), /non-negative integer/)
})

test('isContentRejection recognizes the provider wording and nothing adjacent', () => {
  const resolved = resolveConfig()
  assert.equal(isContentRejection({ message: 'Content Exists Risk', code: 'INVALID_REQUEST', status: 400 }, resolved), true)
  assert.equal(isContentRejection({ message: 'Content Exists Risk', code: 'INVALID_REQUEST' }, resolved), true, 'a gateway that forwards no status still matches')
  assert.equal(isContentRejection({ message: 'Content Exists Risk', code: 'AUTH', status: 401 }, resolved), false, 'the status gate holds')
  assert.equal(isContentRejection({ message: 'DeepSeek API error (HTTP 400)', code: 'INVALID_REQUEST', status: 400 }, resolved), false, 'a generic 400 is not moderation')
  assert.equal(isContentRejection({ message: 'context length exceeded', code: 'CONTEXT_WINDOW_EXCEEDED', status: 400 }, resolved), false)
  assert.equal(isContentRejection(undefined, resolved), false)
})

test('the generic code is opt-in, for a gateway that mangles the message', () => {
  const resolved = resolveConfig({ matchCodes: ['INVALID_REQUEST'] })
  assert.equal(isContentRejection({ message: 'unrecognizable', code: 'INVALID_REQUEST', status: 400 }, resolved), true)
  assert.equal(isContentRejection({ message: 'unrecognizable', code: 'SERVER', status: 500 }, resolved), false, 'the status gate still applies')
})
