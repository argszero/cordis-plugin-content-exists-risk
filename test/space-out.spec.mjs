/**
 * The transform itself, as a pure function.
 *
 * These tests are the contract the rest of the plugin leans on: code-point
 * iteration (never split a surrogate pair), attachment handling (never detach a
 * combining mark or break a ZWJ sequence), whitespace as a separator, and
 * idempotency — the property that makes a second application harmless.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { isSpace, spaceOut } from '../lib/index.js'

test('every pair of adjacent non-space characters gains a space', () => {
  assert.equal(spaceOut('hello'), 'h e l l o')
  assert.equal(spaceOut('ab'), 'a b')
  assert.equal(spaceOut('a1!'), 'a 1 !')
})

test('nothing is inserted when either side is already a space', () => {
  assert.equal(spaceOut('a b c'), 'a b c')
  assert.equal(spaceOut('h e l l o'), 'h e l l o', 'already spaced text is a fixed point')
  assert.equal(spaceOut('a\nb'), 'a\nb', 'a newline separates just as a space does')
  assert.equal(spaceOut('a\tb'), 'a\tb')
  assert.equal(spaceOut('a\u200bb'), 'a\u200bb', 'a zero-width space separates too')
})

test('strings shorter than two code points are returned unchanged', () => {
  assert.equal(spaceOut(''), '')
  assert.equal(spaceOut('a'), 'a')
  assert.equal(spaceOut(' '), ' ')
  assert.equal(spaceOut('中'), '中')
})

test('CJK, digits and punctuation space out like any other character', () => {
  assert.equal(spaceOut('中文字'), '中 文 字')
  assert.equal(spaceOut('值=1'), '值 = 1')
})

test('a surrogate pair is never split', () => {
  // A UTF-16-unit walk would insert a space between the two halves of the emoji
  // and leave two lone surrogates behind.
  assert.equal(spaceOut('👍a'), '👍 a')
  assert.equal(spaceOut('a👍'), 'a 👍')
  assert.equal(spaceOut('🇦🇪'), '🇦 🇪', 'regional indicators are code points, not halves')
  for (const input of ['👍a', 'a👍', '🇦🇪', '👍👍']) {
    const output = spaceOut(input)
    assert.equal([...output].every(point => point.isWellFormed?.() ?? true), true, `${output} has no lone surrogate`)
    assert.ok(!output.includes('\ud83d ') && !output.includes(' \udc4d'), 'no space between surrogate halves')
  }
})

test('a combining mark stays attached to the character before it', () => {
  // 'e' + U+0301 COMBINING ACUTE ACCENT renders as one character; a space between
  // them would render as two, changing what the text looks like without changing
  // what the provider sees as a sequence.
  assert.equal(spaceOut('e\u0301x'), 'e\u0301 x')
  assert.equal(spaceOut('e\u0301'), 'e\u0301')
  assert.equal(spaceOut('\u0301\u0301'), '\u0301\u0301', 'leading marks have no base to detach from')
})

test('a zero-width joiner keeps an emoji sequence together', () => {
  assert.equal(spaceOut('👨\u200d👩'), '👨\u200d 👩')
  assert.equal(spaceOut('👨\u200d👩\u200d👧'), '👨\u200d 👩\u200d 👧')
})

test('the transform is idempotent', () => {
  const corpus = [
    'hello', 'a b', '', '中文字', '👍a', 'e\u0301x', '👨\u200d👩', 'a\nb\tc',
    'https://example.com/path?q=1#frag', '{"a":1,"b":[2,3]}', 'a\u0301\u200d',
  ]
  for (const input of corpus) {
    const once = spaceOut(input)
    assert.equal(spaceOut(once), once, `${JSON.stringify(input)} is a fixed point after one pass`)
  }
})

test('the transform is what breaks a matched sequence', () => {
  // The whole point: a provider matching a character sequence no longer finds it.
  const trigger = 'FLAG-TOKEN-1234'
  assert.equal(spaceOut(trigger).includes(trigger), false)
})

test('isSpace recognizes whitespace and a zero-width space', () => {
  for (const point of [' ', '\t', '\n', '\r', '\u00a0', '\u3000']) assert.equal(isSpace(point), true, JSON.stringify(point))
  assert.equal(isSpace('\u200b'), true)
  for (const point of ['a', '0', '!', '\u0301', '\u200d']) assert.equal(isSpace(point), false, JSON.stringify(point))
})
