/**
 * Packaging guard: the published artifact must declare every bare specifier it
 * imports, and must not declare packages it never imports.
 *
 * Both directions have bitten this repository's plugins before:
 *
 * - A value import left in `peerDependencies`-only (or, worse,
 *   `devDependencies`-only) resolves fine on the publishing machine — the
 *   workspace is right there — and fails for a consumer with
 *   `ERR_MODULE_NOT_FOUND`. `npm i <name>` in an empty directory is the only
 *   shape that catches it, and it only catches it after the release.
 * - A dependency kept after the import is gone is invisible until someone audits
 *   the tarball, and it silently widens the install closure.
 *
 * So the guard asserts an equivalence, not a one-way rule: the set of bare
 * specifiers reached by `src/**\/*.ts` and `lib/**\/*.js` equals the set of
 * runtime-declared packages (`dependencies` ∪ `peerDependencies`). A missing
 * declaration and a stale declaration both fail.
 *
 * `node:test` walks the same directory as the behaviour suite, so `npm test`
 * cannot pass without it — and the build runs first, so `lib/` is present.
 */

import assert from 'node:assert/strict'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'))

/** Every file under `dir` whose name ends with one of `extensions`. */
function walk(dir, extensions, found = []) {
  if (!existsSync(dir)) return found
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) walk(path, extensions, found)
    else if (extensions.some((extension) => entry.name.endsWith(extension))) found.push(path)
  }
  return found
}

/**
 * Bare specifiers reached by `import`/`export from`, dynamic `import()`, and
 * `require()` in one source text.
 *
 * Relative and absolute targets are not packages; `node:` builtins are not
 * installable. Both are dropped here rather than filtered per call site.
 */
function bareSpecifiers(source) {
  const specifiers = new Set()
  const patterns = [
    /(?:^|[\s;{])(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]/g,
    /(?:^|[\s;(=])(?:import|require)\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
    /(?:^|[\s;{])import\s*['"]([^'"]+)['"]/g,
  ]
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) specifiers.add(match[1])
  }
  return [...specifiers].filter((specifier) => {
    if (specifier.startsWith('.') || specifier.startsWith('/')) return false
    if (specifier.startsWith('node:')) return false
    return true
  })
}

/** The package a specifier resolves to: `@scope/name` or `name`, subpath dropped. */
function packageOf(specifier) {
  const segments = specifier.split('/')
  return specifier.startsWith('@') ? segments.slice(0, 2).join('/') : segments[0]
}

const runtimeDeclared = new Set([
  ...Object.keys(manifest.dependencies ?? {}),
  ...Object.keys(manifest.peerDependencies ?? {}),
])
const devDeclared = new Set(Object.keys(manifest.devDependencies ?? {}))

const sourceFiles = [...walk(join(ROOT, 'src'), ['.ts']), ...walk(join(ROOT, 'lib'), ['.js'])]
const imported = new Set()
for (const file of sourceFiles) {
  for (const specifier of bareSpecifiers(readFileSync(file, 'utf8'))) imported.add(packageOf(specifier))
}

test('the guard has something to guard', () => {
  // A walk that silently finds nothing would make both assertions below vacuous.
  assert.ok(sourceFiles.some((file) => file.includes(join('src', 'index.ts'))), 'src/index.ts must be scanned')
  assert.ok(sourceFiles.some((file) => file.endsWith(join('lib', 'index.js'))), 'lib/index.js must be built and scanned')
  assert.ok(imported.size >= 3, `expected several bare imports, saw ${imported.size}`)
})

test('every bare import in the published artifact is a runtime dependency', () => {
  const undeclared = [...imported].filter((name) => !runtimeDeclared.has(name))
  assert.deepEqual(
    undeclared,
    [],
    `these packages are imported by the shipped code but declared only in devDependencies (or nowhere): ${undeclared.join(', ')}`
      + ' — a consumer installing this package by name gets ERR_MODULE_NOT_FOUND',
  )
})

test('no runtime dependency is declared without being imported', () => {
  const unused = [...runtimeDeclared].filter((name) => !imported.has(name))
  assert.deepEqual(unused, [], `declared but never imported by shipped code: ${unused.join(', ')}`)
})

test('test-only packages stay out of the runtime closure', () => {
  const leaked = [...devDeclared].filter((name) => runtimeDeclared.has(name))
  // cordis and dsh-llm are deliberately both: peers for consumers, dev deps so
  // this repository can build and test itself.
  const allowed = new Set(Object.keys(manifest.peerDependencies ?? {}))
  assert.deepEqual(
    leaked.filter((name) => !allowed.has(name)),
    [],
    `a devDependency also declared as a runtime dependency: ${leaked.join(', ')}`,
  )
})

test('the shipped file list carries the manifest the harness reads', () => {
  // `dsh.bundle.patch` is how a profile finds the mount entry; shipping the
  // package without it makes the published artifact unusable.
  assert.equal(manifest.dsh.bundle.patch, './cordis.patch.yml')
  assert.ok(manifest.files.includes('cordis.patch.yml'), 'the patch file must be in `files`')
  assert.ok(existsSync(join(ROOT, 'cordis.patch.yml')), 'and must exist')
})

test('the entry point ships with the repository, not only with a local build', () => {
  // A consumer installs this package from the repository, and a git install
  // carries only tracked files. Building `lib/` locally proves nothing about the
  // artifact: if `lib/` is ignored, every `dsh plugin add` of this repository
  // attaches a package whose `main` does not exist, and the profile mounts
  // nothing while reporting success. That is silent, so it is asserted here.
  assert.ok(manifest.files.includes('lib/index.js'), 'the built entry must be in `files`')
  assert.ok(existsSync(join(ROOT, 'lib/index.js')), 'and must exist (run the build)')

  const ignore = readFileSync(join(ROOT, '.gitignore'), 'utf8')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '' && !line.startsWith('#'))
  assert.deepEqual(
    ignore.filter((pattern) => pattern === 'lib/' || pattern === 'lib' || pattern === '/lib/'),
    [],
    'lib/ must not be ignored: a git install would ship no entry point',
  )
})
