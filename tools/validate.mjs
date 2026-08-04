/* Content validator: executes the browser IIFEs in an isolated context and
 * builds every diagram so the static geometry checks run. Node built-ins only.
 *
 *   node tools/validate.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const context = { window: {} };
vm.createContext(context);

for (const file of ["assets/chapters.js", "assets/code.js", "assets/diagrams.js"]) {
  vm.runInContext(readFileSync(join(root, file), "utf8"), context, { filename: file });
}

const { MAGI_CHAPTERS, MAGI_CODE, MagiDiagrams } = context.window;
const errors = [];
const check = (cond, msg) => { if (!cond) errors.push(msg); };

check(Array.isArray(MAGI_CHAPTERS) && MAGI_CHAPTERS.length === 8, "expect 8 chapters");

const AUX_KEYS = new Set([
  "attnslice-masktypes", "tmem-map", "pipeline-wave", "mask-segments",
  "correction-handshake", "lpt-swizzle", "bwd-tmem"
]);

for (const chapter of MAGI_CHAPTERS) {
  const where = `chapter ${chapter.id}`;
  check(chapter.order >= 0, `${where}: order`);
  check(chapter.takeaway && chapter.takeaway.length > 40, `${where}: takeaway too short`);
  check((chapter.intuitions || []).length === 3, `${where}: expect 3 intuition cards`);
  check((chapter.motivation || []).length >= 2, `${where}: motivation`);
  check((chapter.explain || []).length >= 3, `${where}: explain sections`);
  const numExercises = (chapter.exercises || []).length;
  check(numExercises >= 5 && numExercises <= 7, `${where}: ${numExercises} exercises (want 5..7)`);
  for (const exercise of chapter.exercises || []) {
    check(exercise.kind && exercise.level && exercise.q && exercise.hint && exercise.answer,
      `${where}: incomplete exercise`);
  }
  check((chapter.sources || []).length >= 3, `${where}: sources`);
  for (const source of chapter.sources || []) {
    check(/^https?:\/\//.test(source.url), `${where}: bad url ${source.url}`);
  }
  const impl = MAGI_CODE[chapter.id];
  check(!!impl, `${where}: missing code entry`);
  if (impl) {
    check(impl.blocks.length >= 5, `${where}: only ${impl.blocks.length} code blocks`);
    for (const block of impl.blocks) {
      check(/^\d{2}$/.test(block.id), `${where}: bad block id ${block.id}`);
      check(block.start > 0 && block.end >= block.start, `${where}: bad lines in block ${block.id}`);
      check(block.code.length > 40, `${where}: block ${block.id} code too short`);
    }
  }
  for (const section of chapter.explain || []) {
    if (section.svg) check(AUX_KEYS.has(section.svg), `${where}: unknown aux svg ${section.svg}`);
  }
  try {
    const report = MagiDiagrams.build(chapter.diagram.key);
    check(report.svg.length > 500, `${where}: diagram svg too small`);
    check(report.notes.length >= 4, `${where}: diagram notes`);
    const ids = [...report.svg.matchAll(/data-code-block="([^"]+)"/g)].map((m) => m[1]);
    check(ids.length >= 5, `${where}: diagram has only ${ids.length} interactive nodes`);
  } catch (error) {
    errors.push(`${where}: diagram build failed: ${error.message}`);
  }
}

for (const key of AUX_KEYS) {
  try {
    const report = MagiDiagrams.buildAux(key);
    check(report && report.svg.length > 300, `aux ${key}: svg too small`);
    check(report && report.caption.length > 10, `aux ${key}: caption`);
  } catch (error) {
    errors.push(`aux ${key}: build failed: ${error.message}`);
  }
}

if (errors.length) {
  console.error(`✗ ${errors.length} problem(s):`);
  for (const error of errors) console.error("  - " + error);
  process.exit(1);
}
console.log("✓ all chapters, code blocks and diagrams validated");
