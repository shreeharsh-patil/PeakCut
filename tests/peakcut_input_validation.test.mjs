import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../web/peakcut-input-validation.js", import.meta.url), "utf8");
const window = {};
vm.runInNewContext(source, { window, globalThis: {} });
const validateOptions = window.PeakCutInputValidation.validateOptions;

test("accepts normal and partially specified range options", () => {
  assert.equal(validateOptions({ minLength: "15", maxLength: "60" }), null);
  assert.equal(validateOptions({ minLength: "15", maxLength: "60", startMinute: "2" }), null);
});

test("rejects inverted clip duration bounds and identifies the field", () => {
  assert.deepEqual(
    JSON.parse(JSON.stringify(validateOptions({ minLength: "60", maxLength: "15" }))),
    { field: "o_maxlen", message: "Shortest clip length must be less than or equal to longest clip length." },
  );
});

test("requires an end minute after a selected start minute", () => {
  assert.equal(validateOptions({ minLength: "15", maxLength: "60", startMinute: "3", endMinute: "3" }).field, "o_to");
  assert.equal(validateOptions({ minLength: "15", maxLength: "60", startMinute: "3", endMinute: "5" }), null);
});
