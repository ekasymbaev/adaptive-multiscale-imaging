#!/usr/bin/env node
/** Build and verify the Step 6 adaptive-fusion results workbook. */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function fatal(error) {
  console.error(error instanceof Error ? `${error.name}: ${error.message}` : String(error));
  process.exit(1);
}
process.on("uncaughtException", fatal);
process.on("unhandledRejection", fatal);

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultRoot = path.join(projectRoot, "results", "adaptive_fusion");
const metricRoot = path.join(resultRoot, "metrics");
const outputRoot = path.join(projectRoot, "outputs", "step6_adaptive_fusion");
const previewRoot = process.argv[2] ?? path.join("/tmp", "step6-adaptive-fusion-workbook-preview");

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + (value % 26)) + name;
    value = Math.floor(value / 26);
  }
  return name;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const appendField = () => {
    const value = field.trim();
    if (value !== "" && /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/.test(value)) {
      row.push(Number(value));
    } else if (value === "True") {
      row.push(true);
    } else if (value === "False") {
      row.push(false);
    } else {
      row.push(value);
    }
    field = "";
  };
  const appendRow = () => {
    appendField();
    if (row.some((value) => value !== "")) rows.push(row);
    row = [];
  };
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      appendField();
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      appendRow();
    } else {
      field += character;
    }
  }
  if (field !== "" || row.length > 0) appendRow();
  return rows;
}

function addCsvSheet(workbook, name, text) {
  const values = parseCsv(text);
  const sheet = workbook.worksheets.add(name);
  const lastColumn = columnName(values[0].length - 1);
  sheet.getRange(`A1:${lastColumn}${values.length}`).values = values;
  return { sheet, values };
}

function headerColumn(values, headerName) {
  const index = values[0].indexOf(headerName);
  if (index < 0) throw new Error(`Missing column: ${headerName}`);
  return columnName(index);
}

function dataCell(values, predicate, headerName) {
  const rowIndex = values.findIndex((row, index) => index > 0 && predicate(row));
  if (rowIndex < 1) throw new Error(`Could not find row for ${headerName}`);
  return `${headerColumn(values, headerName)}${rowIndex + 1}`;
}

function flattenObject(value, prefix = "") {
  const rows = [];
  for (const [key, item] of Object.entries(value)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (item && typeof item === "object" && !Array.isArray(item)) {
      rows.push(...flattenObject(item, fullKey));
    } else {
      rows.push([fullKey, Array.isArray(item) ? item.join(", ") : item]);
    }
  }
  return rows;
}

function formatDataSheet(sheet, values, tableName, freezeColumns = 2) {
  const rowCount = values.length;
  const columnCount = values[0].length;
  const lastColumn = columnName(columnCount - 1);
  const used = sheet.getRange(`A1:${lastColumn}${rowCount}`);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(freezeColumns, columnCount));
  used.format.font = { name: "Aptos", size: 9 };
  used.format.autofitColumns();
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#16324F",
    font: { bold: true, color: "#FFFFFF", name: "Aptos" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#16324F" },
  };
  header.format.rowHeight = 52;
  for (let column = 0; column < columnCount; column += 1) {
    const name = String(values[0][column]);
    const range = sheet.getRange(`${columnName(column)}2:${columnName(column)}${rowCount}`);
    if (/coverage_fraction|improved_image_fraction|recovery_fraction/.test(name)) {
      range.format.numberFormat = "0.0%";
    } else if (/dice|iou|precision|recall|error|fraction|gain|mean|ci_low|ci_high|variance|entropy|ratio|probability/.test(name)) {
      range.format.numberFormat = "0.0000";
    } else if (/pixels|images|tiles|fold|budget|trial|seed|rank|row|column|index|width/.test(name)) {
      range.format.numberFormat = "#,##0";
    }
    if (/file_name|ranking_source|improvement_direction|example_role/.test(name)) {
      range.format.columnWidth = 32;
      range.format.wrapText = true;
    }
  }
  sheet.tables.add(`A1:${lastColumn}${rowCount}`, true, tableName);
}

const files = {
  performance: "performance_cost_summary.csv",
  paired: "paired_comparisons.csv",
  perImage: "per_image_policy_metrics.csv",
  randomTrials: "random_trial_metrics.csv",
  tileEffects: "tile_fusion_effects.csv",
  boundary: "boundary_audit.csv",
  verification: "mask_verification.csv",
  representatives: "representative_samples.csv",
};
const csv = {};
for (const [key, fileName] of Object.entries(files)) {
  csv[key] = await fs.readFile(path.join(metricRoot, fileName), "utf8");
}
const summary = JSON.parse(await fs.readFile(path.join(metricRoot, "summary.json"), "utf8"));
const auditSummary = JSON.parse(
  await fs.readFile(path.join(metricRoot, "adaptive_audit.json"), "utf8"),
);

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
const performance = addCsvSheet(workbook, "Performance Cost", csv.performance);
const paired = addCsvSheet(workbook, "Paired Comparisons", csv.paired);
const perImage = addCsvSheet(workbook, "Per-image Metrics", csv.perImage);
const randomTrials = addCsvSheet(workbook, "Random Trials", csv.randomTrials);
const tileEffects = addCsvSheet(workbook, "Tile Effects", csv.tileEffects);
const boundary = addCsvSheet(workbook, "Boundary Audit", csv.boundary);
const verification = addCsvSheet(workbook, "Mask Verification", csv.verification);
const representatives = addCsvSheet(workbook, "Representative Images", csv.representatives);
const configSheet = workbook.worksheets.add("Run Configuration");

formatDataSheet(performance.sheet, performance.values, "AdaptivePerformanceCostTable", 4);
formatDataSheet(paired.sheet, paired.values, "AdaptivePairedComparisonTable", 5);
formatDataSheet(perImage.sheet, perImage.values, "AdaptivePerImageMetricsTable", 5);
formatDataSheet(randomTrials.sheet, randomTrials.values, "AdaptiveRandomTrialTable", 5);
formatDataSheet(tileEffects.sheet, tileEffects.values, "AdaptiveTileEffectsTable", 5);
formatDataSheet(boundary.sheet, boundary.values, "AdaptiveBoundaryAuditTable", 1);
formatDataSheet(verification.sheet, verification.values, "AdaptiveMaskVerificationTable", 2);
formatDataSheet(representatives.sheet, representatives.values, "AdaptiveRepresentativeTable", 4);

for (const [item, column] of [
  [performance, "dice_gain_over_coarse_mean"],
  [paired, "improvement_mean"],
  [perImage, "area_fraction_error_absolute"],
  [tileEffects, "feathered_error_reduction_pixels"],
]) {
  const letter = headerColumn(item.values, column);
  item.sheet.getRange(`${letter}2:${letter}${item.values.length}`).conditionalFormats.add(
    "colorScale",
    { colors: ["#FECACA", "#FEF3C7", "#D1FAE5"], thresholds: ["min", "50%", "max"] },
  );
}

const configValues = [
  ["parameter", "value"],
  ...flattenObject(summary.config),
  ["runtime.device", summary.device],
  ["runtime.python_version", summary.python_version],
  ["runtime.torch_version", summary.torch_version],
  ["runtime.platform", summary.platform],
  ["runtime.seconds", summary.runtime_seconds],
  ["audit.status", auditSummary.status],
  ["audit.metric_max_absolute_difference", auditSummary.reported_metric_max_absolute_difference],
];
configSheet.getRange(`A1:B${configValues.length}`).values = configValues;
formatDataSheet(configSheet, configValues, "AdaptiveRunConfigurationTable", 1);

summarySheet.showGridLines = false;
summarySheet.freezePanes.freezeRows(2);
summarySheet.getRange("A1:J1").merge();
summarySheet.getRange("A1").values = [["M-A Island Segmentation — Step 6 Adaptive Fusion"]];
summarySheet.getRange("A1:J1").format = {
  fill: "#16324F",
  font: { bold: true, color: "#FFFFFF", size: 18, name: "Aptos Display" },
  verticalAlignment: "center",
};
summarySheet.getRange("A1:J1").format.rowHeight = 34;
summarySheet.getRange("A2:J2").merge();
summarySheet.getRange("A2").values = [[
  "Frozen five-fold models; uncertainty-guided tile selection; 8-pixel feathered probability fusion; 100 matched random trials per held-out image. No model training was performed.",
]];
summarySheet.getRange("A2:J2").format = {
  fill: "#E8F0F7",
  font: { color: "#334155", italic: true, size: 10, name: "Aptos" },
  wrapText: true,
};

function section(address, title) {
  const range = summarySheet.getRange(address);
  range.merge();
  summarySheet.getRange(address.split(":")[0]).values = [[title]];
  range.format = {
    fill: "#2F6B8A",
    font: { bold: true, color: "#FFFFFF", name: "Aptos" },
    borders: { preset: "outside", style: "thin", color: "#2F6B8A" },
  };
}

section("A4:B4", "Experiment design");
summarySheet.getRange("A5:B15").values = [
  ["Held-out original images", summary.held_out_images],
  ["Native tile grid", "8 columns × 6 rows"],
  ["Native tile", "256 × 256"],
  ["Selection budgets", "2, 4, 8, 12, 24, 48 tiles"],
  ["Coverage range", "4.2% to 100%"],
  ["Primary ranking", "Step 4 predictive entropy q90"],
  ["Primary fusion", "8-pixel feathered probability blend"],
  ["Random baseline", "100 trials per image and budget"],
  ["Oracle", "Evaluation-only true fusion error reduction"],
  ["Frozen-mask verification", summary.frozen_probability_verification.status],
  ["Ground truth use", "Metrics and evaluation-only oracle only"],
];
summarySheet.getRange("A5:B15").format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};

function performanceRef(policy, budget, header) {
  const predicate = (row) => row[0] === "all_images" && row[2] === policy && row[4] === budget;
  return `='Performance Cost'!${dataCell(performance.values, predicate, header)}`;
}
section("D4:J4", "Primary operating points");
summarySheet.getRange("D5:J5").values = [[
  "Policy", "Coverage", "Dice", "IoU", "Precision", "Recall", "Absolute area error",
]];
summarySheet.getRange("D5:J5").format = {
  fill: "#DCE6EF",
  font: { bold: true, color: "#16324F", name: "Aptos" },
  wrapText: true,
};
const operatingPoints = [
  ["Coarse only", "coarse_only", 0],
  ["Entropy K=12", "entropy_feathered", 12],
  ["Entropy K=24", "entropy_feathered", 24],
  ["Full fine", "full_fine", 48],
];
summarySheet.getRange("D6:D9").values = operatingPoints.map(([label]) => [label]);
summarySheet.getRange("E6:J9").formulas = operatingPoints.map(([, policy, budget]) => [
  performanceRef(policy, budget, "coverage_fraction"),
  performanceRef(policy, budget, "macro_dice"),
  performanceRef(policy, budget, "macro_iou"),
  performanceRef(policy, budget, "macro_precision"),
  performanceRef(policy, budget, "macro_recall"),
  performanceRef(policy, budget, "area_fraction_error_absolute_mean"),
]);
summarySheet.getRange("D6:J9").format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("E6:E9").format.numberFormat = "0.0%";
summarySheet.getRange("F6:J9").format.numberFormat = "0.0000";

section("D11:J11", "Entropy-guided performance versus matched random selection");
summarySheet.getRange("D12:J12").values = [[
  "K", "Coverage", "Entropy Dice", "Random Dice", "Entropy − random", "95% CI low", "95% CI high",
]];
summarySheet.getRange("D12:J12").format = {
  fill: "#DCE6EF",
  font: { bold: true, color: "#16324F", name: "Aptos" },
  wrapText: true,
};
const budgets = summary.selection.budgets;
summarySheet.getRange(`D13:D${12 + budgets.length}`).values = budgets.map((value) => [value]);
summarySheet.getRange(`E13:J${12 + budgets.length}`).formulas = budgets.map((budget) => {
  const predicate = (row) =>
    row[0] === "all_images"
    && row[2] === "entropy_vs_random"
    && row[5] === budget
    && row[7] === "dice";
  return [
    performanceRef("entropy_feathered", budget, "coverage_fraction"),
    performanceRef("entropy_feathered", budget, "macro_dice"),
    performanceRef("random_feathered", budget, "macro_dice"),
    `='Paired Comparisons'!${dataCell(paired.values, predicate, "improvement_mean")}`,
    `='Paired Comparisons'!${dataCell(paired.values, predicate, "improvement_ci_low")}`,
    `='Paired Comparisons'!${dataCell(paired.values, predicate, "improvement_ci_high")}`,
  ];
});
summarySheet.getRange(`D13:J${12 + budgets.length}`).format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange(`E13:E${12 + budgets.length}`).format.numberFormat = "0.0%";
summarySheet.getRange(`F13:J${12 + budgets.length}`).format.numberFormat = "0.0000";

section("A17:B17", "Boundary audit");
summarySheet.getRange("A18:B21").values = [
  ["Policy", "Seam / interior error"],
  ["Coarse only", null],
  ["Full fine", null],
  ["Entropy feathered K=12", null],
];
summarySheet.getRange("B19:B21").formulas = [
  [`='Boundary Audit'!${dataCell(boundary.values, (row) => row[0] === "coarse_only", "seam_to_interior_error_ratio")}`],
  [`='Boundary Audit'!${dataCell(boundary.values, (row) => row[0] === "full_fine", "seam_to_interior_error_ratio")}`],
  [`='Boundary Audit'!${dataCell(boundary.values, (row) => row[0] === "entropy_feathered_k12", "seam_to_interior_error_ratio")}`],
];
summarySheet.getRange("A18:B21").format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("A18:B18").format.font = { bold: true, color: "#16324F" };
summarySheet.getRange("B19:B21").format.numberFormat = "0.00";

summarySheet.getRange("A23:J23").merge();
summarySheet.getRange("A23").values = [["Scientific conclusion"]];
summarySheet.getRange("A23:J23").format = {
  fill: "#DCEFE7",
  font: { bold: true, color: "#14532D", name: "Aptos" },
};
summarySheet.getRange("A24:J28").merge();
summarySheet.getRange("A24").values = [[
  "Uncertainty-guided adaptive fusion provides a real performance-versus-cost benefit. At K=12 (25% native coverage), Dice is 0.7968 versus 0.7799 coarse and 0.7888 random, recovering 45.8% of the full-fine gain. At K=24 (50% coverage), Dice reaches 0.8061 and recovers 71.0% of the gain, although it remains 0.0107 below full fine on average. Entropy beats matched random selection at every partial budget and in every source regime. Feathering reduces the full-fine tile seam/interior error ratio from 1.27× to 1.09× at K=12. The experiment is an offline counterfactual: reported cost is selected native-tile coverage, not measured microscope or wall-clock acquisition time.",
]];
summarySheet.getRange("A24:J28").format = {
  fill: "#F0FDF4",
  font: { color: "#334155", name: "Aptos" },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: "#86B89A" },
};
summarySheet.getRange("A24:J28").format.rowHeight = 24;
summarySheet.getRange("A1:J28").format.columnWidth = 16;
summarySheet.getRange("A:A").format.columnWidth = 29;
summarySheet.getRange("B:B").format.columnWidth = 35;
summarySheet.getRange("D:D").format.columnWidth = 23;

await fs.mkdir(previewRoot, { recursive: true });
const previewRanges = {
  Summary: "A1:J28",
  "Performance Cost": "A1:N22",
  "Paired Comparisons": "A1:Q20",
  "Per-image Metrics": "A1:N15",
  "Random Trials": "A1:N15",
  "Tile Effects": "A1:N15",
  "Boundary Audit": "A1:G4",
  "Mask Verification": "A1:E15",
  "Representative Images": "A1:N8",
  "Run Configuration": "A1:B40",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const rendered = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewRoot, `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await rendered.arrayBuffer()),
  );
}

const summaryInspection = await workbook.inspect({
  kind: "table,formula",
  sheetId: "Summary",
  range: "A1:J28",
  maxChars: 12000,
  tableMaxRows: 28,
  tableMaxCols: 10,
  options: { maxResults: 100 },
});
await fs.writeFile(
  path.join(previewRoot, "summary_inspection.txt"),
  String(summaryInspection.ndjson ?? summaryInspection),
  "utf8",
);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 10000,
});
await fs.writeFile(
  path.join(previewRoot, "formula_errors.txt"),
  String(errors.ndjson ?? errors),
  "utf8",
);

await fs.mkdir(outputRoot, { recursive: true });
const outputPath = path.join(outputRoot, "adaptive_fusion_results.xlsx");
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const finalBlob = await FileBlob.load(outputPath);
const finalWorkbook = await SpreadsheetFile.importXlsx(finalBlob);
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const rendered = await finalWorkbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewRoot, `final_${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await rendered.arrayBuffer()),
  );
}
const finalErrors = await finalWorkbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 10000,
});
await fs.writeFile(
  path.join(previewRoot, "final_formula_errors.txt"),
  String(finalErrors.ndjson ?? finalErrors),
  "utf8",
);
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });

console.log(`Workbook: ${outputPath}`);
console.log(`Previews: ${previewRoot}`);
