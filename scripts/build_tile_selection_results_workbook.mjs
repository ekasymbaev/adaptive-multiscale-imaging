#!/usr/bin/env node
/** Build and verify the Step 4 tile-selection results workbook. */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function reportFatalError(error) {
  const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  console.error(message);
  if (error instanceof Error && error.stack) {
    for (const line of error.stack.split("\n").filter((item) => item.includes("build_tile_selection_results_workbook"))) {
      console.error(line.trim());
    }
  }
  process.exit(1);
}

process.on("uncaughtException", reportFatalError);
process.on("unhandledRejection", reportFatalError);

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultRoot = path.join(projectRoot, "results", "tile_selection");
const outputRoot = path.join(projectRoot, "outputs", "step4_tile_selection");
const previewRoot = process.argv[2] ?? path.join("/tmp", "step4-tile-selection-workbook-preview");

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

  function appendField() {
    const value = field.trim();
    if (value !== "" && /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/.test(value)) {
      row.push(Number(value));
    } else {
      row.push(value);
    }
    field = "";
  }

  function appendRow() {
    appendField();
    if (row.some((value) => value !== "")) rows.push(row);
    row = [];
  }

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
  const finalColumn = columnName(values[0].length - 1);
  sheet.getRange(`A1:${finalColumn}${values.length}`).values = values;
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
  const finalColumn = columnName(columnCount - 1);
  const used = sheet.getRange(`A1:${finalColumn}${rowCount}`);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(freezeColumns, columnCount));
  used.format.font = { name: "Aptos", size: 9 };
  used.format.autofitColumns();
  const header = sheet.getRange(`A1:${finalColumn}1`);
  header.format = {
    fill: "#16324F",
    font: { bold: true, color: "#FFFFFF", name: "Aptos" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#16324F" },
  };
  header.format.rowHeight = 48;
  for (let column = 0; column < columnCount; column += 1) {
    const name = String(values[0][column]);
    const range = sheet.getRange(`${columnName(column)}2:${columnName(column)}${rowCount}`);
    if (tableName === "TileCorrelationTable" && /per_image_ci_low|per_image_ci_high/.test(name)) {
      range.format.numberFormat = "0.0000";
    } else if (/coverage_fraction|error_capture|capture_difference|capture_minus|capture_gap|win_fraction|ci_low|ci_high|error_fraction|selected_error_rate/.test(name)) {
      range.format.numberFormat = "0.0%";
    } else if (/correlation|enrichment|entropy|variance|score|mean|std|ratio|fraction_of_oracle/.test(name)) {
      range.format.numberFormat = "0.0000";
    } else if (/p$/.test(name)) {
      range.format.numberFormat = "0.00E+00";
    } else if (/pixels|images|trials|seed|fold|tile|row|column|budget|rank|resamples|index/.test(name)) {
      range.format.numberFormat = "#,##0";
    }
  }
  sheet.tables.add(`A1:${finalColumn}${rowCount}`, true, tableName);
}

const metricFiles = {
  budget: "budget_summary.csv",
  regime: "regime_budget_summary.csv",
  paired: "paired_policy_vs_random.csv",
  correlation: "correlation_summary.csv",
  perImage: "per_image_budget_metrics.csv",
  rankings: "uncertainty_rankings.csv",
  evaluatedTiles: "tile_evaluation.csv",
  randomTrials: "random_selection_trials.csv",
};
const csv = {};
for (const [key, fileName] of Object.entries(metricFiles)) {
  csv[key] = await fs.readFile(path.join(resultRoot, "metrics", fileName), "utf8");
}
const summary = JSON.parse(
  await fs.readFile(path.join(resultRoot, "metrics", "summary.json"), "utf8"),
);

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
const budget = addCsvSheet(workbook, "Budget Summary", csv.budget);
const regime = addCsvSheet(workbook, "Regime Summary", csv.regime);
const paired = addCsvSheet(workbook, "Paired vs Random", csv.paired);
const correlation = addCsvSheet(workbook, "Correlations", csv.correlation);
const perImage = addCsvSheet(workbook, "Per-image Budgets", csv.perImage);
const rankings = addCsvSheet(workbook, "Uncertainty Rankings", csv.rankings);
const evaluatedTiles = addCsvSheet(workbook, "Tile Evaluation", csv.evaluatedTiles);
const randomTrials = addCsvSheet(workbook, "Random Trials", csv.randomTrials);
const configSheet = workbook.worksheets.add("Run Configuration");

formatDataSheet(budget.sheet, budget.values, "TileBudgetSummaryTable", 2);
formatDataSheet(regime.sheet, regime.values, "TileRegimeSummaryTable", 3);
formatDataSheet(paired.sheet, paired.values, "TilePairedComparisonTable", 3);
formatDataSheet(correlation.sheet, correlation.values, "TileCorrelationTable", 3);
formatDataSheet(perImage.sheet, perImage.values, "TilePerImageBudgetTable", 4);
formatDataSheet(rankings.sheet, rankings.values, "UncertaintyRankingTable", 5);
formatDataSheet(evaluatedTiles.sheet, evaluatedTiles.values, "TileEvaluationTable", 5);
formatDataSheet(randomTrials.sheet, randomTrials.values, "RandomTrialTable", 5);

const budgetEnrichmentColumn = headerColumn(budget.values, "capture_enrichment_over_random");
budget.sheet
  .getRange(`${budgetEnrichmentColumn}2:${budgetEnrichmentColumn}${budget.values.length}`)
  .conditionalFormats.add("colorScale", {
    colors: ["#FECACA", "#FEF3C7", "#D1FAE5"],
    thresholds: ["min", "50%", "max"],
  });
const pairedLowColumn = headerColumn(paired.values, "capture_difference_ci_low");
paired.sheet
  .getRange(`${pairedLowColumn}2:${pairedLowColumn}${paired.values.length}`)
  .conditionalFormats.add("cellIs", {
    operator: "greaterThan",
    formula: 0,
    format: { fill: "#D1FAE5", font: { color: "#14532D" } },
  });

const configValues = [
  ["parameter", "value"],
  ...flattenObject(summary.config),
  ["ranking.primary_policy", summary.ranking.primary_policy],
  ["ranking.secondary_policy", summary.ranking.secondary_policy],
  ["ranking.scope", summary.ranking.ranking_scope],
  ["ranking.ground_truth_use", summary.ranking.ground_truth_use],
  ["confidence_interval.method", summary.paired_confidence_intervals.method],
  ["runtime.seconds", summary.runtime.seconds],
  ["runtime.python_version", summary.runtime.python_version],
  ["runtime.numpy_version", summary.runtime.numpy_version],
  ["runtime.pandas_version", summary.runtime.pandas_version],
  ["runtime.opencv_version", summary.runtime.opencv_version],
];
configSheet.getRange(`A1:B${configValues.length}`).values = configValues;
formatDataSheet(configSheet, configValues, "TileConfigurationTable", 1);

summarySheet.showGridLines = false;
summarySheet.freezePanes.freezeRows(2);
summarySheet.getRange("A1:J1").merge();
summarySheet.getRange("A1").values = [["M-A Island Coarse U-Net — Step 4 Tile Selection Results"]];
summarySheet.getRange("A1:J1").format = {
  fill: "#16324F",
  font: { bold: true, color: "#FFFFFF", size: 18, name: "Aptos Display" },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summarySheet.getRange("A1:J1").format.rowHeight = 34;
summarySheet.getRange("A2:J2").merge();
summarySheet.getRange("A2").values = [[
  "Within-image q90 uncertainty ranking on 48 native tiles per held-out image; labels enter only during evaluation and oracle construction.",
]];
summarySheet.getRange("A2:J2").format = {
  fill: "#E8F0F7",
  font: { color: "#334155", italic: true, size: 10, name: "Aptos" },
  wrapText: true,
};

summarySheet.getRange("A4:B4").values = [["Experiment design", "Value"]];
summarySheet.getRange("A5:B13").values = [
  ["Held-out images", summary.held_out_images],
  ["Native image shape", "2048 x 1536"],
  ["Native tile grid", "8 columns x 6 rows"],
  ["Native tile size", "256 x 256"],
  ["Coarse block per tile", "64 x 64"],
  ["Primary tile score", "90th-percentile predictive entropy"],
  ["Random trials/image", summary.random_trials_per_image],
  ["Bootstrap resamples", summary.paired_confidence_intervals.resamples],
  ["Ground-truth use", "Evaluation and oracle only"],
];

summarySheet.getRange("D4:E4").values = [["Tile-error association", "Mean"]];
summarySheet.getRange("D5:D10").values = [
  ["Entropy Pearson correlation"],
  ["Entropy Spearman correlation"],
  ["Variance Pearson correlation"],
  ["Variance Spearman correlation"],
  ["Regime I entropy Spearman"],
  ["K=8 entropy win fraction"],
];
function correlationRef(policy, method, scope = "all_images") {
  const cell = dataCell(
    correlation.values,
    (row) => row[0] === scope && row[1] === policy && row[2] === method,
    "per_image_mean",
  );
  return `='Correlations'!${cell}`;
}
summarySheet.getRange("E5:E9").formulas = [
  [correlationRef("entropy_q90", "pearson")],
  [correlationRef("entropy_q90", "spearman")],
  [correlationRef("variance_q90", "pearson")],
  [correlationRef("variance_q90", "spearman")],
  [correlationRef("entropy_q90", "spearman", "regime_I")],
];
const pairedK8WinCell = dataCell(
  paired.values,
  (row) => row[0] === "all_images" && row[1] === "entropy_q90" && row[2] === 8,
  "win_fraction",
);
summarySheet.getRange("E10").formulas = [[`='Paired vs Random'!${pairedK8WinCell}`]];

summarySheet.getRange("A15:H15").values = [[
  "K",
  "Coverage",
  "Entropy capture",
  "Random capture",
  "Entropy − random",
  "Paired CI low",
  "Paired CI high",
  "Entropy / oracle",
]];
const budgets = summary.budgets;
summarySheet.getRange(`A16:A${15 + budgets.length}`).values = budgets.map((value) => [value]);
const budgetRows = [];
for (const budgetValue of budgets) {
  const entropyPredicate = (row) => row[0] === "entropy_q90" && row[1] === budgetValue;
  const randomPredicate = (row) => row[0] === "random" && row[1] === budgetValue;
  const pairedPredicate = (row) => row[0] === "all_images" && row[1] === "entropy_q90" && row[2] === budgetValue;
  budgetRows.push([
    `='Budget Summary'!${dataCell(budget.values, entropyPredicate, "coverage_fraction")}`,
    `='Budget Summary'!${dataCell(budget.values, entropyPredicate, "error_capture_mean")}`,
    `='Budget Summary'!${dataCell(budget.values, randomPredicate, "error_capture_mean")}`,
    `='Paired vs Random'!${dataCell(paired.values, pairedPredicate, "capture_difference_mean")}`,
    `='Paired vs Random'!${dataCell(paired.values, pairedPredicate, "capture_difference_ci_low")}`,
    `='Paired vs Random'!${dataCell(paired.values, pairedPredicate, "capture_difference_ci_high")}`,
    `='Budget Summary'!${dataCell(budget.values, entropyPredicate, "capture_fraction_of_oracle")}`,
  ]);
}
summarySheet.getRange(`B16:H${15 + budgets.length}`).formulas = budgetRows;

for (const rangeAddress of ["A4:B4", "D4:E4", "A15:H15"]) {
  summarySheet.getRange(rangeAddress).format = {
    fill: "#2F6B8A",
    font: { bold: true, color: "#FFFFFF", name: "Aptos" },
    borders: { preset: "outside", style: "thin", color: "#2F6B8A" },
  };
}
for (const rangeAddress of ["A5:B13", "D5:E10", `A16:H${15 + budgets.length}`]) {
  summarySheet.getRange(rangeAddress).format = {
    fill: "#F8FAFC",
    font: { name: "Aptos" },
    borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
  };
}
summarySheet.getRange("E5:E9").format.numberFormat = "0.000";
summarySheet.getRange("E10").format.numberFormat = "0.0%";
summarySheet.getRange(`B16:H${15 + budgets.length}`).format.numberFormat = "0.0%";
summarySheet.getRange(`A16:A${15 + budgets.length}`).format.numberFormat = "0";

summarySheet.getRange("A24:J24").merge();
summarySheet.getRange("A24").values = [["Interpretation and Step 5 readiness"]];
summarySheet.getRange("A24:J24").format = {
  fill: "#DCEFE7",
  font: { bold: true, color: "#14532D", name: "Aptos" },
};
summarySheet.getRange("A25:J28").merge();
summarySheet.getRange("A25").values = [[
  "Within-image entropy-q90 ranking consistently captures more coarse segmentation error than 100-trial random selection at every partial budget. At K=8 (16.7% coverage), entropy captures about 27.5% of error versus 16.7% for random and reaches about 87% of the oracle capture. The paired 95% confidence interval excludes zero in every source regime. This supports training a fine-resolution model for a controlled downstream benefit-versus-cost experiment; it does not yet prove that selected high-resolution inspection will correct the errors.",
]];
summarySheet.getRange("A25:J28").format = {
  fill: "#F0FDF4",
  font: { color: "#334155", name: "Aptos" },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: "#86B89A" },
};
summarySheet.getRange("A25:J28").format.rowHeight = 25;
summarySheet.getRange("A1:J28").format.columnWidth = 17;
summarySheet.getRange("A:A").format.columnWidth = 26;
summarySheet.getRange("B:B").format.columnWidth = 31;
summarySheet.getRange("D:D").format.columnWidth = 29;

await fs.mkdir(previewRoot, { recursive: true });
const previewRanges = {
  Summary: "A1:J28",
  "Budget Summary": "A1:L25",
  "Regime Summary": "A1:L20",
  "Paired vs Random": "A1:N18",
  Correlations: "A1:J21",
  "Per-image Budgets": "A1:L14",
  "Uncertainty Rankings": "A1:N14",
  "Tile Evaluation": "A1:N14",
  "Random Trials": "A1:L14",
  "Run Configuration": "A1:B30",
};
const sheetNames = Object.keys(previewRanges);
for (const sheetName of sheetNames) {
  const rendered = await workbook.render({
    sheetName,
    range: previewRanges[sheetName],
    scale: 1,
    format: "png",
  });
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
const outputPath = path.join(outputRoot, "tile_selection_results.xlsx");
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const finalBlob = await FileBlob.load(outputPath);
const finalWorkbook = await SpreadsheetFile.importXlsx(finalBlob);
for (const sheetName of sheetNames) {
  const rendered = await finalWorkbook.render({
    sheetName,
    range: previewRanges[sheetName],
    scale: 1,
    format: "png",
  });
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
