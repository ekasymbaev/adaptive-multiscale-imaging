#!/usr/bin/env node
/** Build and verify the Step 5 fine-model results workbook. */

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
const resultRoot = path.join(projectRoot, "results", "fine_model");
const metricRoot = path.join(resultRoot, "metrics");
const outputRoot = path.join(projectRoot, "outputs", "step5_fine_model");
const previewRoot = process.argv[2] ?? path.join("/tmp", "step5-fine-model-workbook-preview");

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
  const rows = values.length;
  const columns = values[0].length;
  const lastColumn = columnName(columns - 1);
  const used = sheet.getRange(`A1:${lastColumn}${rows}`);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(freezeColumns, columns));
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
  header.format.rowHeight = 50;
  for (let column = 0; column < columns; column += 1) {
    const name = String(values[0][column]);
    const range = sheet.getRange(`${columnName(column)}2:${columnName(column)}${rows}`);
    if (/dice|iou|precision|recall|fraction|error_rate|ratio|mean|std|loss|weight|learning_rate|ci_low|ci_high|improvement/.test(name)) {
      range.format.numberFormat = "0.0000";
    } else if (/p$|_p_/.test(name)) {
      range.format.numberFormat = "0.00E+00";
    } else if (/pixels|images|tiles|fold|epoch|seed|row|column|index|count|parameters|width|height/.test(name)) {
      range.format.numberFormat = "#,##0";
    }
    if (/path|file_name|reason|checkpoint/.test(name)) {
      range.format.columnWidth = 34;
      range.format.wrapText = true;
    }
  }
  sheet.tables.add(`A1:${lastColumn}${rows}`, true, tableName);
}

const files = {
  model: "model_summary.csv",
  paired: "paired_comparison.csv",
  audit: "reconstruction_audit.csv",
  folds: "fold_metrics.csv",
  comparison: "fine_vs_coarse_per_image.csv",
  images: "per_image_fine_metrics.csv",
  tiles: "per_tile_test_metrics.csv",
  manifest: "native_tile_manifest.csv",
  foldManifest: "fold_tile_manifest.csv",
};
const csv = {};
for (const [key, fileName] of Object.entries(files)) {
  csv[key] = await fs.readFile(path.join(metricRoot, fileName), "utf8");
}
csv.history = await fs.readFile(path.join(resultRoot, "history", "all_folds_history.csv"), "utf8");
const summary = JSON.parse(await fs.readFile(path.join(metricRoot, "summary.json"), "utf8"));
const reconstructionAudit = JSON.parse(
  await fs.readFile(path.join(metricRoot, "reconstruction_audit.json"), "utf8"),
);

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
const model = addCsvSheet(workbook, "Model Summary", csv.model);
const paired = addCsvSheet(workbook, "Paired Comparison", csv.paired);
const audit = addCsvSheet(workbook, "Reconstruction Audit", csv.audit);
const folds = addCsvSheet(workbook, "Fold Metrics", csv.folds);
const comparison = addCsvSheet(workbook, "Per-image Comparison", csv.comparison);
const images = addCsvSheet(workbook, "Fine Image Metrics", csv.images);
const history = addCsvSheet(workbook, "Training History", csv.history);
const tiles = addCsvSheet(workbook, "Tile Metrics", csv.tiles);
const manifest = addCsvSheet(workbook, "Tile Manifest", csv.manifest);
const foldManifest = addCsvSheet(workbook, "Fold Tile Manifest", csv.foldManifest);
const configSheet = workbook.worksheets.add("Run Configuration");

formatDataSheet(model.sheet, model.values, "FineModelSummaryTable", 3);
formatDataSheet(paired.sheet, paired.values, "FinePairedComparisonTable", 3);
formatDataSheet(audit.sheet, audit.values, "FineReconstructionAuditTable", 1);
formatDataSheet(folds.sheet, folds.values, "FineFoldMetricsTable", 2);
formatDataSheet(comparison.sheet, comparison.values, "FinePerImageComparisonTable", 4);
formatDataSheet(images.sheet, images.values, "FineImageMetricsTable", 4);
formatDataSheet(history.sheet, history.values, "FineTrainingHistoryTable", 2);
formatDataSheet(tiles.sheet, tiles.values, "FineTileMetricsTable", 5);
formatDataSheet(manifest.sheet, manifest.values, "FineTileManifestTable", 5);
formatDataSheet(foldManifest.sheet, foldManifest.values, "FineFoldTileManifestTable", 5);

for (const [item, column] of [
  [comparison, "fine_minus_coarse_dice"],
  [comparison, "fine_minus_coarse_iou"],
  [comparison, "area_fraction_absolute_error_improvement"],
  [paired, "improvement_mean"],
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
  ["runtime.total_training_seconds", summary.total_training_seconds],
  ["audit.status", reconstructionAudit.status],
  ["audit.image_level_split_disjointness", reconstructionAudit.image_level_split_disjointness],
];
configSheet.getRange(`A1:B${configValues.length}`).values = configValues;
formatDataSheet(configSheet, configValues, "FineRunConfigurationTable", 1);

summarySheet.showGridLines = false;
summarySheet.freezePanes.freezeRows(2);
summarySheet.getRange("A1:J1").merge();
summarySheet.getRange("A1").values = [["M-A Island Segmentation — Step 5 Fine-Resolution Model"]];
summarySheet.getRange("A1:J1").format = {
  fill: "#16324F",
  font: { bold: true, color: "#FFFFFF", size: 18, name: "Aptos Display" },
  verticalAlignment: "center",
};
summarySheet.getRange("A1:J1").format.rowHeight = 34;
summarySheet.getRange("A2:J2").merge();
summarySheet.getRange("A2").values = [[
  "Five-fold image-level cross-validation; native 256×256 tiles are reassembled into 2048×1536 held-out maps and compared with the frozen coarse model on the same native masks.",
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
summarySheet.getRange("A5:B14").values = [
  ["Held-out original images", summary.held_out_images],
  ["Held-out native tiles", summary.held_out_tiles],
  ["Native image", "2048 × 1536"],
  ["Tile grid", "8 columns × 6 rows"],
  ["Native tile", "256 × 256"],
  ["Train / validation / test images per fold", "24 / 8 / 8"],
  ["Compact U-Net parameters", summary.model_parameters],
  ["Loss", "0.5 weighted BCE + 0.5 soft Dice"],
  ["Sampling", "Uniform over all training-image tiles"],
  ["Split leakage audit", "Passed — original images are disjoint"],
];
summarySheet.getRange("A5:B14").format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};

section("D4:J4", "Held-out native-grid comparison");
summarySheet.getRange("D5:J5").values = [[
  "Metric", "Coarse mean", "Fine mean", "Improvement", "95% CI low", "95% CI high", "Images improved",
]];
summarySheet.getRange("D5:J5").format = {
  fill: "#DCE6EF",
  font: { bold: true, color: "#16324F", name: "Aptos" },
  wrapText: true,
};
const metricRows = [
  ["Dice", "dice"],
  ["IoU", "iou"],
  ["Foreground precision", "precision"],
  ["Foreground recall", "recall"],
  ["Absolute area-fraction error", "area_fraction_absolute_error"],
];
summarySheet.getRange("D6:D10").values = metricRows.map(([label]) => [label]);
summarySheet.getRange("E6:J10").formulas = metricRows.map(([, metric]) => {
  const predicate = (row) => row[0] === "all_images" && row[1] === metric;
  return [
    `='Paired Comparison'!${dataCell(paired.values, predicate, "coarse_mean")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "fine_mean")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improvement_mean")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improvement_ci_low")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improvement_ci_high")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improved_image_fraction")}`,
  ];
});
summarySheet.getRange("D6:J10").format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("E6:I10").format.numberFormat = "0.0000";
summarySheet.getRange("J6:J10").format.numberFormat = "0.0%";

section("D12:J12", "Dice improvement by source regime");
summarySheet.getRange("D13:J13").values = [[
  "Regime", "Coarse Dice", "Fine Dice", "Fine − coarse", "95% CI low", "95% CI high", "Images improved",
]];
summarySheet.getRange("D13:J13").format = {
  fill: "#DCE6EF",
  font: { bold: true, color: "#16324F", name: "Aptos" },
};
const regimes = ["I", "II", "III", "IV"];
summarySheet.getRange("D14:D17").values = regimes.map((value) => [`Regime ${value}`]);
summarySheet.getRange("E14:J17").formulas = regimes.map((regime) => {
  const predicate = (row) => row[0] === `regime_${regime}` && row[1] === "dice";
  return [
    `='Paired Comparison'!${dataCell(paired.values, predicate, "coarse_mean")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "fine_mean")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improvement_mean")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improvement_ci_low")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improvement_ci_high")}`,
    `='Paired Comparison'!${dataCell(paired.values, predicate, "improved_image_fraction")}`,
  ];
});
summarySheet.getRange("D14:J17").format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("E14:I17").format.numberFormat = "0.0000";
summarySheet.getRange("J14:J17").format.numberFormat = "0.0%";

section("A16:B16", "Tile-boundary audit");
summarySheet.getRange("A17:B19").values = [
  ["Model", "Seam / interior error"],
  ["Coarse control", null],
  ["Fine model", null],
];
summarySheet.getRange("B18:B19").formulas = [
  [`='Reconstruction Audit'!${dataCell(audit.values, (row) => row[0] === "coarse", "seam_to_interior_error_ratio")}`],
  [`='Reconstruction Audit'!${dataCell(audit.values, (row) => row[0] === "fine", "seam_to_interior_error_ratio")}`],
];
summarySheet.getRange("A17:B19").format = {
  fill: "#F8FAFC",
  font: { name: "Aptos" },
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("A17:B17").format.font = { bold: true, color: "#16324F" };
summarySheet.getRange("B18:B19").format.numberFormat = "0.00";

summarySheet.getRange("A21:J21").merge();
summarySheet.getRange("A21").values = [["Interpretation and Step 6 readiness"]];
summarySheet.getRange("A21:J21").format = {
  fill: "#DCEFE7",
  font: { bold: true, color: "#14532D", name: "Aptos" },
};
summarySheet.getRange("A22:J26").merge();
summarySheet.getRange("A22").values = [[
  "Native-resolution segmentation provides a meaningful held-out improvement: mean Dice rises by 0.037 (95% paired bootstrap CI 0.026 to 0.049), and absolute M–A area-fraction error falls by 0.016 (95% CI 0.009 to 0.025). Gains are largest in Regimes I and IV. The model is suitable for a controlled adaptive-fusion experiment, but fusion should account for non-overlapping tile seams: fine-model error is 1.27× higher in a 4-pixel internal seam band, whereas the coarse control shows no seam penalty. This workbook reports Step 5 only; no adaptive fusion was implemented.",
]];
summarySheet.getRange("A22:J26").format = {
  fill: "#F0FDF4",
  font: { color: "#334155", name: "Aptos" },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: "#86B89A" },
};
summarySheet.getRange("A22:J26").format.rowHeight = 24;
summarySheet.getRange("A1:J26").format.columnWidth = 16;
summarySheet.getRange("A:A").format.columnWidth = 27;
summarySheet.getRange("B:B").format.columnWidth = 31;
summarySheet.getRange("D:D").format.columnWidth = 25;

await fs.mkdir(previewRoot, { recursive: true });
const previewRanges = {
  Summary: "A1:J26",
  "Model Summary": "A1:N11",
  "Paired Comparison": "A1:N26",
  "Reconstruction Audit": "A1:G3",
  "Fold Metrics": "A1:N6",
  "Per-image Comparison": "A1:N14",
  "Fine Image Metrics": "A1:N14",
  "Training History": "A1:J18",
  "Tile Metrics": "A1:N14",
  "Tile Manifest": "A1:N14",
  "Fold Tile Manifest": "A1:N14",
  "Run Configuration": "A1:B35",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const rendered = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewRoot, `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await rendered.arrayBuffer()),
  );
}

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
const outputPath = path.join(outputRoot, "fine_model_results.xlsx");
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
