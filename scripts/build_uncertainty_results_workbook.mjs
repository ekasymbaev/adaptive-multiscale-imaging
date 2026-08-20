#!/usr/bin/env node
/** Build and verify the Step 3 MC-dropout uncertainty results workbook. */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function reportFatalError(error) {
  const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  console.error(message);
  if (error instanceof Error && error.stack) {
    for (const line of error.stack.split("\n").filter((item) => item.includes("build_uncertainty_results_workbook"))) {
      console.error(line.trim());
    }
  }
  process.exit(1);
}

process.on("uncaughtException", reportFatalError);
process.on("unhandledRejection", reportFatalError);

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultRoot = path.join(projectRoot, "results", "coarse_uncertainty");
const outputRoot = path.join(projectRoot, "outputs", "step3_coarse_uncertainty");
const previewRoot = process.argv[2] ?? path.join("/tmp", "step3-uncertainty-workbook-preview");

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

function headerColumn(values, headerName) {
  const index = values[0].indexOf(headerName);
  if (index < 0) throw new Error(`Missing column: ${headerName}`);
  return columnName(index);
}

function meanFormula(sheetName, values, headerName) {
  const column = headerColumn(values, headerName);
  return `=AVERAGE('${sheetName}'!${column}2:${column}${values.length})`;
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
  return sheet;
}

function formatDataSheet(sheet, tableName, freezeColumns = 3) {
  const used = sheet.getUsedRange();
  const values = used.values;
  const rowCount = values.length;
  const columnCount = values[0].length;
  const finalColumn = columnName(columnCount - 1);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(freezeColumns, columnCount));
  used.format.font = { name: "Aptos", size: 9 };
  used.format.autofitColumns();
  used.format.autofitRows();
  const header = sheet.getRange(`A1:${finalColumn}1`);
  header.format = {
    fill: "#16324F",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#16324F" },
  };
  header.format.rowHeight = 48;
  for (let column = 0; column < columnCount; column += 1) {
    const name = String(values[0][column]);
    const range = sheet.getRange(
      `${columnName(column)}2:${columnName(column)}${rowCount}`,
    );
    if (/fraction|dice|entropy|variance|correlation|pearson|spearman|auc|precision|recall|capture|enrichment|agreement|mean|rate|ratio|difference|delta|p$/.test(name)) {
      range.format.numberFormat = "0.0000";
    } else if (/seconds/.test(name)) {
      range.format.numberFormat = "0.0";
    } else if (/pixels|images|passes|seed|fold|region|row|column|size/.test(name)) {
      range.format.numberFormat = "#,##0";
    }
  }
  sheet.tables.add(`A1:${finalColumn}${rowCount}`, true, tableName);
  return values;
}

const perImageCsv = await fs.readFile(
  path.join(resultRoot, "metrics", "per_image_uncertainty_metrics.csv"),
  "utf8",
);
const regimeCsv = await fs.readFile(
  path.join(resultRoot, "metrics", "regime_uncertainty_metrics.csv"),
  "utf8",
);
const concentrationCsv = await fs.readFile(
  path.join(resultRoot, "metrics", "uncertainty_concentration.csv"),
  "utf8",
);
const localCsv = await fs.readFile(
  path.join(resultRoot, "metrics", "local_region_metrics.csv"),
  "utf8",
);
const summary = JSON.parse(
  await fs.readFile(path.join(resultRoot, "metrics", "summary.json"), "utf8"),
);

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
// The library's CSV hydration API requires an empty collaborative document,
// so parse the audited local CSV files and populate ranges directly.
const perImageSheet = addCsvSheet(workbook, "Per-image Metrics", perImageCsv);
const regimeSheet = addCsvSheet(workbook, "Regime Metrics", regimeCsv);
const concentrationSheet = addCsvSheet(workbook, "Concentration", concentrationCsv);
const localSheet = addCsvSheet(workbook, "Local Regions", localCsv);
const configSheet = workbook.worksheets.add("Run Configuration");

const perImageValues = formatDataSheet(perImageSheet, "UncertaintyPerImageTable");
const regimeValues = formatDataSheet(regimeSheet, "UncertaintyRegimeTable", 1);
formatDataSheet(concentrationSheet, "UncertaintyConcentrationTable", 3);
formatDataSheet(localSheet, "UncertaintyLocalRegionsTable", 4);

const regimeAucColumn = headerColumn(regimeValues, "entropy_error_roc_auc_image_mean");
const regimeRegionColumn = headerColumn(
  regimeValues,
  "region_entropy_error_pearson_image_mean",
);
regimeSheet
  .getRange(`${regimeAucColumn}2:${regimeAucColumn}${regimeValues.length}`)
  .conditionalFormats.add("colorScale", {
    colors: ["#FECACA", "#FEF3C7", "#D1FAE5"],
    thresholds: ["min", "50%", "max"],
  });
regimeSheet
  .getRange(`${regimeRegionColumn}2:${regimeRegionColumn}${regimeValues.length}`)
  .conditionalFormats.add("colorScale", {
    colors: ["#FECACA", "#FEF3C7", "#D1FAE5"],
    thresholds: ["min", "50%", "max"],
  });

const configValues = [
  ["parameter", "value"],
  ...flattenObject(summary.config),
  ["inference.dropout_modules_enabled", summary.dropout_modules_enabled.join(", ")],
  ["inference.dropout_probability", summary.dropout_probability],
  ["inference.prediction_under_evaluation", summary.prediction_under_evaluation],
  ["inference.ground_truth_use", summary.ground_truth_use],
  ["runtime.device", summary.runtime.device],
  ["runtime.seconds", summary.runtime.seconds],
  ["runtime.python_version", summary.runtime.python_version],
  ["runtime.torch_version", summary.runtime.torch_version],
  ["runtime.numpy_version", summary.runtime.numpy_version],
  ["runtime.opencv_version", summary.runtime.opencv_version],
];
configSheet.getRange(`A1:B${configValues.length}`).values = configValues;
formatDataSheet(configSheet, "UncertaintyConfigurationTable", 1);

summarySheet.showGridLines = false;
summarySheet.freezePanes.freezeRows(2);
summarySheet.getRange("A1:H1").merge();
summarySheet.getRange("A1").values = [[
  "M-A Island Coarse U-Net — Step 3 Uncertainty Results",
]];
summarySheet.getRange("A1:H1").format = {
  fill: "#16324F",
  font: { bold: true, color: "#FFFFFF", size: 18, name: "Aptos Display" },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summarySheet.getRange("A1:H1").format.rowHeight = 34;
summarySheet.getRange("A2:H2").merge();
summarySheet.getRange("A2").values = [[
  "Eight MC-dropout passes on each of 40 held-out images; labels were used only after uncertainty maps were generated.",
]];
summarySheet.getRange("A2:H2").format = {
  fill: "#E8F0F7",
  font: { color: "#334155", italic: true, size: 10, name: "Aptos" },
  wrapText: true,
};

summarySheet.getRange("A4:B4").values = [["Experiment design", "Value"]];
summarySheet.getRange("A5:B12").values = [
  ["Held-out images", summary.held_out_images],
  ["MC passes per image", summary.mc_passes],
  ["Active dropout modules", summary.dropout_modules_enabled[0]],
  ["Dropout probability", summary.dropout_probability],
  ["Prediction evaluated", "Saved deterministic Step 2 mask"],
  ["Local analysis grid", "48 non-overlapping 64 x 64 regions/image"],
  ["Total analyzed pixels", summary.overall.pixels],
  ["Total local regions", summary.local_analysis_grid.total_regions],
];

summarySheet.getRange("D4:E4").values = [["Pixel-level evidence", "Mean"]];
summarySheet.getRange("D5:D11").values = [
  ["Entropy-error correlation"],
  ["Error-detection ROC AUC"],
  ["Entropy error/correct ratio"],
  ["Top 10% error capture"],
  ["Top 20% error capture"],
  ["Top 20% error enrichment"],
  ["Incorrect > correct p-value"],
];
summarySheet.getRange("E5").formulas = [[
  meanFormula("Per-image Metrics", perImageValues, "entropy_error_pearson"),
]];
summarySheet.getRange("E6").formulas = [[
  meanFormula("Per-image Metrics", perImageValues, "entropy_error_roc_auc"),
]];
summarySheet.getRange("E7").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "entropy_incorrect_to_correct_ratio",
  ),
]];
summarySheet.getRange("E8").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "entropy_top_10pct_error_capture",
  ),
]];
summarySheet.getRange("E9").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "entropy_top_20pct_error_capture",
  ),
]];
summarySheet.getRange("E10").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "entropy_top_20pct_error_enrichment",
  ),
]];
summarySheet.getRange("E11").values = [[
  summary.overall.per_image_macro.entropy_incorrect_greater_wilcoxon_p,
]];

summarySheet.getRange("G4:H4").values = [["Local-region evidence", "Mean"]];
summarySheet.getRange("G5:G10").values = [
  ["64 x 64 entropy-error correlation"],
  ["Top 10% region error capture"],
  ["Top 20% region error capture"],
  ["Top 20% region enrichment"],
  ["MC mean minus baseline Dice"],
  ["MC/baseline pixel agreement"],
];
summarySheet.getRange("H5").formulas = [[
  meanFormula("Per-image Metrics", perImageValues, "region_entropy_error_pearson"),
]];
summarySheet.getRange("H6").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "entropy_top_10pct_regions_error_capture",
  ),
]];
summarySheet.getRange("H7").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "entropy_top_20pct_regions_error_capture",
  ),
]];
summarySheet.getRange("H8").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "entropy_top_20pct_regions_error_enrichment",
  ),
]];
summarySheet.getRange("H9").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "mc_minus_deterministic_dice",
  ),
]];
summarySheet.getRange("H10").formulas = [[
  meanFormula(
    "Per-image Metrics",
    perImageValues,
    "deterministic_mc_pixel_agreement",
  ),
]];

for (const rangeAddress of ["A4:B4", "D4:E4", "G4:H4"]) {
  summarySheet.getRange(rangeAddress).format = {
    fill: "#2F6B8A",
    font: { bold: true, color: "#FFFFFF", name: "Aptos" },
    borders: { preset: "outside", style: "thin", color: "#2F6B8A" },
  };
}
for (const rangeAddress of ["A5:B12", "D5:E11", "G5:H10"]) {
  summarySheet.getRange(rangeAddress).format = {
    fill: "#F8FAFC",
    font: { name: "Aptos" },
    borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
  };
}
summarySheet.getRange("E5:E10").format.numberFormat = "0.0000";
summarySheet.getRange("E11").format.numberFormat = "0.00E+00";
summarySheet.getRange("H5:H10").format.numberFormat = "0.0000";
summarySheet.getRange("B5:B7").format.numberFormat = "#,##0";
summarySheet.getRange("B8").format.numberFormat = "0.0";
summarySheet.getRange("B11:B12").format.numberFormat = "#,##0";

summarySheet.getRange("A15:H15").merge();
summarySheet.getRange("A15").values = [["Interpretation and Step 4 readiness"]];
summarySheet.getRange("A15:H15").format = {
  fill: "#DCEFE7",
  font: { bold: true, color: "#14532D", name: "Aptos" },
};
summarySheet.getRange("A16:H19").merge();
summarySheet.getRange("A16").values = [[
  "Predictive entropy meaningfully tracks held-out segmentation errors. The signal is strong at pixel level and remains useful, but weaker, after aggregation into future 64 x 64 coarse-image regions. Regime I is the hardest case: it retains positive error association but captures less error than the other regimes. Raw entropy is not calibrated across images, so future selection should rank regions within each image and must be compared with random selection. These results support a controlled Step 4 tile-selection experiment; they do not yet demonstrate an adaptive-imaging benefit.",
]];
summarySheet.getRange("A16:H19").format = {
  fill: "#F0FDF4",
  font: { color: "#334155", name: "Aptos" },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: "#86B89A" },
};
summarySheet.getRange("A16:H19").format.rowHeight = 25;
summarySheet.getRange("A1:H19").format.columnWidth = 18;
summarySheet.getRange("A:A").format.columnWidth = 27;
summarySheet.getRange("B:B").format.columnWidth = 31;
summarySheet.getRange("D:D").format.columnWidth = 29;
summarySheet.getRange("G:G").format.columnWidth = 31;

await fs.mkdir(previewRoot, { recursive: true });
const previewRanges = {
  Summary: "A1:H19",
  "Per-image Metrics": "A1:L12",
  "Regime Metrics": "A1:L5",
  Concentration: "A1:H20",
  "Local Regions": "A1:L15",
  "Run Configuration": "A1:B25",
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

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 10000,
  tableMaxRows: 5,
  tableMaxCols: 8,
  options: { maxResults: 100 },
});
await fs.writeFile(
  path.join(previewRoot, "inspection.txt"),
  String(inspection.ndjson ?? inspection),
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
const outputPath = path.join(outputRoot, "uncertainty_results.xlsx");
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
// Some artifact-tool versions emit a large inspection sidecar beside imported
// workbooks. It is a transient diagnostic, not a research result.
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });

console.log(`Workbook: ${outputPath}`);
console.log(`Previews: ${previewRoot}`);
