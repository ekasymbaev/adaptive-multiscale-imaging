#!/usr/bin/env node
/** Build a formatted, formula-backed workbook for the coarse-model experiment. */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pathToFileURL } from "node:url";

let artifactTool;
try {
  artifactTool = await import("@oai/artifact-tool");
} catch (error) {
  const runtimeNodeModules = process.env.CODEX_WORKSPACE_NODE_MODULES;
  if (!runtimeNodeModules) throw error;
  artifactTool = await import(
    pathToFileURL(
      path.join(runtimeNodeModules, "@oai", "artifact-tool", "dist", "artifact_tool.mjs"),
    ).href
  );
}
const { FileBlob, SpreadsheetFile, Workbook } = artifactTool;

const scriptPath = fileURLToPath(import.meta.url);
const projectRoot = path.resolve(path.dirname(scriptPath), "..");
const resultsRoot = path.join(projectRoot, "results", "coarse_model");
const previewRoot = process.argv[2] ?? path.join("/tmp", "coarse-model-workbook-preview");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows.filter((values) => values.some((value) => value !== ""));
}

function coerce(value) {
  if (value === "") return null;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) {
    return Number(value);
  }
  return value;
}

async function csvValues(relativePath) {
  const text = await fs.readFile(path.join(projectRoot, relativePath), "utf8");
  const rows = parseCsv(text);
  return rows.map((row, rowIndex) =>
    row.map((value) => (rowIndex === 0 ? value : coerce(value))),
  );
}

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

function formatDataSheet(sheet, values, tableName) {
  const rowCount = values.length;
  const columnCount = values[0].length;
  const finalColumn = columnName(columnCount - 1);
  const used = sheet.getRange(`A1:${finalColumn}${rowCount}`);
  used.values = values;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(3, columnCount));
  const header = sheet.getRange(`A1:${finalColumn}1`);
  header.format = {
    fill: "#16324F",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#16324F" },
  };
  used.format.font = { name: "Aptos", size: 9 };
  used.format.autofitColumns();
  used.format.autofitRows();
  header.format.rowHeight = 46;
  for (let column = 0; column < columnCount; column += 1) {
    const headerName = String(values[0][column]);
    const columnRange = sheet.getRange(
      `${columnName(column)}2:${columnName(column)}${rowCount}`,
    );
    if (/fraction|dice|iou|precision|recall|loss|mean|std|weight|threshold/.test(headerName)) {
      columnRange.format.numberFormat = "0.0000";
    } else if (/seconds/.test(headerName)) {
      columnRange.format.numberFormat = "0.0";
    } else if (/epoch|images|pixels|positive|negative|seed|fold/.test(headerName)) {
      columnRange.format.numberFormat = "0";
    }
  }
  sheet.tables.add(`A1:${finalColumn}${rowCount}`, true, tableName);
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

const foldValues = await csvValues("results/coarse_model/metrics/fold_metrics.csv");
const perImageValues = await csvValues(
  "results/coarse_model/metrics/per_image_test_metrics.csv",
);
const historyValues = await csvValues(
  "results/coarse_model/history/all_folds_history.csv",
);
const coarseValues = await csvValues(
  "data/processed/ma_islands/coarse/coarse_manifest.csv",
);
const summary = JSON.parse(
  await fs.readFile(path.join(resultsRoot, "metrics", "summary.json"), "utf8"),
);
const environment = JSON.parse(
  await fs.readFile(path.join(resultsRoot, "run_environment.json"), "utf8"),
);

const workbook = Workbook.create();
const foldSheet = workbook.worksheets.add("Fold Metrics");
const perImageSheet = workbook.worksheets.add("Per-image Metrics");
const historySheet = workbook.worksheets.add("Training History");
const coarseSheet = workbook.worksheets.add("Coarse Manifest");
const regimeSheet = workbook.worksheets.add("Regime Summary");
const configSheet = workbook.worksheets.add("Run Configuration");
const summarySheet = workbook.worksheets.add("Summary");

formatDataSheet(foldSheet, foldValues, "FoldMetricsTable");
formatDataSheet(perImageSheet, perImageValues, "PerImageMetricsTable");
formatDataSheet(historySheet, historyValues, "TrainingHistoryTable");
formatDataSheet(coarseSheet, coarseValues, "CoarseManifestTable");

regimeSheet.showGridLines = false;
regimeSheet.getRange("A1:E1").values = [[
  "source_regime",
  "mean_dice",
  "mean_iou",
  "mean_precision",
  "mean_recall",
]];
regimeSheet.getRange("A2:A5").values = [["I"], ["II"], ["III"], ["IV"]];
regimeSheet.getRange("B2").formulas = [[
  "=AVERAGEIF('Per-image Metrics'!$Q$2:$Q$41,A2,'Per-image Metrics'!$L$2:$L$41)",
]];
regimeSheet.getRange("B2:B5").fillDown();
regimeSheet.getRange("C2").formulas = [[
  "=AVERAGEIF('Per-image Metrics'!$Q$2:$Q$41,A2,'Per-image Metrics'!$M$2:$M$41)",
]];
regimeSheet.getRange("C2:C5").fillDown();
regimeSheet.getRange("D2").formulas = [[
  "=AVERAGEIF('Per-image Metrics'!$Q$2:$Q$41,A2,'Per-image Metrics'!$N$2:$N$41)",
]];
regimeSheet.getRange("D2:D5").fillDown();
regimeSheet.getRange("E2").formulas = [[
  "=AVERAGEIF('Per-image Metrics'!$Q$2:$Q$41,A2,'Per-image Metrics'!$O$2:$O$41)",
]];
regimeSheet.getRange("E2:E5").fillDown();
regimeSheet.getRange("A1:E1").format = {
  fill: "#16324F",
  font: { bold: true, color: "#FFFFFF" },
};
regimeSheet.getRange("B2:E5").format.numberFormat = "0.0000";
regimeSheet.getRange("A1:E5").format.autofitColumns();
regimeSheet.tables.add("A1:E5", true, "RegimeSummaryTable");
regimeSheet.getRange("B2:B5").conditionalFormats.add("colorScale", {
  colors: ["#FECACA", "#FEF3C7", "#D1FAE5"],
  thresholds: ["min", "50%", "max"],
});

const configValues = [
  ["parameter", "value"],
  ...flattenObject(environment.config),
  ["runtime.device", environment.device],
  ["runtime.python_version", environment.python_version],
  ["runtime.torch_version", environment.torch_version],
  ["runtime.numpy_version", environment.numpy_version],
  ["runtime.opencv_version", environment.opencv_version],
];
formatDataSheet(configSheet, configValues, "RunConfigurationTable");

summarySheet.showGridLines = false;
summarySheet.getRange("A1:H1").merge();
summarySheet.getRange("A1").values = [["M-A Island Coarse U-Net — Step 2 Results"]];
summarySheet.getRange("A1:H1").format = {
  fill: "#16324F",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summarySheet.getRange("A1:H1").format.rowHeight = 34;
summarySheet.getRange("A2:H2").merge();
summarySheet.getRange("A2").values = [[
  "Five-fold image-level out-of-fold evaluation; each original SEM image appears in one held-out test fold.",
]];
summarySheet.getRange("A2:H2").format = {
  fill: "#E8F0F7",
  font: { color: "#334155", italic: true, size: 10 },
  wrapText: true,
};

summarySheet.getRange("A4:B4").values = [["Experiment design", "Value"]];
summarySheet.getRange("A5:B13").values = [
  ["Coarse dimensions", `${summary.coarse_dimensions[0]} x ${summary.coarse_dimensions[1]}`],
  ["Model parameters", summary.model_parameters],
  ["Held-out test images", summary.test_images],
  ["Cross-validation", "5 folds; 24 train / 8 validation / 8 test"],
  ["Loss", "0.5 weighted BCE + 0.5 soft Dice"],
  ["Decision threshold", summary.prediction_threshold],
  ["Checkpoint selection", "Highest validation macro Dice"],
  ["Training device", summary.device],
  ["Total training time (s)", summary.total_training_seconds],
];

summarySheet.getRange("D4:E4").values = [["Macro test metric", "Mean"]];
summarySheet.getRange("D5:D8").values = [["Dice"], ["IoU"], ["Precision"], ["Recall"]];
summarySheet.getRange("E5").formulas = [["=AVERAGE('Per-image Metrics'!L2:L41)"]];
summarySheet.getRange("E6").formulas = [["=AVERAGE('Per-image Metrics'!M2:M41)"]];
summarySheet.getRange("E7").formulas = [["=AVERAGE('Per-image Metrics'!N2:N41)"]];
summarySheet.getRange("E8").formulas = [["=AVERAGE('Per-image Metrics'!O2:O41)"]];

summarySheet.getRange("G4:H4").values = [["Pixel-micro metric", "Value"]];
summarySheet.getRange("G5:G8").values = [["Dice"], ["IoU"], ["Precision"], ["Recall"]];
summarySheet.getRange("H5").formulas = [[
  "=2*SUM('Per-image Metrics'!H2:H41)/(2*SUM('Per-image Metrics'!H2:H41)+SUM('Per-image Metrics'!I2:I41)+SUM('Per-image Metrics'!J2:J41))",
]];
summarySheet.getRange("H6").formulas = [[
  "=SUM('Per-image Metrics'!H2:H41)/(SUM('Per-image Metrics'!H2:H41)+SUM('Per-image Metrics'!I2:I41)+SUM('Per-image Metrics'!J2:J41))",
]];
summarySheet.getRange("H7").formulas = [[
  "=SUM('Per-image Metrics'!H2:H41)/(SUM('Per-image Metrics'!H2:H41)+SUM('Per-image Metrics'!I2:I41))",
]];
summarySheet.getRange("H8").formulas = [[
  "=SUM('Per-image Metrics'!H2:H41)/(SUM('Per-image Metrics'!H2:H41)+SUM('Per-image Metrics'!J2:J41))",
]];

for (const rangeAddress of ["A4:B4", "D4:E4", "G4:H4"]) {
  summarySheet.getRange(rangeAddress).format = {
    fill: "#2F6B8A",
    font: { bold: true, color: "#FFFFFF" },
    borders: { preset: "outside", style: "thin", color: "#2F6B8A" },
  };
}
summarySheet.getRange("A5:B13").format = {
  fill: "#F8FAFC",
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("D5:E8").format = {
  fill: "#F8FAFC",
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("G5:H8").format = {
  fill: "#F8FAFC",
  borders: { preset: "outside", style: "thin", color: "#CBD5E1" },
};
summarySheet.getRange("E5:E8").format.numberFormat = "0.0000";
summarySheet.getRange("H5:H8").format.numberFormat = "0.0000";
summarySheet.getRange("B6:B7").format.numberFormat = "0";
summarySheet.getRange("B10").format.numberFormat = "0.00";
summarySheet.getRange("B13").format.numberFormat = "0.0";

summarySheet.getRange("A16:H16").merge();
summarySheet.getRange("A16").values = [["Interpretation and Step 3 readiness"]];
summarySheet.getRange("A16:H16").format = {
  fill: "#DCEFE7",
  font: { bold: true, color: "#14532D" },
};
summarySheet.getRange("A17:H19").merge();
summarySheet.getRange("A17").values = [[
  "The coarse model is suitable as a first-pass predictor for uncertainty experiments: it achieves useful held-out overlap while leaving substantial, structured errors for uncertainty estimation to identify. Recall exceeds precision, indicating mild over-segmentation. Source regime I is the main weakness (mean Dice about 0.668), whereas regime III is strongest (about 0.905). These findings are exploratory because the dataset contains only 40 original images.",
]];
summarySheet.getRange("A17:H19").format = {
  fill: "#F0FDF4",
  font: { color: "#334155" },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: "#86B89A" },
};
summarySheet.getRange("A17:H19").format.rowHeight = 26;
summarySheet.getRange("A1:H19").format.font.name = "Aptos";
summarySheet.getRange("A1:H19").format.columnWidth = 18;
summarySheet.getRange("A:A").format.columnWidth = 25;
summarySheet.getRange("B:B").format.columnWidth = 25;
summarySheet.getRange("D:D").format.columnWidth = 22;
summarySheet.getRange("G:G").format.columnWidth = 22;
summarySheet.freezePanes.freezeRows(2);

await fs.mkdir(previewRoot, { recursive: true });
for (const sheetName of [
  "Summary",
  "Fold Metrics",
  "Per-image Metrics",
  "Training History",
  "Coarse Manifest",
  "Regime Summary",
  "Run Configuration",
]) {
  const rendered = await workbook.render({
    sheetName,
    autoCrop: "all",
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
  maxChars: 12000,
  tableMaxRows: 6,
  tableMaxCols: 8,
  options: { maxResults: 100 },
});
await fs.writeFile(
  path.join(previewRoot, "inspection.txt"),
  String(inspection.ndjson ?? inspection),
  "utf8",
);

const formulaCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 12000,
});
await fs.writeFile(
  path.join(previewRoot, "formula_errors.txt"),
  String(formulaCheck.ndjson ?? formulaCheck),
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(resultsRoot, "coarse_model_results.xlsx");
await output.save(outputPath);

const exportedBlob = await FileBlob.load(outputPath);
const exportedWorkbook = await SpreadsheetFile.importXlsx(exportedBlob);
for (const sheetName of [
  "Summary",
  "Fold Metrics",
  "Per-image Metrics",
  "Training History",
  "Coarse Manifest",
  "Regime Summary",
  "Run Configuration",
]) {
  const rendered = await exportedWorkbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewRoot, `final_${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await rendered.arrayBuffer()),
  );
}
const exportedFormulaCheck = await exportedWorkbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 12000,
});
await fs.writeFile(
  path.join(previewRoot, "final_formula_errors.txt"),
  String(exportedFormulaCheck.ndjson ?? exportedFormulaCheck),
  "utf8",
);
console.log(`Workbook: ${outputPath}`);
console.log(`Previews: ${previewRoot}`);
