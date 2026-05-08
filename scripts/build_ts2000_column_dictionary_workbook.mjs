#!/usr/bin/env node

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectDir = path.resolve(__dirname, "..");
const ts2000Dir = path.join(projectDir, "data", "external", "ts2000");
const dictionaryDir = path.join(ts2000Dir, "column_dictionary");
const metadataPath = path.join(dictionaryDir, "ts2000_column_dictionary_metadata.json");
const modelManifestPath = path.join(ts2000Dir, "TS2000_Model_V1_Manifest.json");
const outputPath = path.join(dictionaryDir, "TS2000_Credit_Model_Column_Dictionary.xlsx");
const previewDir = path.join(os.tmpdir(), "ts2000_column_dictionary_previews");

const metadata = JSON.parse(await fs.readFile(metadataPath, "utf8"));
const modelManifest = JSON.parse(await fs.readFile(modelManifestPath, "utf8"));

const COLORS = {
  navy: "#1F4E78",
  blue: "#DCE6F1",
  paleBlue: "#F7FAFD",
  paleGray: "#F4F6F8",
  grayBorder: "#D0D7DE",
  darkText: "#1F2933",
  white: "#FFFFFF",
  softGreen: "#EAF7EA",
  softYellow: "#FFF7D6",
  softOrange: "#FFF1E6",
};

const sectionLabelMap = {
  activity: "활동성",
  analysis_label: "분석 전용 라벨",
  audit: "감사 관련",
  business_structure: "사업구조",
  cashflow_solvency: "현금흐름/상환능력",
  financial_bs: "재무상태표",
  financial_cf: "현금흐름표",
  financial_is: "손익계산서",
  growth: "성장성",
  identifier: "식별자",
  macro: "거시경제",
  profile: "기업 프로필",
  profitability: "수익성",
  stability: "안정성",
  target: "타겟",
  time_reference: "시점 기준",
  trend: "추세변수",
};

const usageRoleLabelMap = {
  analysis_only_label: "분석 전용 라벨",
  feature_deferred: "1차 모델 제외 후보",
  feature_x: "입력변수(X)",
  id_reference: "식별 참조",
  target_reference: "타겟 참조",
  target_y: "정답변수(Y)",
  time_reference: "시점 참조",
};

const modelSetRoleLabelMap = {
  analysis_only_label: "분석 전용 라벨",
  audit_extension: "감사 확장셋 전용",
  feature_deferred: "보류 후보",
  feature_x: "기본셋 포함",
  id_reference: "식별 참조",
  target_reference: "타겟 참조",
  target_y: "정답변수(Y)",
  time_reference: "시점 참조",
};

const thematicGroupLabelMap = {
  activity: "활동성",
  audit: "감사",
  business_structure: "사업구조",
  cashflow_solvency: "현금흐름/상환능력",
  growth: "성장성",
  identifier: "식별자",
  macro: "거시",
  profitability: "수익성",
  profile: "기업 프로필",
  stability: "안정성",
  time_reference: "시점 참조",
  target: "타겟",
  trend: "추세",
};

const thematicSubgroupLabelMap = {
  audit_extension_candidate: "감사의견 범주 확장 후보",
  macro_core: "매크로 기본셋",
  macro_rate_level: "금리 레벨 확장셋",
  working_capital_level: "운전자본 수준",
  working_capital_trend: "운전자본 악화 추세",
  trend_delta_change: "전년 변화량/변화폭",
  trend_lag: "lag 변수",
  trend_transition_flag: "전환 더미",
  trend_persistence_flag: "연속 악화 더미",
  trend_volatility_cv: "변동성/CV",
  trend_earnings_quality_capital: "이익품질/자본구조 추세",
};

function colLabel(colNumber) {
  let n = colNumber;
  let label = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    label = String.fromCharCode(65 + remainder) + label;
    n = Math.floor((n - 1) / 26);
  }
  return label;
}

function rangeRef(startRow, startCol, rowCount = 1, colCount = 1) {
  const start = `${colLabel(startCol)}${startRow}`;
  const end = `${colLabel(startCol + colCount - 1)}${startRow + rowCount - 1}`;
  return rowCount === 1 && colCount === 1 ? start : `${start}:${end}`;
}

function writeMatrix(sheet, startRow, startCol, matrix) {
  const rowCount = matrix.length;
  const colCount = matrix[0]?.length ?? 1;
  const range = sheet.getRange(rangeRef(startRow, startCol, rowCount, colCount));
  range.values = matrix;
  return range;
}

function setFill(range, color) {
  range.format.fill.color = color;
}

function setFont(range, fontConfig = {}) {
  const { bold, italic, size, name, color } = fontConfig;
  if (typeof bold === "boolean") range.format.font.bold = bold;
  if (typeof italic === "boolean") range.format.font.italic = italic;
  if (typeof size === "number") range.format.font.size = size;
  if (typeof name === "string") range.format.font.name = name;
  if (typeof color === "string") range.format.font.color = color;
}

function setBorders(range, { color = COLORS.grayBorder, weight = 1, style = "solid" } = {}) {
  const edges = [
    range.format.borders.top,
    range.format.borders.bottom,
    range.format.borders.left,
    range.format.borders.right,
    range.format.borders.insideHorizontal,
    range.format.borders.insideVertical,
  ];
  for (const edge of edges) {
    edge.style = style;
    edge.weight = weight;
    edge.color = color;
  }
}

function setColumnWidthPx(sheet, columnLetter, widthPx) {
  sheet.getRange(`${columnLetter}:${columnLetter}`).format.columnWidthPx = widthPx;
}

function formatHeaderRow(range) {
  setFill(range, COLORS.navy);
  setFont(range, { bold: true, size: 10, color: COLORS.white });
  range.format.wrapText = true;
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  setBorders(range, { color: COLORS.navy, weight: 1 });
}

function formatDataRange(range, { wrapText = false, fillColor = null } = {}) {
  if (fillColor) setFill(range, fillColor);
  setFont(range, { size: 10, color: COLORS.darkText });
  range.format.wrapText = wrapText;
  range.format.verticalAlignment = "top";
  setBorders(range, { color: COLORS.grayBorder, weight: 1 });
}

function formatTitleBlock(sheet, mergeRangeRef, cellRef, text, { size = 16 } = {}) {
  const range = sheet.getRange(mergeRangeRef);
  range.merge();
  sheet.getRange(cellRef).values = [[text]];
  setFont(range, { bold: true, size, color: COLORS.navy });
  range.format.horizontalAlignment = "left";
  range.format.verticalAlignment = "center";
  return range;
}

function formatSubtitleBlock(sheet, mergeRangeRef, cellRef, text) {
  const range = sheet.getRange(mergeRangeRef);
  range.merge();
  sheet.getRange(cellRef).values = [[text]];
  setFont(range, { size: 10, color: COLORS.darkText });
  range.format.wrapText = true;
  range.format.horizontalAlignment = "left";
  range.format.verticalAlignment = "top";
  return range;
}

function reverseGroupMaps() {
  const thematicGroupByColumn = new Map();
  const thematicSubgroupByColumn = new Map();

  for (const [subgroup, columns] of Object.entries(modelManifest.macro_feature_groups ?? {})) {
    for (const column of columns) {
      thematicGroupByColumn.set(column, "macro");
      thematicSubgroupByColumn.set(column, subgroup);
    }
  }
  for (const [subgroup, columns] of Object.entries(modelManifest.activity_feature_groups ?? {})) {
    for (const column of columns) {
      thematicGroupByColumn.set(column, "activity");
      thematicSubgroupByColumn.set(column, subgroup);
    }
  }
  for (const [subgroup, columns] of Object.entries(modelManifest.trend_feature_groups ?? {})) {
    for (const column of columns) {
      thematicGroupByColumn.set(column, "trend");
      thematicSubgroupByColumn.set(column, subgroup);
    }
  }
  for (const column of modelManifest.audit_extension_columns ?? []) {
    thematicGroupByColumn.set(column, "audit");
    thematicSubgroupByColumn.set(column, "audit_extension_candidate");
  }
  return { thematicGroupByColumn, thematicSubgroupByColumn };
}

function buildClassificationRows() {
  const baselineFeatures = new Set(modelManifest.feature_columns ?? []);
  const auditExpandedFeatures = new Set(modelManifest.audit_expanded_feature_columns ?? []);
  const deferredFeatures = new Set(modelManifest.deferred_feature_columns ?? []);
  const auditExtensions = new Set(modelManifest.audit_extension_columns ?? []);
  const { thematicGroupByColumn, thematicSubgroupByColumn } = reverseGroupMaps();

  return metadata.columns.map((row) => {
    const modelSetRole = auditExtensions.has(row.variable_name)
      ? "audit_extension"
      : row.usage_role;
    const thematicGroup = thematicGroupByColumn.get(row.variable_name) ?? row.section;
    const thematicSubgroup = thematicSubgroupByColumn.get(row.variable_name) ?? "";
    return {
      ...row,
      model_set_role: modelSetRole,
      baseline_included: baselineFeatures.has(row.variable_name) ? 1 : 0,
      audit_expanded_included: auditExpandedFeatures.has(row.variable_name) ? 1 : 0,
      deferred_included: deferredFeatures.has(row.variable_name) ? 1 : 0,
      thematic_group: thematicGroup,
      thematic_subgroup: thematicSubgroup,
    };
  });
}

function buildSummarySheet(workbook) {
  const sheet = workbook.worksheets.add("요약");

  formatTitleBlock(sheet, "A1:J1", "A1", "TS2000 신용위험 모델 컬럼 사전", { size: 18 });
  formatSubtitleBlock(
    sheet,
    "A2:J2",
    "A2",
    "최종 모델링 파일과 baseline/확장셋 분류 기준을 함께 담은 설명서입니다. 워크북 안에서 컬럼 의미, 모델 포함 여부, 추세 세부 그룹까지 한 번에 확인할 수 있도록 정리했습니다.",
  );

  writeMatrix(sheet, 4, 1, [["항목", "내용"]]);
  formatHeaderRow(sheet.getRange("A4:B4"));

  const summaryRows = [
    ["[전체] 파일 경로", "data/external/ts2000/TS2000_Credit_Model_Dataset.csv"],
    ["[기본셋] 파일 경로", "data/external/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv"],
    ["[감사 확장셋] 파일 경로", "data/external/ts2000/TS2000_Credit_Model_Dataset_Model_V1_Audit_Expanded.csv"],
    ["[학습 manifest] 파일 경로", "data/external/ts2000/TS2000_Model_V1_Manifest.json"],
    ["전체 컬럼 수", metadata.summary.full_column_count],
    ["기본셋 컬럼 수", metadata.summary.model_column_count],
    ["감사 확장셋 컬럼 수", metadata.summary.audit_expanded_column_count],
    ["기본 feature 수", metadata.summary.model_feature_count],
    ["감사 확장 feature 수", metadata.summary.audit_expanded_feature_count],
    ["행 수", metadata.summary.row_count],
    ["회계연도 범위", metadata.summary.fiscal_year_range],
    ["타겟 컬럼", metadata.summary.target_column],
    ["타겟 매핑", metadata.summary.target_mapping],
    ["병합 기준", metadata.summary.join_rule],
    ["누수 방지 메모", metadata.summary.leakage_note],
    ["데이터셋 운용 메모", metadata.summary.dataset_variant_note],
  ];
  writeMatrix(sheet, 5, 1, summaryRows);
  formatDataRange(sheet.getRange(`A5:B${4 + summaryRows.length}`), { wrapText: true });

  writeMatrix(sheet, 4, 4, [["usage_role", "설명", "개수"]]);
  formatHeaderRow(sheet.getRange("D4:F4"));
  const roleRows = metadata.role_counts.map((row) => [
    row.usage_role,
    usageRoleLabelMap[row.usage_role] ?? row.usage_role,
    row.count,
  ]);
  writeMatrix(sheet, 5, 4, roleRows);
  formatDataRange(sheet.getRange(`D5:F${4 + roleRows.length}`), { wrapText: true });

  writeMatrix(sheet, 4, 8, [["section", "설명", "개수"]]);
  formatHeaderRow(sheet.getRange("H4:J4"));
  const sectionRows = metadata.section_counts.map((row) => [
    row.section,
    sectionLabelMap[row.section] ?? row.section,
    row.count,
  ]);
  writeMatrix(sheet, 5, 8, sectionRows);
  formatDataRange(sheet.getRange(`H5:J${4 + sectionRows.length}`), { wrapText: true });

  writeMatrix(sheet, 24, 4, [["추천 실험 순서", "설명"]]);
  formatHeaderRow(sheet.getRange("D24:E24"));
  const experimentRows = [
    ["Model V1 baseline", "audit_opinion_category 없이 감사 flag 4개만 유지하는 기본 실험"],
    ["Model V1 + audit_opinion_category", "감사의견 범주형 자체가 추가 설명력을 주는지 비교"],
    ["Model V1 + deferred 일부", "보류변수를 하나씩 재투입해 성능과 해석가능성을 비교"],
  ];
  writeMatrix(sheet, 25, 4, experimentRows);
  formatDataRange(sheet.getRange(`D25:E${24 + experimentRows.length}`), {
    wrapText: true,
    fillColor: COLORS.softGreen,
  });

  writeMatrix(sheet, 24, 8, [["분류 읽는 법", "설명"]]);
  formatHeaderRow(sheet.getRange("H24:I24"));
  const readingRows = [
    ["baseline_included", "1이면 기본셋 CSV에서 바로 학습 후보로 사용"],
    ["audit_expanded_included", "1이면 감사 확장셋 CSV에 포함"],
    ["deferred_included", "1이면 보류변수 비교 파일에서 확인"],
    ["thematic_subgroup", "발표/문서용 세부 그룹. 특히 activity, trend에서 유용"],
  ];
  writeMatrix(sheet, 25, 8, readingRows);
  formatDataRange(sheet.getRange(`H25:I${24 + readingRows.length}`), {
    wrapText: true,
    fillColor: COLORS.softYellow,
  });

  sheet.freezePanes.freezeRows(4);

  setColumnWidthPx(sheet, "A", 150);
  setColumnWidthPx(sheet, "B", 430);
  setColumnWidthPx(sheet, "D", 170);
  setColumnWidthPx(sheet, "E", 220);
  setColumnWidthPx(sheet, "F", 80);
  setColumnWidthPx(sheet, "H", 170);
  setColumnWidthPx(sheet, "I", 220);
  setColumnWidthPx(sheet, "J", 80);

  sheet.getRange("A1:J32").format.autofitRows();
  return sheet;
}

function buildDictionarySheet(workbook, classificationRows) {
  const sheet = workbook.worksheets.add("컬럼사전");
  const headers = [
    "order",
    "variable_name",
    "section",
    "usage_role",
    "model_set_role",
    "thematic_group",
    "thematic_subgroup",
    "korean_name",
    "description",
    "formula_or_logic",
    "unit",
    "note",
  ];

  formatTitleBlock(sheet, "A1:L1", "A1", "TS2000_Credit_Model_Dataset 컬럼 설명");
  formatSubtitleBlock(
    sheet,
    "A2:L2",
    "A2",
    "기존 변수 사전에 baseline/확장셋 분류와 문서용 세부 그룹을 추가했습니다. variable_name은 코드용 변수명, model_set_role은 기본셋/보류셋 위치, thematic_subgroup은 발표용 분류입니다.",
  );

  writeMatrix(sheet, 4, 1, [headers]);
  formatHeaderRow(sheet.getRange("A4:L4"));

  const dataRows = classificationRows.map((row) => [
    row.order,
    row.variable_name,
    row.section,
    row.usage_role,
    row.model_set_role,
    row.thematic_group,
    row.thematic_subgroup,
    row.korean_name,
    row.description,
    row.formula_or_logic,
    row.unit,
    row.note,
  ]);
  writeMatrix(sheet, 5, 1, dataRows);
  formatDataRange(sheet.getRange(`A5:L${4 + dataRows.length}`), { wrapText: true });

  const table = sheet.tables.add(rangeRef(4, 1, dataRows.length + 1, headers.length), true);
  table.name = "ts2000_column_dictionary";

  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(2);

  const widths = {
    A: 60,
    B: 210,
    C: 120,
    D: 140,
    E: 150,
    F: 110,
    G: 180,
    H: 150,
    I: 220,
    J: 260,
    K: 90,
    L: 220,
  };
  for (const [columnLetter, widthPx] of Object.entries(widths)) {
    setColumnWidthPx(sheet, columnLetter, widthPx);
  }

  sheet.getRange(`A1:L${4 + dataRows.length}`).format.autofitRows();
  return sheet;
}

function buildModelSetSheet(workbook, classificationRows) {
  const sheet = workbook.worksheets.add("모델셋 분류");
  const featureRows = classificationRows.filter((row) =>
    ["feature_x", "feature_deferred", "audit_extension"].includes(row.model_set_role),
  );

  formatTitleBlock(sheet, "A1:J1", "A1", "기본셋/확장셋/보류셋 분류");
  formatSubtitleBlock(
    sheet,
    "A2:J2",
    "A2",
    "1차 모델 실험 순서를 염두에 두고 feature만 따로 모았습니다. audit_opinion_category는 baseline에서 제외하고 audit 확장셋에서만 비교하도록 표시했습니다.",
  );

  writeMatrix(sheet, 4, 1, [
    [
      "variable_name",
      "korean_name",
      "model_set_role",
      "baseline_included",
      "audit_expanded_included",
      "deferred_included",
      "thematic_group",
      "thematic_subgroup",
      "usage_note",
    ],
  ]);
  formatHeaderRow(sheet.getRange("A4:I4"));

  const rows = featureRows.map((row) => [
    row.variable_name,
    row.korean_name,
    modelSetRoleLabelMap[row.model_set_role] ?? row.model_set_role,
    row.baseline_included,
    row.audit_expanded_included,
    row.deferred_included,
    thematicGroupLabelMap[row.thematic_group] ?? row.thematic_group,
    thematicSubgroupLabelMap[row.thematic_subgroup] ?? row.thematic_subgroup,
    row.note,
  ]);
  writeMatrix(sheet, 5, 1, rows);
  formatDataRange(sheet.getRange(`A5:I${4 + rows.length}`), { wrapText: true });

  const table = sheet.tables.add(rangeRef(4, 1, rows.length + 1, 9), true);
  table.name = "ts2000_model_set_classification";

  writeMatrix(sheet, 5, 11, [["실험 메모"]]);
  formatHeaderRow(sheet.getRange("K5:L5"));
  const noteRows = [
    ["기본셋", "TS2000_Credit_Model_Dataset_Model_V1.csv를 그대로 사용합니다."],
    ["감사 확장셋", "audit_opinion_category만 추가된 비교 실험입니다."],
    ["활동성 분류", "working_capital_level vs working_capital_trend로 문서화하면 설명이 깔끔합니다."],
    ["추세 분류", "delta/lag/전환 더미/연속 악화/CV/이익품질·자본구조로 쪼개서 설명하면 좋습니다."],
  ];
  writeMatrix(sheet, 6, 11, noteRows);
  formatDataRange(sheet.getRange(`K6:L${5 + noteRows.length}`), {
    wrapText: true,
    fillColor: COLORS.softOrange,
  });

  sheet.freezePanes.freezeRows(4);
  setColumnWidthPx(sheet, "A", 210);
  setColumnWidthPx(sheet, "B", 170);
  setColumnWidthPx(sheet, "C", 140);
  setColumnWidthPx(sheet, "D", 95);
  setColumnWidthPx(sheet, "E", 115);
  setColumnWidthPx(sheet, "F", 95);
  setColumnWidthPx(sheet, "G", 110);
  setColumnWidthPx(sheet, "H", 170);
  setColumnWidthPx(sheet, "I", 250);
  setColumnWidthPx(sheet, "K", 120);
  setColumnWidthPx(sheet, "L", 300);

  sheet.getRange(`A1:L${5 + noteRows.length}`).format.autofitRows();
  return sheet;
}

function buildCodeSheet(workbook) {
  const sheet = workbook.worksheets.add("코드값 참고");

  formatTitleBlock(sheet, "A1:G1", "A1", "주요 코드값 및 해석 참고표");
  formatSubtitleBlock(
    sheet,
    "A2:G2",
    "A2",
    "타겟, 시장구분, 기업규모 그룹, 산업 대분류, 감사의견, 상장폐지 분석용 라벨의 코드 의미를 빠르게 확인할 수 있도록 정리했습니다.",
  );

  writeMatrix(sheet, 4, 1, [["field_name", "code_or_value", "meaning", "note"]]);
  formatHeaderRow(sheet.getRange("A4:D4"));

  const codeRows = metadata.code_values.map((row) => [
    row.field_name,
    row.code_or_value,
    row.meaning,
    row.note,
  ]);
  writeMatrix(sheet, 5, 1, codeRows);
  formatDataRange(sheet.getRange(`A5:D${4 + codeRows.length}`), { wrapText: true });

  const table = sheet.tables.add(rangeRef(4, 1, codeRows.length + 1, 4), true);
  table.name = "ts2000_code_values";

  writeMatrix(sheet, 5, 6, [["추천 사용법"]]);
  formatHeaderRow(sheet.getRange("F5:G5"));
  const usageNotes = [
    ["기본 학습", "TS2000_Credit_Model_Dataset_Model_V1.csv + feature_columns를 우선 사용합니다."],
    ["감사 비교", "audit_opinion_category는 감사 확장셋에서만 추가 비교합니다."],
    ["보류 변수", "feature_deferred 분류는 metadata/manifest에서만 참고하고 기본 baseline에는 넣지 않습니다."],
    ["분석 전용", "analysis_* 컬럼은 사후분석용 라벨이며 모델 입력에 넣지 않습니다."],
  ];
  writeMatrix(sheet, 6, 6, usageNotes);
  formatDataRange(sheet.getRange(`F6:G${5 + usageNotes.length}`), {
    wrapText: true,
    fillColor: COLORS.softYellow,
  });

  sheet.freezePanes.freezeRows(4);
  setColumnWidthPx(sheet, "A", 180);
  setColumnWidthPx(sheet, "B", 130);
  setColumnWidthPx(sheet, "C", 210);
  setColumnWidthPx(sheet, "D", 170);
  setColumnWidthPx(sheet, "F", 120);
  setColumnWidthPx(sheet, "G", 310);

  sheet.getRange(`A1:G${5 + usageNotes.length}`).format.autofitRows();
  return sheet;
}

await fs.mkdir(dictionaryDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const classificationRows = buildClassificationRows();

buildSummarySheet(workbook);
buildDictionarySheet(workbook, classificationRows);
buildModelSetSheet(workbook, classificationRows);
buildCodeSheet(workbook);

const summaryCheck = await workbook.inspect({
  kind: "table",
  range: "요약!A1:J30",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 10,
});
console.log("=== SUMMARY CHECK ===");
console.log(summaryCheck.ndjson);

const modelSetCheck = await workbook.inspect({
  kind: "table",
  range: "모델셋 분류!A1:I12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 9,
});
console.log("=== MODEL SET CHECK ===");
console.log(modelSetCheck.ndjson);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log("=== ERROR SCAN ===");
console.log(errorScan.ndjson);

const renderTargets = [
  { sheetName: "요약", fileName: "summary.png" },
  { sheetName: "컬럼사전", range: "A1:L18", fileName: "dictionary_top.png" },
  { sheetName: "모델셋 분류", range: "A1:L20", fileName: "model_set.png" },
  { sheetName: "코드값 참고", range: "A1:G18", fileName: "code_values.png" },
];

for (const target of renderTargets) {
  const blob = await workbook.render({
    sheetName: target.sheetName,
    range: target.range,
    scale: 2,
  });
  const imageBuffer = Buffer.from(await blob.arrayBuffer());
  await fs.writeFile(path.join(previewDir, target.fileName), imageBuffer);
}

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

console.log(outputPath);
console.log(previewDir);
