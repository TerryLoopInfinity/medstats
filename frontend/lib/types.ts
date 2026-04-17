// ────────────────────────────────────────────────────────────
// Upload
// ────────────────────────────────────────────────────────────

export interface UploadResponse {
  file_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  preview: (string | number | null)[][];
  warnings: string[];
}

// ────────────────────────────────────────────────────────────
// Analysis results
// ────────────────────────────────────────────────────────────

export interface TableResult {
  title: string;
  headers: string[];
  rows: (string | number | null)[][];
}

export interface ChartResult {
  title: string;
  chart_type: string;
  /** ECharts option JSON — passed directly to the chart component */
  option: Record<string, unknown>;
}

export interface AnalysisResult {
  method: string;
  tables: TableResult[];
  charts: ChartResult[];
  summary: string;
  warnings: string[];
}

// ────────────────────────────────────────────────────────────
// Analysis methods
// ────────────────────────────────────────────────────────────

export type AnalysisMethod =
  | "descriptive"
  | "table_one"
  | "ttest"
  | "hypothesis"
  | "correlation"
  | "linear_reg"
  | "linear_reg_adjusted"
  | "logistic_reg"
  | "survival"
  | "cox_reg"
  | "psm"
  | "prediction"
  | "forest_plot"
  | "rcs"
  | "threshold"
  | "mediation"
  | "sample_size";
