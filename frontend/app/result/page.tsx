"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { EChartsRenderer } from "@/components/charts/EChartsRenderer";
import { ForestPlot } from "@/components/charts/ForestPlot";
import type { AnalysisResult, ChartResult, TableResult } from "@/lib/types";

// ── 生存分析 / Cox 回归需要可折叠生存表和 PH 警告行，扩展 props ──
type CollapsibleTableProps = { table: TableResult; defaultOpen?: boolean; highlightKeyword?: string };
type SurvivalResultProps = { result: AnalysisResult };

const METHOD_LABELS: Record<string, string> = {
  descriptive:          "统计描述 & 正态性检验",
  table_one:            "三线表生成",
  ttest:                "差异性分析",
  hypothesis:           "假设检验",
  correlation:          "相关分析",
  linear_reg:           "线性回归",
  linear_reg_adjusted:  "线性回归 — 控制混杂偏倚",
  logistic_reg:         "Logistic 回归分析",
  logistic_reg_adjusted: "Logistic 回归 — 控制混杂偏倚",
  survival:             "生存分析（Kaplan-Meier）",
  cox_reg:              "Cox 比例风险回归",
  psm:                  "倾向性得分匹配（PSM）",
  prediction:           "临床预测模型",
  forest_plot:          "亚组分析 & 森林图",
  rcs:                  "RCS 曲线（限制性立方样条）",
  threshold:            "阈值效应分析",
};

export default function ResultPage() {
  const router = useRouter();

  // 惰性初始化：从 localStorage 直接读取，避免在 effect 中同步调 setState
  const [result] = useState<AnalysisResult | null>(() => {
    try {
      const raw = globalThis.localStorage?.getItem("ms_result");
      return raw ? (JSON.parse(raw) as AnalysisResult) : null;
    } catch {
      return null;
    }
  });

  // 直方图 + QQ 图配对（由 result 派生，用 useMemo 缓存）
  const chartPairs = useMemo<[ChartResult, ChartResult][]>(() => {
    if (!result) return [];
    const pairs: [ChartResult, ChartResult][] = [];
    for (let i = 0; i < result.charts.length - 1; i += 2) {
      pairs.push([result.charts[i], result.charts[i + 1]]);
    }
    return pairs;
  }, [result]);

  if (!result) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-10">
        <div className="rounded-xl border border-border p-8 text-center space-y-3">
          <p className="text-muted-foreground">尚无分析结果</p>
          <a
            href="/upload"
            className="inline-block px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90"
          >
            返回上传
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-10">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">
            {METHOD_LABELS[result.method] ?? result.method}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{result.summary}</p>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            onClick={() => router.push("/analyze")}
            className="px-3 py-1.5 border border-border rounded-lg text-sm hover:bg-muted/50 transition-colors"
          >
            重新分析
          </button>
          <button
            onClick={() => exportJson(result)}
            className="px-3 py-1.5 border border-border rounded-lg text-sm hover:bg-muted/50 transition-colors"
          >
            导出 JSON
          </button>
        </div>
      </div>

      {/* Warnings */}
      {result.warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 space-y-1">
          {result.warnings.map((w, i) => (
            <p key={i} className="text-sm text-amber-700">
              ⚠ {w}
            </p>
          ))}
        </div>
      )}

      {/* Tables */}
      <section className="space-y-6">
        {(result.method === "survival" || result.method === "cox_reg")
          ? <SurvivalCoxTables result={result} />
          : result.method === "psm"
          ? <PSMTables result={result} />
          : result.tables.map((t) => (
              <ResultTable
                key={t.title}
                table={t}
                highlightKeyword={
                  result.method === "linear_reg_adjusted" || result.method === "logistic_reg_adjusted"
                    ? "★"
                    : undefined
                }
              />
            ))
        }
      </section>

      {/* Charts */}
      {result.charts.length > 0 && (
        <section className="space-y-8">
          <h2 className="text-lg font-semibold border-b border-border pb-2">图表</h2>
          {result.method === "descriptive" ? (
            /* 统计描述：直方图 + QQ 图配对展示 */
            chartPairs.map(([hist, qq], i) => (
              <div key={i} className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-xl border border-border p-4">
                  <EChartsRenderer option={hist.option} height={300} />
                </div>
                <div className="rounded-xl border border-border p-4">
                  <EChartsRenderer option={qq.option} height={300} />
                </div>
              </div>
            ))
          ) : result.method === "correlation" ? (
            /* 相关分析：热力图全宽，散点图两列并排 */
            <div className="space-y-4">
              {result.charts.map((chart, i) => (
                <div
                  key={i}
                  className={`rounded-xl border border-border p-4 ${
                    chart.chart_type === "heatmap" ? "w-full" : ""
                  }`}
                >
                  <EChartsRenderer
                    option={chart.option}
                    height={chart.chart_type === "heatmap" ? 420 : 320}
                  />
                </div>
              ))}
              {/* 散点图改为两列 */}
              {result.charts.filter(c => c.chart_type === "scatter").length > 1 && (
                <p className="text-xs text-muted-foreground -mt-2">* 散点图已在上方独立展示</p>
              )}
            </div>
          ) : result.method === "logistic_reg" ? (
            /* Logistic 回归：ROC 全宽 → 概率分布+混淆矩阵并排 → 森林图全宽 → 校准曲线全宽 */
            <LogisticRegCharts charts={result.charts} />
          ) : result.method === "linear_reg_adjusted" ? (
            /* 线性回归控制混杂：森林图全宽，条形图全宽 */
            <div className="space-y-6">
              {result.charts.map((chart, i) => (
                <div key={i} className="rounded-xl border border-border p-4">
                  {chart.chart_type === "forest_plot" ? (
                    <ForestPlot
                      option={chart.option}
                      height={Math.max(
                        320,
                        ((chart.option.forestData as unknown[])?.length ?? 4) * 52 + 80
                      )}
                    />
                  ) : (
                    <EChartsRenderer option={chart.option} height={320} />
                  )}
                </div>
              ))}
            </div>
          ) : result.method === "logistic_reg_adjusted" ? (
            /* Logistic 回归控制混杂：森林图全宽，条形图全宽 */
            <LogisticRegAdjustedCharts charts={result.charts} />
          ) : result.method === "survival" ? (
            /* 生存分析：KM 曲线全宽，累积风险图全宽 */
            <SurvivalCharts charts={result.charts} />
          ) : result.method === "cox_reg" ? (
            /* Cox 回归：HR 森林图全宽，其余两列 */
            <CoxRegCharts charts={result.charts} />
          ) : result.method === "psm" ? (
            /* PSM：Love plot 全宽，PS 分布并排，SMD 条形图全宽，KM 全宽 */
            <PSMCharts charts={result.charts} />
          ) : result.method === "prediction" ? (
            /* 临床预测模型：Nomogram 全宽 → ROC/校准并排 → DCA 全宽 → Bootstrap 直方图 */
            <PredictionCharts charts={result.charts} />
          ) : result.method === "forest_plot" ? (
            /* 亚组分析森林图：全宽展示 */
            <ForestPlotCharts charts={result.charts} />
          ) : result.method === "rcs" ? (
            /* RCS 曲线：全宽，含 rug plot + 直方图 */
            <RCSCharts charts={result.charts} />
          ) : result.method === "threshold" ? (
            /* 阈值效应：全宽效应图 + 对数似然曲线（可折叠） */
            <ThresholdCharts charts={result.charts} />
          ) : (
            /* 其他方法（差异性分析、线性回归等）：两列并排 */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {result.charts.map((chart, i) => (
                <div key={i} className="rounded-xl border border-border p-4">
                  <EChartsRenderer option={chart.option} height={320} />
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Logistic 回归图表布局
// ─────────────────────────────────────────────────────────────────────────────

function LogisticRegCharts({ charts }: { charts: ChartResult[] }) {
  const roc      = charts.find((c) => c.title === "ROC 曲线");
  const hist     = charts.find((c) => c.title === "预测概率分布");
  const cm       = charts.find((c) => c.title === "混淆矩阵");
  const forest   = charts.find((c) => c.chart_type === "forest_plot");
  const calib    = charts.find((c) => c.title === "校准曲线");

  const forestRowCount =
    ((forest?.option.forestData as unknown[])?.length ?? 4);

  return (
    <div className="space-y-6">
      {/* ROC 全宽 */}
      {roc && (
        <div className="rounded-xl border border-border p-4">
          <EChartsRenderer option={roc.option} height={420} />
        </div>
      )}

      {/* 预测概率分布 + 混淆矩阵并排 */}
      {(hist || cm) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {hist && (
            <div className="rounded-xl border border-border p-4">
              <EChartsRenderer option={hist.option} height={300} />
            </div>
          )}
          {cm && (
            <div className="rounded-xl border border-border p-4">
              <EChartsRenderer option={cm.option} height={300} />
            </div>
          )}
        </div>
      )}

      {/* OR 森林图全宽 */}
      {forest && (
        <div className="rounded-xl border border-border p-4">
          <ForestPlot
            option={forest.option}
            height={Math.max(320, forestRowCount * 52 + 80)}
          />
        </div>
      )}

      {/* 校准曲线全宽 */}
      {calib && (
        <div className="rounded-xl border border-border p-4">
          <EChartsRenderer option={calib.option} height={320} />
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Logistic 回归控制混杂图表布局
// ─────────────────────────────────────────────────────────────────────────────

function LogisticRegAdjustedCharts({ charts }: { charts: ChartResult[] }) {
  const forestCharts = charts.filter((c) => c.chart_type === "forest_plot");
  const barCharts = charts.filter((c) => c.chart_type === "bar");

  return (
    <div className="space-y-6">
      {/* OR 森林图（模型对比 + 分层分析）全宽 */}
      {forestCharts.map((chart, i) => (
        <div key={i} className="rounded-xl border border-border p-4">
          <ForestPlot
            option={chart.option}
            height={Math.max(320, ((chart.option.forestData as unknown[])?.length ?? 4) * 52 + 80)}
          />
        </div>
      ))}
      {/* AUC 条形图 + 协变量贡献图并排（如有两个）*/}
      {barCharts.length === 2 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {barCharts.map((chart, i) => (
            <div key={i} className="rounded-xl border border-border p-4">
              <EChartsRenderer option={chart.option} height={300} />
            </div>
          ))}
        </div>
      ) : (
        barCharts.map((chart, i) => (
          <div key={i} className="rounded-xl border border-border p-4">
            <EChartsRenderer option={chart.option} height={320} />
          </div>
        ))
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ResultTable 组件
// ─────────────────────────────────────────────────────────────────────────────

function ResultTable({
  table,
  highlightKeyword,
  phWarningCol,
  pHighlightCol,
}: {
  table: TableResult;
  highlightKeyword?: string;
  /** 列名：该列值含 "⚠ 违反" 时整行标红（Cox PH 假设检验） */
  phWarningCol?: string;
  /** 列名：强制对该列进行 p 值着色（默认列名为 "p 值"） */
  pHighlightCol?: string;
}) {
  const pValIdx = table.headers.indexOf(pHighlightCol ?? "p 值");
  const phColIdx = phWarningCol ? table.headers.indexOf(phWarningCol) : -1;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">{table.title}</h2>
        <button
          onClick={() => exportCsv(table)}
          className="px-3 py-1 border border-border rounded-lg text-xs text-muted-foreground hover:bg-muted/50 transition-colors shrink-0"
        >
          导出 CSV
        </button>
      </div>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40">
            <tr>
              {table.headers.map((h) => (
                <th
                  key={h}
                  className="px-3 py-2.5 text-left font-semibold whitespace-nowrap border-b-2 border-border"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, i) => {
              // 分类变量标题行：第一列有值，但统计列全为空
              const isCategoryHeader =
                row[0] !== "" &&
                row[0] !== null &&
                !String(row[0]).startsWith("  ") &&
                row.slice(1, -2).every((c) => c === "" || c === null);

              // 暴露变量高亮行（含 ★ 标记）
              const isHighlightRow =
                highlightKeyword &&
                typeof row[0] === "string" &&
                String(row[0]).includes(highlightKeyword);

              // PH 违反行（结论列含 "⚠ 违反"）
              const isPHViolation =
                phColIdx >= 0 &&
                typeof row[phColIdx] === "string" &&
                String(row[phColIdx]).includes("⚠");

              return (
                <tr
                  key={i}
                  className={`border-b border-border last:border-0 transition-colors ${
                    isPHViolation
                      ? "bg-destructive/5 border-l-2 border-l-destructive"
                      : isHighlightRow
                      ? "bg-primary/5 border-l-2 border-l-primary"
                      : isCategoryHeader
                      ? "bg-muted/20"
                      : "hover:bg-muted/20"
                  }`}
                >
                  {row.map((cell, j) => {
                    // p 值着色（跳过空值）
                    const isPvalCell =
                      j === pValIdx &&
                      typeof cell === "string" &&
                      cell !== "—" &&
                      cell !== "" &&
                      cell !== null;

                    let pvalClass = "";
                    if (isPvalCell) {
                      const numericP = Number(String(cell).replace("< ", ""));
                      pvalClass =
                        numericP <= 0.05
                          ? "text-destructive font-semibold"
                          : "text-green-600 font-semibold";
                    }

                    const isFirstCol = j === 0;
                    const isIndented =
                      isFirstCol &&
                      typeof cell === "string" &&
                      cell.startsWith("  ");

                    // β 变化列：对含 ⚠ 的单元格着色
                    const isWarningCell =
                      typeof cell === "string" && cell.includes("⚠");

                    return (
                      <td
                        key={j}
                        className={`px-3 py-2 whitespace-nowrap ${
                          isFirstCol
                            ? isIndented
                              ? "pl-7 text-muted-foreground"
                              : isCategoryHeader
                              ? "font-semibold"
                              : isHighlightRow
                              ? "font-bold text-primary"
                              : "font-medium"
                            : "text-muted-foreground"
                        } ${pvalClass} ${isWarningCell ? "text-amber-600 dark:text-amber-400 font-semibold" : ""}`}
                      >
                        {cell === null
                          ? "—"
                          : cell === ""
                          ? ""
                          : isIndented
                          ? String(cell).trimStart()
                          : String(cell)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 导出工具
// ─────────────────────────────────────────────────────────────────────────────

function exportCsv(table: TableResult) {
  const escapeCell = (v: unknown): string => {
    const s = v === null || v === undefined ? "" : String(v).trimStart();
    if (s.includes(",") || s.includes('"') || s.includes("\n")) {
      return `"${s.replace(/"/g, '""')}"`;
    }
    return s;
  };

  const lines = [
    table.headers.map(escapeCell).join(","),
    ...table.rows.map((row) => row.map(escapeCell).join(",")),
  ];

  // UTF-8 BOM for Excel Chinese compatibility
  const blob = new Blob(["\ufeff" + lines.join("\n")], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${table.title.replace(/[\s—]+/g, "_")}_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportJson(result: AnalysisResult) {
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `medstats_${result.method}_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─────────────────────────────────────────────────────────────────────────────
// 生存分析 / Cox 回归 — 表格区域
// 生存表可折叠；PH 违反行高亮；Log-rank p 值高亮
// ─────────────────────────────────────────────────────────────────────────────

function SurvivalCoxTables({ result }: SurvivalResultProps) {
  const COLLAPSIBLE_KEYWORDS = ["生存表", "Life Table"];
  const LOGRANK_KEYWORD = "Log-rank";
  const PH_KEYWORD = "Schoenfeld";

  return (
    <div className="space-y-6">
      {result.tables.map((t) => {
        const isCollapsible = COLLAPSIBLE_KEYWORDS.some((k) => t.title.includes(k));
        const isLogrank = t.title.includes(LOGRANK_KEYWORD);
        const isPH = t.title.includes(PH_KEYWORD) || t.title.includes("比例风险");

        if (isCollapsible) {
          return <CollapsibleTable key={t.title} table={t} defaultOpen={false} />;
        }
        if (isLogrank) {
          return (
            <ResultTable
              key={t.title}
              table={t}
              highlightKeyword="***"
              pHighlightCol="p 值"
            />
          );
        }
        if (isPH) {
          return (
            <ResultTable
              key={t.title}
              table={t}
              phWarningCol="结论"
            />
          );
        }
        return <ResultTable key={t.title} table={t} />;
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 可折叠表格
// ─────────────────────────────────────────────────────────────────────────────

function CollapsibleTable({ table, defaultOpen = false }: CollapsibleTableProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-2 text-lg font-semibold hover:text-primary transition-colors"
        >
          <span>{open ? "▾" : "▸"}</span>
          <span>{table.title}</span>
          {!open && (
            <span className="text-xs font-normal text-muted-foreground ml-1">
              ({table.rows.length} 行，点击展开)
            </span>
          )}
        </button>
        {open && (
          <button
            onClick={() => exportCsv(table)}
            className="px-3 py-1 border border-border rounded-lg text-xs text-muted-foreground hover:bg-muted/50 transition-colors shrink-0"
          >
            导出 CSV
          </button>
        )}
      </div>
      {open && (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted/40">
              <tr>
                {table.headers.map((h) => (
                  <th key={h} className="px-3 py-2.5 text-left font-semibold whitespace-nowrap border-b-2 border-border">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i} className="border-b border-border last:border-0 hover:bg-muted/20 transition-colors">
                  {row.map((cell, j) => (
                    <td key={j} className={`px-3 py-2 whitespace-nowrap text-sm ${j === 0 ? "font-medium" : "text-muted-foreground"}`}>
                      {cell === null ? "—" : String(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 生存分析图表布局
// KM 曲线全宽（含 number at risk）；累积风险图全宽
// ─────────────────────────────────────────────────────────────────────────────

function SurvivalCharts({ charts }: { charts: ChartResult[] }) {
  const kmChart = charts.find((c) => c.chart_type === "kaplan_meier");
  const otherCharts = charts.filter((c) => c.chart_type !== "kaplan_meier");

  return (
    <div className="space-y-6">
      {/* KM 曲线全宽 */}
      {kmChart && (
        <div className="rounded-xl border border-border p-4 space-y-3">
          <p className="text-sm font-semibold text-muted-foreground">{kmChart.title}</p>
          <EChartsRenderer option={kmChart.option} height={420} />
          {/* Number at risk 表 */}
          <NumberAtRiskTable narData={kmChart.option.numberAtRisk as NarEntry[] | undefined} />
        </div>
      )}
      {/* 其他图表（累积风险函数）全宽 */}
      {otherCharts.map((chart, i) => (
        <div key={i} className="rounded-xl border border-border p-4">
          <EChartsRenderer option={chart.option} height={340} />
        </div>
      ))}
    </div>
  );
}

interface NarEntry {
  group: string;
  times: number[];
  counts: number[];
}

function NumberAtRiskTable({ narData }: { narData?: NarEntry[] }) {
  if (!narData || narData.length === 0) return null;
  const times = narData[0].times;
  return (
    <div className="overflow-x-auto mt-2">
      <table className="text-xs w-full border-t border-border">
        <thead>
          <tr>
            <td className="pr-3 py-1 font-semibold text-muted-foreground whitespace-nowrap">Number at risk</td>
            {times.map((t) => (
              <td key={t} className="px-2 py-1 text-center text-muted-foreground">{Math.round(t)}</td>
            ))}
          </tr>
        </thead>
        <tbody>
          {narData.map((row) => (
            <tr key={row.group} className="border-t border-border/50">
              <td className="pr-3 py-1 font-medium">{row.group}</td>
              {row.counts.map((c, i) => (
                <td key={i} className="px-2 py-1 text-center">{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Cox 回归图表布局
// HR 森林图全宽；调整后生存曲线全宽；Schoenfeld + log-log 并排
// ─────────────────────────────────────────────────────────────────────────────

function CoxRegCharts({ charts }: { charts: ChartResult[] }) {
  const forestCharts = charts.filter((c) => c.chart_type === "forest_plot");
  const lineCharts   = charts.filter((c) => c.chart_type === "line");
  const scatterCharts = charts.filter((c) => c.chart_type === "scatter");

  return (
    <div className="space-y-6">
      {/* HR 森林图全宽 */}
      {forestCharts.map((chart, i) => (
        <div key={i} className="rounded-xl border border-border p-4">
          <ForestPlot
            option={chart.option}
            height={Math.max(320, ((chart.option.forestData as unknown[])?.length ?? 4) * 52 + 80)}
          />
        </div>
      ))}
      {/* 调整后生存曲线（line）全宽 */}
      {lineCharts.map((chart, i) => (
        <div key={i} className="rounded-xl border border-border p-4">
          <EChartsRenderer option={chart.option} height={380} />
        </div>
      ))}
      {/* Schoenfeld 残差 + log-log 并排 */}
      {scatterCharts.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {scatterCharts.map((chart, i) => (
            <div key={i} className="rounded-xl border border-border p-4">
              <EChartsRenderer option={chart.option} height={300} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PSM 表格区域 — 协变量平衡表 SMD 着色
// ─────────────────────────────────────────────────────────────────────────────

function PSMTables({ result }: SurvivalResultProps) {
  return (
    <div className="space-y-6">
      {result.tables.map((t) => {
        const isBalance = t.title.includes("平衡");
        if (!isBalance) return <ResultTable key={t.title} table={t} />;

        // 协变量平衡表：SMD 着色
        const smdAfterIdx = t.headers.indexOf("匹配后 SMD");
        const statusIdx = t.headers.indexOf("平衡状态");
        return (
          <div key={t.title} className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">{t.title}</h2>
              <button
                onClick={() => exportCsv(t)}
                className="px-3 py-1 border border-border rounded-lg text-xs text-muted-foreground hover:bg-muted/50 transition-colors shrink-0"
              >
                导出 CSV
              </button>
            </div>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/40">
                  <tr>
                    {t.headers.map((h) => (
                      <th key={h} className="px-3 py-2.5 text-left font-semibold whitespace-nowrap border-b-2 border-border">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {t.rows.map((row, i) => {
                    const status = statusIdx >= 0 ? String(row[statusIdx] ?? "") : "";
                    const isHeader = row[0] !== "" && !String(row[0]).startsWith("  ") && row.slice(1, -1).every((c) => c === "" || c === null);
                    const rowClass = isHeader
                      ? "bg-muted/20"
                      : status.startsWith("✗")
                      ? "bg-red-50 dark:bg-red-950/20"
                      : "hover:bg-muted/20";
                    return (
                      <tr key={i} className={`border-b border-border last:border-0 transition-colors ${rowClass}`}>
                        {row.map((cell, j) => {
                          let cellClass = j === 0 ? (String(cell).startsWith("  ") ? "pl-7 text-muted-foreground" : isHeader ? "font-semibold" : "font-medium") : "text-muted-foreground";
                          if (j === smdAfterIdx && typeof cell === "string" && cell !== "—") {
                            const v = parseFloat(cell);
                            if (!isNaN(v)) {
                              cellClass += v < 0.1 ? " text-green-600 font-semibold" : v < 0.2 ? " text-amber-600 font-semibold" : " text-red-600 font-semibold";
                            }
                          }
                          if (j === statusIdx) {
                            cellClass += status.startsWith("✓") ? " text-green-600 font-medium" : status.startsWith("✗") ? " text-red-600 font-medium" : "";
                          }
                          return (
                            <td key={j} className={`px-3 py-2 whitespace-nowrap ${cellClass}`}>
                              {cell === null ? "—" : String(cell).startsWith("  ") ? String(cell).trimStart() : String(cell)}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PSM 图表布局
// Love plot 全宽 → PS 分布并排 → SMD 条形图全宽 → KM 曲线全宽
// ─────────────────────────────────────────────────────────────────────────────

function PSMCharts({ charts }: { charts: ChartResult[] }) {
  const lovePlot  = charts.find((c) => c.title.includes("Love"));
  const psCharts  = charts.filter((c) => c.title.includes("PS 核密度"));
  const smdBar    = charts.find((c) => c.chart_type === "bar" && c.title.includes("SMD"));
  const kmChart   = charts.find((c) => c.chart_type === "kaplan_meier");

  return (
    <div className="space-y-6">
      {/* Love plot 全宽（最核心）*/}
      {lovePlot && (
        <div className="rounded-xl border border-border p-4">
          <p className="text-sm font-semibold text-muted-foreground mb-2">{lovePlot.title}</p>
          <EChartsRenderer option={lovePlot.option} height={Math.max(320, (lovePlot.option.yAxis as {data?: unknown[]})?.data?.length ?? 4) * 40 + 80} />
        </div>
      )}

      {/* PS 核密度图并排（匹配前 + 匹配后）*/}
      {psCharts.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {psCharts.map((c, i) => (
            <div key={i} className="rounded-xl border border-border p-4">
              <EChartsRenderer option={c.option} height={300} />
            </div>
          ))}
        </div>
      )}

      {/* SMD 条形图全宽 */}
      {smdBar && (
        <div className="rounded-xl border border-border p-4">
          <EChartsRenderer option={smdBar.option} height={Math.max(300, ((smdBar.option.yAxis as {data?: unknown[]})?.data?.length ?? 4) * 30 + 80)} />
        </div>
      )}

      {/* KM 曲线全宽（仅生存结局） */}
      {kmChart && (
        <div className="rounded-xl border border-border p-4 space-y-3">
          <p className="text-sm font-semibold text-muted-foreground">{kmChart.title}</p>
          <EChartsRenderer option={kmChart.option} height={380} />
          <NumberAtRiskTable narData={kmChart.option.numberAtRisk as NarEntry[] | undefined} />
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 临床预测模型图表布局
// ─────────────────────────────────────────────────────────────────────────────

function PredictionCharts({ charts }: { charts: ChartResult[] }) {
  const nomo      = charts.find((c) => c.chart_type === "nomogram");
  const roc       = charts.find((c) => c.title === "ROC 曲线");
  const calib     = charts.find((c) => c.title === "校准曲线");
  const dca       = charts.find((c) => c.title.includes("DCA"));
  const bootstrap = charts.find((c) => c.title.includes("Bootstrap"));
  const tdepAuc   = charts.find((c) => c.title.includes("时间依赖"));
  const coxCalib  = charts.find((c) => c.title.includes("Cox 校准"));

  return (
    <div className="space-y-6">
      {/* Nomogram 全宽（最核心图表） */}
      {nomo && (
        <div className="rounded-xl border border-border p-4">
          <p className="text-sm font-semibold mb-3">列线图（Nomogram）</p>
          <NomogramChart data={nomo.option.nomogramData as NomogramData} />
        </div>
      )}

      {/* ROC + 校准曲线并排（logistic） */}
      {(roc || calib) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {roc && (
            <div className="rounded-xl border border-border p-4">
              <EChartsRenderer option={roc.option} height={340} />
            </div>
          )}
          {calib && (
            <div className="rounded-xl border border-border p-4">
              <EChartsRenderer option={calib.option} height={340} />
            </div>
          )}
        </div>
      )}

      {/* 时间依赖 AUC（Cox） */}
      {tdepAuc && (
        <div className="rounded-xl border border-border p-4">
          <EChartsRenderer option={tdepAuc.option} height={300} />
        </div>
      )}

      {/* Cox 校准曲线（Cox） */}
      {coxCalib && (
        <div className="rounded-xl border border-border p-4">
          <EChartsRenderer option={coxCalib.option} height={300} />
        </div>
      )}

      {/* DCA 曲线全宽 */}
      {dca && (
        <div className="rounded-xl border border-border p-4">
          <EChartsRenderer option={dca.option} height={320} />
        </div>
      )}

      {/* Bootstrap AUC/C-index 分布直方图 */}
      {bootstrap && (
        <div className="rounded-xl border border-border p-4">
          <EChartsRenderer option={bootstrap.option} height={280} />
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Nomogram 渲染组件（基于 SVG，ECharts 不原生支持）
// ─────────────────────────────────────────────────────────────────────────────

interface NomogramTick { label: string; pts: number }
interface NomogramVariable { name: string; type: string; ticks: NomogramTick[] }
interface NomogramData {
  model_type: string;
  variables: NomogramVariable[];
  total_points: { min: number; max: number; ticks: number[] };
  prob_scale: { pts: number; prob: number }[];
  prob_label?: string;
}

function NomogramChart({ data }: { data?: NomogramData }) {
  if (!data || !data.variables?.length) {
    return <p className="text-sm text-muted-foreground text-center py-8">Nomogram 数据不可用</p>;
  }

  const { variables, total_points, prob_scale, prob_label } = data;
  const totalMax = total_points.max || 100;

  // SVG layout
  const rowH = 52;
  const labelW = 140;
  const axisW = 600;
  const paddingX = 20;
  const svgW = labelW + axisW + paddingX * 2;
  const headerRows = 1; // "Points" header row
  const totalPtsRow = 1;
  const probRow = 1;
  const totalRows = headerRows + variables.length + totalPtsRow + probRow;
  const svgH = totalRows * rowH + 40;

  const ptsToX = (pts: number) => labelW + paddingX + (pts / totalMax) * axisW;

  // Points header ticks (0..100 evenly)
  const headerTicks = Array.from({ length: 11 }, (_, i) => Math.round((i / 10) * 100));

  // Colors
  const rowColors = ["#f0f4ff", "#fff", "#f0fff4", "#fff8f0", "#f8f0ff"];

  return (
    <div className="overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        className="font-sans text-xs"
        style={{ fontFamily: "system-ui, sans-serif" }}
      >
        {/* ── Points header row ── */}
        <rect x={0} y={0} width={svgW} height={rowH} fill="#f8fafc" />
        <text x={labelW - 8} y={rowH / 2 + 5} textAnchor="end" fontSize={12} fontWeight={600} fill="#374151">
          分值
        </text>
        {headerTicks.map((tick) => {
          const scaledPts = (tick / 100) * totalMax;
          const x = ptsToX(scaledPts);
          return (
            <g key={tick}>
              <line x1={x} y1={rowH * 0.3} x2={x} y2={rowH * 0.7} stroke="#9ca3af" strokeWidth={1} />
              <text x={x} y={rowH * 0.9} textAnchor="middle" fontSize={10} fill="#6b7280">{tick}</text>
            </g>
          );
        })}
        <line x1={labelW + paddingX} y1={rowH / 2} x2={labelW + paddingX + axisW} y2={rowH / 2} stroke="#d1d5db" strokeWidth={1} />

        {/* ── Variable rows ── */}
        {variables.map((v, vi) => {
          const y = (vi + 1) * rowH;
          const bg = rowColors[vi % rowColors.length];
          const minPts = Math.min(...v.ticks.map((t) => t.pts));
          const maxPts = Math.max(...v.ticks.map((t) => t.pts));
          return (
            <g key={v.name}>
              <rect x={0} y={y} width={svgW} height={rowH} fill={bg} />
              <text x={labelW - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={500} fill="#1f2937">
                {v.name}
              </text>
              {/* Axis line */}
              <line
                x1={ptsToX(minPts)} y1={y + rowH / 2}
                x2={ptsToX(maxPts)} y2={y + rowH / 2}
                stroke="#6b7280" strokeWidth={1.5}
              />
              {/* Ticks */}
              {v.ticks.map((tick, ti) => {
                const x = ptsToX(tick.pts);
                const isTop = ti % 2 === 0;
                return (
                  <g key={ti}>
                    <line x1={x} y1={y + rowH * 0.3} x2={x} y2={y + rowH * 0.7} stroke="#374151" strokeWidth={1} />
                    <text
                      x={x} y={isTop ? y + rowH * 0.22 : y + rowH * 0.9}
                      textAnchor="middle" fontSize={9} fill="#374151"
                    >
                      {tick.label}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* ── Total Points row ── */}
        {(() => {
          const y = (variables.length + 1) * rowH;
          return (
            <g>
              <rect x={0} y={y} width={svgW} height={rowH} fill="#fef3c7" />
              <text x={labelW - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={600} fill="#92400e">
                总分
              </text>
              <line x1={ptsToX(0)} y1={y + rowH / 2} x2={ptsToX(totalMax)} y2={y + rowH / 2} stroke="#d97706" strokeWidth={1.5} />
              {total_points.ticks.map((tp, ti) => {
                const x = ptsToX(tp);
                return (
                  <g key={ti}>
                    <line x1={x} y1={y + rowH * 0.3} x2={x} y2={y + rowH * 0.7} stroke="#92400e" strokeWidth={1} />
                    <text x={x} y={y + rowH * 0.85} textAnchor="middle" fontSize={9} fill="#92400e">{Math.round(tp)}</text>
                  </g>
                );
              })}
            </g>
          );
        })()}

        {/* ── Probability row ── */}
        {(() => {
          const y = (variables.length + 2) * rowH;
          const pLabel = prob_label ?? (data.model_type === "cox" ? "生存率" : "预测概率");
          // Sample prob ticks every ~5th point
          const showProbs = prob_scale.filter((_, i) => i % 4 === 0 || i === prob_scale.length - 1);
          return (
            <g>
              <rect x={0} y={y} width={svgW} height={rowH} fill="#f0fdf4" />
              <text x={labelW - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={600} fill="#065f46">
                {pLabel}
              </text>
              <line x1={ptsToX(0)} y1={y + rowH / 2} x2={ptsToX(totalMax)} y2={y + rowH / 2} stroke="#059669" strokeWidth={1.5} />
              {showProbs.map((pt, pi) => {
                const x = ptsToX(pt.pts);
                const isTop = pi % 2 === 0;
                return (
                  <g key={pi}>
                    <line x1={x} y1={y + rowH * 0.3} x2={x} y2={y + rowH * 0.7} stroke="#065f46" strokeWidth={1} />
                    <text
                      x={x} y={isTop ? y + rowH * 0.22 : y + rowH * 0.9}
                      textAnchor="middle" fontSize={9} fill="#065f46"
                    >
                      {pt.prob.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 亚组分析 & 森林图 图表布局
// ─────────────────────────────────────────────────────────────────────────────

function ForestPlotCharts({ charts }: { charts: ChartResult[] }) {
  const forestCharts = charts.filter((c) => c.chart_type === "forest_plot");
  const otherCharts  = charts.filter((c) => c.chart_type !== "forest_plot");

  return (
    <div className="space-y-6">
      {forestCharts.map((chart, i) => {
        const rowCount = (chart.option.forestData as unknown[])?.length ?? 4;
        return (
          <div key={i} className="rounded-xl border border-border p-4">
            <p className="text-sm font-semibold text-muted-foreground mb-2">{chart.title}</p>
            <ForestPlot
              option={chart.option}
              height={Math.max(400, rowCount * 48 + 100)}
            />
          </div>
        );
      })}
      {/* 其他辅助图表（如有） */}
      {otherCharts.map((chart, i) => (
        <div key={i} className="rounded-xl border border-border p-4">
          <EChartsRenderer option={chart.option} height={320} />
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RCS 曲线 图表布局
// ─────────────────────────────────────────────────────────────────────────────

function RCSCharts({ charts }: { charts: ChartResult[] }) {
  const [histOpen, setHistOpen] = useState(false);
  const rcsCurve = charts.find((c) => c.title.includes("RCS") || c.title.includes("曲线"));
  const histChart = charts.find((c) => c.chart_type === "bar" && c.title.includes("分布"));

  return (
    <div className="space-y-6">
      {/* RCS 主曲线全宽 */}
      {rcsCurve && (
        <div className="rounded-xl border border-border p-4">
          <p className="text-sm font-semibold text-muted-foreground mb-2">{rcsCurve.title}</p>
          <EChartsRenderer option={rcsCurve.option} height={460} />
        </div>
      )}
      {/* 其余 line/scatter 图 */}
      {charts.filter((c) => c !== rcsCurve && c !== histChart).map((chart, i) => (
        <div key={i} className="rounded-xl border border-border p-4">
          <EChartsRenderer option={chart.option} height={320} />
        </div>
      ))}
      {/* 分布直方图可折叠 */}
      {histChart && (
        <div className="rounded-xl border border-border">
          <button
            onClick={() => setHistOpen((o) => !o)}
            className="w-full flex items-center gap-2 px-4 py-3 text-sm font-semibold hover:bg-muted/30 transition-colors rounded-xl"
          >
            <span>{histOpen ? "▾" : "▸"}</span>
            <span>暴露变量分布直方图（辅助参考）</span>
          </button>
          {histOpen && (
            <div className="px-4 pb-4">
              <EChartsRenderer option={histChart.option} height={260} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 阈值效应 图表布局
// ─────────────────────────────────────────────────────────────────────────────

function ThresholdCharts({ charts }: { charts: ChartResult[] }) {
  const [llOpen, setLlOpen] = useState(false);
  const effectChart = charts.find((c) => c.title.includes("阈值") || c.title.includes("效应"));
  const llChart     = charts.find((c) => c.title.includes("对数似然") || c.title.includes("似然曲线"));

  return (
    <div className="space-y-6">
      {/* 阈值效应主图全宽 */}
      {effectChart && (
        <div className="rounded-xl border border-border p-4">
          <p className="text-sm font-semibold text-muted-foreground mb-2">{effectChart.title}</p>
          <EChartsRenderer option={effectChart.option} height={440} />
        </div>
      )}
      {/* 其余图表（若有） */}
      {charts.filter((c) => c !== effectChart && c !== llChart).map((chart, i) => (
        <div key={i} className="rounded-xl border border-border p-4">
          <EChartsRenderer option={chart.option} height={320} />
        </div>
      ))}
      {/* 对数似然曲线可折叠 */}
      {llChart && (
        <div className="rounded-xl border border-border">
          <button
            onClick={() => setLlOpen((o) => !o)}
            className="w-full flex items-center gap-2 px-4 py-3 text-sm font-semibold hover:bg-muted/30 transition-colors rounded-xl"
          >
            <span>{llOpen ? "▾" : "▸"}</span>
            <span>对数似然曲线（候选拐点搜索过程）</span>
          </button>
          {llOpen && (
            <div className="px-4 pb-4">
              <EChartsRenderer option={llChart.option} height={300} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ResultTable 扩展：phWarningCol（PH 违反行标红）、pHighlightCol
// ─────────────────────────────────────────────────────────────────────────────
