"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { EChartsRenderer } from "@/components/charts/EChartsRenderer";
import { ForestPlot } from "@/components/charts/ForestPlot";
import type { AnalysisResult, ChartResult, TableResult } from "@/lib/types";

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
        {result.tables.map((t) => (
          <ResultTable
            key={t.title}
            table={t}
            highlightKeyword={
            result.method === "linear_reg_adjusted" || result.method === "logistic_reg_adjusted"
              ? "★"
              : undefined
          }
          />
        ))}
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
}: {
  table: TableResult;
  highlightKeyword?: string;
}) {
  const pValIdx = table.headers.indexOf("p 值");

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

              return (
                <tr
                  key={i}
                  className={`border-b border-border last:border-0 transition-colors ${
                    isHighlightRow
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
