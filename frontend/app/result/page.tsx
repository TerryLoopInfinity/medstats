"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { EChartsRenderer } from "@/components/charts/EChartsRenderer";
import type { AnalysisResult, ChartResult, TableResult } from "@/lib/types";

const METHOD_LABELS: Record<string, string> = {
  descriptive: "统计描述 & 正态性检验",
};

export default function ResultPage() {
  const router = useRouter();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  // pair up histogram + qq chart for each variable
  const [chartPairs, setChartPairs] = useState<[ChartResult, ChartResult][]>([]);

  useEffect(() => {
    const raw = localStorage.getItem("ms_result");
    if (!raw) return;
    const data: AnalysisResult = JSON.parse(raw);
    setResult(data);

    // Build [histogram, qqplot] pairs (charts are interleaved: hist, qq, hist, qq …)
    const pairs: [ChartResult, ChartResult][] = [];
    for (let i = 0; i < data.charts.length - 1; i += 2) {
      pairs.push([data.charts[i], data.charts[i + 1]]);
    }
    setChartPairs(pairs);
  }, []);

  if (!result) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-10">
        <div className="rounded-xl border border-border p-8 text-center space-y-3">
          <p className="text-muted-foreground">尚无分析结果</p>
          <a href="/upload" className="inline-block px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90">
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
          <h1 className="text-2xl font-bold">{METHOD_LABELS[result.method] ?? result.method}</h1>
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
            <p key={i} className="text-sm text-amber-700">⚠ {w}</p>
          ))}
        </div>
      )}

      {/* Tables */}
      <section className="space-y-6">
        {result.tables.map((t) => (
          <ResultTable key={t.title} table={t} />
        ))}
      </section>

      {/* Charts — histogram + QQ side by side per variable */}
      {chartPairs.length > 0 && (
        <section className="space-y-8">
          <h2 className="text-lg font-semibold border-b border-border pb-2">图表</h2>
          {chartPairs.map(([hist, qq], i) => (
            <div key={i} className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-xl border border-border p-4">
                <EChartsRenderer option={hist.option} height={300} />
              </div>
              <div className="rounded-xl border border-border p-4">
                <EChartsRenderer option={qq.option} height={300} />
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

function ResultTable({ table }: { table: TableResult }) {
  return (
    <div className="space-y-2">
      <h2 className="text-lg font-semibold">{table.title}</h2>
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
            {table.rows.map((row, i) => (
              <tr key={i} className="border-b border-border last:border-0 hover:bg-muted/20 transition-colors">
                {row.map((cell, j) => (
                  <td
                    key={j}
                    className={`px-3 py-2 whitespace-nowrap ${
                      j === 0 ? "font-medium" : "text-muted-foreground"
                    } ${
                      // highlight non-normal p-values
                      j === table.headers.indexOf("p 值") && typeof cell === "string" && cell !== "—"
                        ? Number(cell.replace("< ", "")) <= 0.05
                          ? "text-destructive font-medium"
                          : "text-green-600 font-medium"
                        : ""
                    }`}
                  >
                    {cell === null || cell === "" ? "—" : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function exportJson(result: AnalysisResult) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `medstats_${result.method}_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
