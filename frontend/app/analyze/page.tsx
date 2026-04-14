"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { analyze } from "@/lib/api";
import type { AnalysisMethod, UploadResponse } from "@/lib/types";

const METHODS: { value: AnalysisMethod; label: string; available: boolean }[] = [
  { value: "descriptive",  label: "统计描述 & 正态性检验", available: true },
  { value: "table_one",    label: "三线表生成",             available: false },
  { value: "ttest",        label: "差异性分析",             available: false },
  { value: "correlation",  label: "相关 & 线性回归",        available: false },
  { value: "logistic_reg", label: "Logistic 回归",          available: false },
  { value: "survival",     label: "生存分析 & Cox 回归",    available: false },
  { value: "psm",          label: "倾向性评分匹配",         available: false },
  { value: "prediction",   label: "临床预测模型",           available: false },
  { value: "forest_plot",  label: "亚组分析 & 森林图",      available: false },
  { value: "rcs",          label: "RCS 曲线",               available: false },
  { value: "threshold",    label: "阈值效应分析",           available: false },
  { value: "mediation",    label: "中介分析",               available: false },
  { value: "sample_size",  label: "样本量计算",             available: false },
];

export default function AnalyzePage() {
  const router = useRouter();
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [method, setMethod] = useState<AnalysisMethod>("descriptive");
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem("ms_upload");
    if (!raw) return;
    const data: UploadResponse = JSON.parse(raw);
    setUpload(data);
    // default: select all numeric-looking columns (select all for now)
    setSelected(data.column_names);
  }, []);

  const toggleAll = (checked: boolean) => {
    setSelected(checked ? (upload?.column_names ?? []) : []);
  };

  const toggleVar = (col: string) => {
    setSelected((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const handleSubmit = async () => {
    if (!upload) return;
    setError(null);
    setLoading(true);
    try {
      const result = await analyze(method, upload.file_id, { variables: selected });
      localStorage.setItem("ms_result", JSON.stringify(result));
      router.push("/result");
    } catch (e) {
      setError(e instanceof Error ? e.message : "分析失败");
    } finally {
      setLoading(false);
    }
  };

  if (!upload) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-10">
        <div className="rounded-xl border border-border p-8 text-center space-y-3">
          <p className="text-muted-foreground">尚未上传数据文件</p>
          <a href="/upload" className="inline-block px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90">
            去上传数据
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold">配置分析</h1>
        <p className="text-sm text-muted-foreground mt-1">
          文件：{upload.filename} · {upload.rows} 行 · {upload.columns} 列
        </p>
      </div>

      {/* Method selector */}
      <section className="space-y-3">
        <h2 className="font-semibold">选择分析方法</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {METHODS.map(({ value, label, available }) => (
            <label
              key={value}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
                !available
                  ? "opacity-40 cursor-not-allowed border-border"
                  : method === value
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              }`}
            >
              <input
                type="radio"
                name="method"
                value={value}
                checked={method === value}
                disabled={!available}
                onChange={() => setMethod(value)}
                className="accent-primary"
              />
              <span className="text-sm">{label}</span>
              {!available && <span className="ml-auto text-xs text-muted-foreground">开发中</span>}
            </label>
          ))}
        </div>
      </section>

      {/* Variable selector */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold">选择变量</h2>
          <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={selected.length === upload.column_names.length}
              onChange={(e) => toggleAll(e.target.checked)}
              className="accent-primary"
            />
            全选
          </label>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {upload.column_names.map((col) => (
            <label
              key={col}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                selected.includes(col)
                  ? "border-primary bg-primary/5 font-medium"
                  : "border-border hover:border-primary/40"
              }`}
            >
              <input
                type="checkbox"
                checked={selected.includes(col)}
                onChange={() => toggleVar(col)}
                className="accent-primary"
              />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">已选 {selected.length} / {upload.column_names.length} 列</p>
      </section>

      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="flex gap-3 justify-end">
        <button onClick={() => router.push("/upload")} className="px-4 py-2 border border-border rounded-lg text-sm hover:bg-muted/50 transition-colors">
          ← 重新上传
        </button>
        <button
          onClick={handleSubmit}
          disabled={loading || selected.length === 0}
          className="px-5 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading && <Spinner />}
          {loading ? "分析中…" : "开始分析"}
        </button>
      </div>
    </div>
  );
}

function Spinner() {
  return <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />;
}
