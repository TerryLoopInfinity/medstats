"use client";

import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { uploadFile } from "@/lib/api";
import type { UploadResponse } from "@/lib/types";

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await uploadFile(file);
      setResult(res);
      // persist for downstream pages
      localStorage.setItem("ms_upload", JSON.stringify(res));
    } catch (e) {
      setError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setLoading(false);
    }
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold">上传数据</h1>
        <p className="text-muted-foreground text-sm mt-1">支持 CSV / XLSX，最大 10 MB</p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
          dragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx"
          className="hidden"
          onChange={onInputChange}
        />
        {loading ? (
          <div className="flex flex-col items-center gap-3">
            <Spinner />
            <p className="text-muted-foreground text-sm">正在上传并解析…</p>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-4xl">📂</p>
            <p className="font-medium">拖拽文件到此处，或点击选择文件</p>
            <p className="text-sm text-muted-foreground">CSV / XLSX · 最大 10 MB</p>
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="rounded-xl border border-border p-4 flex flex-wrap gap-6">
            <Stat label="文件名" value={result.filename} />
            <Stat label="行数" value={result.rows.toLocaleString()} />
            <Stat label="列数" value={result.columns.toString()} />
            {result.warnings.length > 0 && (
              <div className="w-full space-y-1">
                {result.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-amber-600">⚠ {w}</p>
                ))}
              </div>
            )}
          </div>

          {/* Data preview */}
          <div>
            <h2 className="font-semibold mb-2">数据预览（前 20 行）</h2>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    {result.column_names.map((h) => (
                      <th key={h} className="px-3 py-2 text-left font-medium whitespace-nowrap border-b border-border">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preview.map((row, i) => (
                    <tr key={i} className="border-b border-border last:border-0 hover:bg-muted/20">
                      {row.map((cell, j) => (
                        <td key={j} className="px-3 py-1.5 whitespace-nowrap text-muted-foreground">
                          {cell === null || cell === "" ? (
                            <span className="text-muted-foreground/40 italic">—</span>
                          ) : String(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="flex justify-end">
            <button
              onClick={() => router.push("/analyze")}
              className="px-5 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:opacity-90 transition-opacity"
            >
              下一步：选择分析方法 →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-semibold">{value}</p>
    </div>
  );
}

function Spinner() {
  return (
    <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
  );
}
