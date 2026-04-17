"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { analyze } from "@/lib/api";
import type { AnalysisMethod, UploadResponse } from "@/lib/types";

const METHODS: { value: AnalysisMethod; label: string; available: boolean }[] = [
  { value: "descriptive",  label: "统计描述 & 正态性检验", available: true },
  { value: "table_one",    label: "三线表生成",             available: true },
  { value: "ttest",        label: "差异性分析",             available: true },
  { value: "hypothesis",   label: "假设检验",               available: true },
  { value: "correlation",  label: "相关分析",               available: true  },
  { value: "linear_reg",          label: "线性回归",               available: true  },
  { value: "linear_reg_adjusted", label: "线性回归控制混杂",         available: true  },
  { value: "logistic_reg",        label: "Logistic 回归",          available: true  },
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

  // ── 统计描述 params ────────────────────────────────────────────
  const [selected, setSelected] = useState<string[]>([]);

  // ── 三线表 params ──────────────────────────────────────────────
  const [groupVar, setGroupVar] = useState<string>("");
  const [continuousVars, setContinuousVars] = useState<string[]>([]);
  const [categoricalVars, setCategoricalVars] = useState<string[]>([]);

  // ── 差异性分析 params ──────────────────────────────────────────
  const [ttestGroupVar, setTtestGroupVar] = useState<string>("");
  const [compareVars, setCompareVars] = useState<string[]>([]);
  const [compareType, setCompareType] = useState<"independent" | "paired">("independent");

  // ── 相关分析 params ────────────────────────────────────────────
  const [corrVars, setCorrVars] = useState<string[]>([]);
  const [corrMethod, setCorrMethod] = useState<"auto" | "pearson" | "spearman" | "kendall">("auto");

  // ── 线性回归 params ────────────────────────────────────────────
  const [lrOutcome, setLrOutcome] = useState<string>("");
  const [lrPredictors, setLrPredictors] = useState<string[]>([]);
  const [lrMode, setLrMode] = useState<"both" | "univariate" | "multivariate">("both");

  // ── 线性回归控制混杂 params ────────────────────────────────────
  const [lraOutcome, setLraOutcome] = useState<string>("");
  const [lraExposure, setLraExposure] = useState<string>("");
  const [lraCovariates, setLraCovariates] = useState<string[]>([]);
  const [lraModel2Covs, setLraModel2Covs] = useState<string[]>([]);
  const [lraStratifyVar, setLraStratifyVar] = useState<string>("");
  const [lraInteractionVar, setLraInteractionVar] = useState<string>("");
  const [lraMode, setLraMode] = useState<"both" | "crude" | "adjusted">("both");

  // ── Logistic 回归 params ──────────────────────────────────────
  const [lrLogOutcome, setLrLogOutcome] = useState<string>("");
  const [lrLogPredictors, setLrLogPredictors] = useState<string[]>([]);
  const [lrLogCatVars, setLrLogCatVars] = useState<string[]>([]);
  const [lrLogRefCats, setLrLogRefCats] = useState<Record<string, string>>({});
  const [lrLogMode, setLrLogMode] = useState<"both" | "univariate" | "multivariate">("both");

  // ── 假设检验 params ────────────────────────────────────────────
  const [testType, setTestType] = useState<"normality" | "variance" | "chi2" | "onesample">("normality");
  const [hypoVars, setHypoVars] = useState<string[]>([]);
  const [hypoGroupVar, setHypoGroupVar] = useState<string>("");
  const [rowVar, setRowVar] = useState<string>("");
  const [colVar, setColVar] = useState<string>("");
  const [mu, setMu] = useState<string>("0");
  const [alternative, setAlternative] = useState<"two-sided" | "less" | "greater">("two-sided");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem("ms_upload");
    if (!raw) return;
    const data: UploadResponse = JSON.parse(raw);
    setUpload(data);
    setSelected(data.column_names);
  }, []);

  // 切换方法时重置错误
  const handleMethodChange = (m: AnalysisMethod) => {
    setMethod(m);
    setError(null);
  };

  // ── 统计描述辅助 ───────────────────────────────────────────────
  const toggleAll = (checked: boolean) =>
    setSelected(checked ? (upload?.column_names ?? []) : []);
  const toggleVar = (col: string) =>
    setSelected((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );

  // ── 提交 ──────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!upload) return;
    setError(null);

    let params: Record<string, unknown> = {};

    if (method === "descriptive") {
      params = { variables: selected };
    } else if (method === "table_one") {
      if (!groupVar) {
        setError("请选择分组变量");
        return;
      }
      if (!continuousVars.length && !categoricalVars.length) {
        setError("请至少选择一个连续变量或分类变量");
        return;
      }
      params = {
        group_var: groupVar,
        continuous_vars: continuousVars,
        categorical_vars: categoricalVars,
      };
    } else if (method === "ttest") {
      if (!ttestGroupVar) {
        setError("请选择分组变量");
        return;
      }
      if (!compareVars.length) {
        setError("请至少选择一个待比较变量");
        return;
      }
      params = {
        group_var: ttestGroupVar,
        compare_vars: compareVars,
        compare_type: compareType,
      };
    } else if (method === "correlation") {
      if (corrVars.length < 2) {
        setError("请至少选择 2 个变量进行相关分析");
        return;
      }
      params = { variables: corrVars, method: corrMethod };
    } else if (method === "linear_reg") {
      if (!lrOutcome) {
        setError("请选择因变量");
        return;
      }
      if (!lrPredictors.length) {
        setError("请至少选择一个自变量");
        return;
      }
      params = { outcome: lrOutcome, predictors: lrPredictors, mode: lrMode };
    } else if (method === "linear_reg_adjusted") {
      if (!lraOutcome) { setError("请选择因变量"); return; }
      if (!lraExposure) { setError("请选择暴露变量"); return; }
      params = {
        outcome: lraOutcome,
        exposure: lraExposure,
        covariates: lraCovariates,
        mode: lraMode,
        ...(lraModel2Covs.length > 0 ? { model2_covariates: lraModel2Covs } : {}),
        ...(lraStratifyVar ? { stratify_var: lraStratifyVar } : {}),
        ...(lraInteractionVar ? { interaction_var: lraInteractionVar } : {}),
      };
    } else if (method === "logistic_reg") {
      if (!lrLogOutcome) { setError("请选择因变量"); return; }
      if (!lrLogPredictors.length) { setError("请至少选择一个自变量"); return; }
      params = {
        outcome: lrLogOutcome,
        predictors: lrLogPredictors,
        categorical_vars: lrLogCatVars,
        ref_categories: lrLogRefCats,
        mode: lrLogMode,
      };
    } else if (method === "hypothesis") {
      params = { test_type: testType };
      if (testType === "normality") {
        if (!hypoVars.length) { setError("请至少选择一个变量"); return; }
        params = { ...params, variables: hypoVars };
      } else if (testType === "variance") {
        if (!hypoVars.length) { setError("请至少选择一个变量"); return; }
        if (!hypoGroupVar) { setError("请选择分组变量"); return; }
        params = { ...params, variables: hypoVars, group_var: hypoGroupVar };
      } else if (testType === "chi2") {
        if (!rowVar || !colVar) { setError("请选择行变量和列变量"); return; }
        params = { ...params, row_var: rowVar, col_var: colVar };
      } else if (testType === "onesample") {
        if (!hypoVars.length) { setError("请至少选择一个变量"); return; }
        const muNum = parseFloat(mu);
        if (isNaN(muNum)) { setError("请输入有效的假设均值 μ"); return; }
        params = { ...params, variables: hypoVars, mu: muNum, alternative };
      }
    }

    setLoading(true);
    try {
      const result = await analyze(method, upload.file_id, params);
      localStorage.setItem("ms_result", JSON.stringify(result));
      router.push("/result");
    } catch (e) {
      setError(e instanceof Error ? e.message : "分析失败");
    } finally {
      setLoading(false);
    }
  };

  const canSubmit = (() => {
    if (!upload || loading) return false;
    if (method === "descriptive") return selected.length > 0;
    if (method === "table_one")
      return !!groupVar && (continuousVars.length > 0 || categoricalVars.length > 0);
    if (method === "ttest") return !!ttestGroupVar && compareVars.length > 0;
    if (method === "correlation") return corrVars.length >= 2;
    if (method === "linear_reg") return !!lrOutcome && lrPredictors.length > 0;
    if (method === "linear_reg_adjusted") return !!lraOutcome && !!lraExposure;
    if (method === "logistic_reg") return !!lrLogOutcome && lrLogPredictors.length > 0;
    if (method === "hypothesis") {
      if (testType === "normality" || testType === "onesample") return hypoVars.length > 0;
      if (testType === "variance") return hypoVars.length > 0 && !!hypoGroupVar;
      if (testType === "chi2") return !!rowVar && !!colVar;
    }
    return false;
  })();

  if (!upload) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-10">
        <div className="rounded-xl border border-border p-8 text-center space-y-3">
          <p className="text-muted-foreground">尚未上传数据文件</p>
          <a
            href="/upload"
            className="inline-block px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90"
          >
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

      {/* ── 分析方法选择 ── */}
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
                onChange={() => handleMethodChange(value)}
                className="accent-primary"
              />
              <span className="text-sm">{label}</span>
              {!available && (
                <span className="ml-auto text-xs text-muted-foreground">开发中</span>
              )}
            </label>
          ))}
        </div>
      </section>

      {/* ── 方法特定配置 ── */}
      {method === "table_one" ? (
        <TableOneConfig
          upload={upload}
          groupVar={groupVar}
          setGroupVar={(v) => {
            setGroupVar(v);
            setContinuousVars((prev) => prev.filter((c) => c !== v));
            setCategoricalVars((prev) => prev.filter((c) => c !== v));
          }}
          continuousVars={continuousVars}
          setContinuousVars={setContinuousVars}
          categoricalVars={categoricalVars}
          setCategoricalVars={setCategoricalVars}
        />
      ) : method === "correlation" ? (
        <CorrelationConfig
          upload={upload}
          corrVars={corrVars}
          setCorrVars={setCorrVars}
          corrMethod={corrMethod}
          setCorrMethod={setCorrMethod}
        />
      ) : method === "linear_reg" ? (
        <LinearRegConfig
          upload={upload}
          outcome={lrOutcome}
          setOutcome={(v) => { setLrOutcome(v); setLrPredictors((p) => p.filter((c) => c !== v)); }}
          predictors={lrPredictors}
          setPredictors={setLrPredictors}
          mode={lrMode}
          setMode={setLrMode}
        />
      ) : method === "linear_reg_adjusted" ? (
        <LinearRegAdjustedConfig
          upload={upload}
          outcome={lraOutcome}
          setOutcome={(v) => {
            setLraOutcome(v);
            setLraExposure((e) => e === v ? "" : e);
            setLraCovariates((c) => c.filter((x) => x !== v));
            setLraModel2Covs((c) => c.filter((x) => x !== v));
            setLraStratifyVar((s) => s === v ? "" : s);
            setLraInteractionVar((i) => i === v ? "" : i);
          }}
          exposure={lraExposure}
          setExposure={(v) => {
            setLraExposure(v);
            setLraCovariates((c) => c.filter((x) => x !== v));
            setLraModel2Covs((c) => c.filter((x) => x !== v));
          }}
          covariates={lraCovariates}
          setCovariates={(covs) => {
            setLraCovariates(covs);
            setLraModel2Covs((m2) => m2.filter((c) => covs.includes(c)));
          }}
          model2Covs={lraModel2Covs}
          setModel2Covs={setLraModel2Covs}
          stratifyVar={lraStratifyVar}
          setStratifyVar={setLraStratifyVar}
          interactionVar={lraInteractionVar}
          setInteractionVar={setLraInteractionVar}
          mode={lraMode}
          setMode={setLraMode}
        />
      ) : method === "logistic_reg" ? (
        <LogisticRegConfig
          upload={upload}
          outcome={lrLogOutcome}
          setOutcome={(v) => {
            setLrLogOutcome(v);
            setLrLogPredictors((p) => p.filter((c) => c !== v));
            setLrLogCatVars((c) => c.filter((x) => x !== v));
          }}
          predictors={lrLogPredictors}
          setPredictors={(preds) => {
            setLrLogPredictors(preds);
            setLrLogCatVars((c) => c.filter((x) => preds.includes(x)));
          }}
          catVars={lrLogCatVars}
          setCatVars={setLrLogCatVars}
          refCats={lrLogRefCats}
          setRefCats={setLrLogRefCats}
          mode={lrLogMode}
          setMode={setLrLogMode}
        />
      ) : method === "ttest" ? (
        <TTestConfig
          upload={upload}
          groupVar={ttestGroupVar}
          setGroupVar={(v) => { setTtestGroupVar(v); setCompareVars((p) => p.filter((c) => c !== v)); }}
          compareVars={compareVars}
          setCompareVars={setCompareVars}
          compareType={compareType}
          setCompareType={setCompareType}
        />
      ) : method === "hypothesis" ? (
        <HypothesisConfig
          upload={upload}
          testType={testType}
          setTestType={(t) => { setTestType(t); setHypoVars([]); setHypoGroupVar(""); setRowVar(""); setColVar(""); }}
          hypoVars={hypoVars}
          setHypoVars={setHypoVars}
          hypoGroupVar={hypoGroupVar}
          setHypoGroupVar={setHypoGroupVar}
          rowVar={rowVar}
          setRowVar={setRowVar}
          colVar={colVar}
          setColVar={setColVar}
          mu={mu}
          setMu={setMu}
          alternative={alternative}
          setAlternative={setAlternative}
        />
      ) : (
        /* 通用变量选择器（descriptive 及未来方法） */
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
                <span className="truncate" title={col}>
                  {col}
                </span>
              </label>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            已选 {selected.length} / {upload.column_names.length} 列
          </p>
        </section>
      )}

      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="flex gap-3 justify-end">
        <button
          onClick={() => router.push("/upload")}
          className="px-4 py-2 border border-border rounded-lg text-sm hover:bg-muted/50 transition-colors"
        >
          ← 重新上传
        </button>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="px-5 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading && <Spinner />}
          {loading ? "分析中…" : "开始分析"}
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 三线表配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface TableOneConfigProps {
  upload: UploadResponse;
  groupVar: string;
  setGroupVar: (v: string) => void;
  continuousVars: string[];
  setContinuousVars: React.Dispatch<React.SetStateAction<string[]>>;
  categoricalVars: string[];
  setCategoricalVars: React.Dispatch<React.SetStateAction<string[]>>;
}

function TableOneConfig({
  upload,
  groupVar,
  setGroupVar,
  continuousVars,
  setContinuousVars,
  categoricalVars,
  setCategoricalVars,
}: TableOneConfigProps) {
  const cols = upload.column_names;
  const analysisCols = cols.filter((c) => c !== groupVar);

  const toggleContinuous = (col: string) => {
    setCategoricalVars((prev) => prev.filter((c) => c !== col)); // 互斥
    setContinuousVars((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const toggleCategorical = (col: string) => {
    setContinuousVars((prev) => prev.filter((c) => c !== col)); // 互斥
    setCategoricalVars((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  return (
    <div className="space-y-6">
      {/* 分组变量 */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">
            分组变量{" "}
            <span className="text-destructive text-xs font-normal">必选</span>
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            选择用于划分组别的变量（如 group、treatment），建议为 2 分类变量
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label
              key={col}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                groupVar === col
                  ? "border-primary bg-primary/5 font-medium"
                  : "border-border hover:border-primary/40"
              }`}
            >
              <input
                type="radio"
                name="group_var"
                value={col}
                checked={groupVar === col}
                onChange={() => setGroupVar(col)}
                className="accent-primary"
              />
              <span className="truncate" title={col}>
                {col}
              </span>
            </label>
          ))}
        </div>
      </section>

      {/* 连续变量 */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">连续变量</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            自动正态性检验：正态 → 均值±SD + t 检验；非正态 → 中位数[IQR] + Mann-Whitney U
          </p>
        </div>
        {analysisCols.length === 0 ? (
          <p className="text-xs text-muted-foreground">请先选择分组变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {analysisCols.map((col) => {
              const inCat = categoricalVars.includes(col);
              return (
                <label
                  key={col}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors ${
                    inCat
                      ? "opacity-40 cursor-not-allowed border-border"
                      : continuousVars.includes(col)
                      ? "border-primary bg-primary/5 font-medium cursor-pointer"
                      : "border-border hover:border-primary/40 cursor-pointer"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={continuousVars.includes(col)}
                    disabled={inCat}
                    onChange={() => toggleContinuous(col)}
                    className="accent-primary"
                  />
                  <span className="truncate" title={col}>
                    {col}
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </section>

      {/* 分类变量 */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">分类变量</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            显示各组 n(%)，χ² 检验或 Fisher 精确检验（期望值 &lt; 5 时）
          </p>
        </div>
        {analysisCols.length === 0 ? (
          <p className="text-xs text-muted-foreground">请先选择分组变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {analysisCols.map((col) => {
              const inCont = continuousVars.includes(col);
              return (
                <label
                  key={col}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors ${
                    inCont
                      ? "opacity-40 cursor-not-allowed border-border"
                      : categoricalVars.includes(col)
                      ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20 font-medium cursor-pointer"
                      : "border-border hover:border-amber-400/60 cursor-pointer"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={categoricalVars.includes(col)}
                    disabled={inCont}
                    onChange={() => toggleCategorical(col)}
                    className="accent-amber-500"
                  />
                  <span className="truncate" title={col}>
                    {col}
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </section>

      {/* 选择摘要 */}
      {groupVar && (continuousVars.length > 0 || categoricalVars.length > 0) && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>
            分组变量：
            <span className="font-medium text-foreground">{groupVar}</span>
          </p>
          {continuousVars.length > 0 && (
            <p>
              连续变量（{continuousVars.length}）：
              <span className="font-medium text-foreground">
                {continuousVars.join("、")}
              </span>
            </p>
          )}
          {categoricalVars.length > 0 && (
            <p>
              分类变量（{categoricalVars.length}）：
              <span className="font-medium text-foreground">
                {categoricalVars.join("、")}
              </span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 差异性分析配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface TTestConfigProps {
  upload: UploadResponse;
  groupVar: string;
  setGroupVar: (v: string) => void;
  compareVars: string[];
  setCompareVars: React.Dispatch<React.SetStateAction<string[]>>;
  compareType: "independent" | "paired";
  setCompareType: (v: "independent" | "paired") => void;
}

function TTestConfig({
  upload, groupVar, setGroupVar, compareVars, setCompareVars, compareType, setCompareType,
}: TTestConfigProps) {
  const cols = upload.column_names;
  const analysisCols = cols.filter((c) => c !== groupVar);

  const toggleCompare = (col: string) =>
    setCompareVars((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  return (
    <div className="space-y-6">
      {/* 分组变量 */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">分组变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">选择用于划分组别的变量（2 组→独立/配对 t 或 U 检验；≥3 组→ANOVA/Kruskal-Wallis）</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${groupVar === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="ttest_group_var" value={col} checked={groupVar === col} onChange={() => setGroupVar(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* 待比较变量 */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">待比较变量 <span className="text-destructive text-xs font-normal">必选（可多选）</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">自动判断正态性和方差齐性，选择最适合的检验方法</p>
        </div>
        {analysisCols.length === 0 ? (
          <p className="text-xs text-muted-foreground">请先选择分组变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {analysisCols.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${compareVars.includes(col) ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                <input type="checkbox" checked={compareVars.includes(col)} onChange={() => toggleCompare(col)} className="accent-primary" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      {/* 比较类型 */}
      <section className="space-y-3">
        <h2 className="font-semibold">比较类型</h2>
        <div className="flex gap-4">
          {([["independent", "独立样本"], ["paired", "配对样本"]] as const).map(([val, label]) => (
            <label key={val} className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${compareType === val ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="compare_type" value={val} checked={compareType === val} onChange={() => setCompareType(val)} className="accent-primary" />
              {label}
            </label>
          ))}
        </div>
        {compareType === "paired" && (
          <p className="text-xs text-amber-600">配对比较仅支持两组，且两组样本量应相等或接近</p>
        )}
      </section>

      {groupVar && compareVars.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>分组变量：<span className="font-medium text-foreground">{groupVar}</span></p>
          <p>待比较变量（{compareVars.length}）：<span className="font-medium text-foreground">{compareVars.join("、")}</span></p>
          <p>比较类型：<span className="font-medium text-foreground">{compareType === "independent" ? "独立样本" : "配对样本"}</span></p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 假设检验配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface HypothesisConfigProps {
  upload: UploadResponse;
  testType: "normality" | "variance" | "chi2" | "onesample";
  setTestType: (v: "normality" | "variance" | "chi2" | "onesample") => void;
  hypoVars: string[];
  setHypoVars: React.Dispatch<React.SetStateAction<string[]>>;
  hypoGroupVar: string;
  setHypoGroupVar: (v: string) => void;
  rowVar: string;
  setRowVar: (v: string) => void;
  colVar: string;
  setColVar: (v: string) => void;
  mu: string;
  setMu: (v: string) => void;
  alternative: "two-sided" | "less" | "greater";
  setAlternative: (v: "two-sided" | "less" | "greater") => void;
}

function HypothesisConfig({
  upload, testType, setTestType, hypoVars, setHypoVars,
  hypoGroupVar, setHypoGroupVar, rowVar, setRowVar, colVar, setColVar,
  mu, setMu, alternative, setAlternative,
}: HypothesisConfigProps) {
  const cols = upload.column_names;
  const toggleVar = (col: string) =>
    setHypoVars((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  const TEST_TYPES = [
    { value: "normality" as const, label: "正态性检验", desc: "Shapiro-Wilk + Kolmogorov-Smirnov" },
    { value: "variance" as const,  label: "方差齐性检验", desc: "Levene + Bartlett" },
    { value: "chi2" as const,      label: "卡方 / Fisher 检验", desc: "列联表关联性检验" },
    { value: "onesample" as const, label: "单样本检验", desc: "单样本 t / Wilcoxon 符号秩" },
  ];

  return (
    <div className="space-y-6">
      {/* 检验类型 */}
      <section className="space-y-3">
        <h2 className="font-semibold">检验类型</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {TEST_TYPES.map(({ value, label, desc }) => (
            <label key={value} className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${testType === value ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}>
              <input type="radio" name="test_type" value={value} checked={testType === value} onChange={() => setTestType(value)} className="accent-primary mt-0.5" />
              <div>
                <p className="text-sm font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">{desc}</p>
              </div>
            </label>
          ))}
        </div>
      </section>

      {/* 正态性检验 / 单样本检验 / 方差齐性：变量选择 */}
      {(testType === "normality" || testType === "variance" || testType === "onesample") && (
        <section className="space-y-3">
          <h2 className="font-semibold">选择变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {cols.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${hypoVars.includes(col) ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                <input type="checkbox" checked={hypoVars.includes(col)} onChange={() => toggleVar(col)} className="accent-primary" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        </section>
      )}

      {/* 方差齐性：分组变量 */}
      {testType === "variance" && (
        <section className="space-y-3">
          <h2 className="font-semibold">分组变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {cols.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${hypoGroupVar === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                <input type="radio" name="hypo_group_var" value={col} checked={hypoGroupVar === col} onChange={() => setHypoGroupVar(col)} className="accent-primary" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        </section>
      )}

      {/* 卡方检验：行变量 + 列变量 */}
      {testType === "chi2" && (
        <>
          <section className="space-y-3">
            <h2 className="font-semibold">行变量（第一个分类变量）<span className="text-destructive text-xs font-normal">必选</span></h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {cols.map((col) => (
                <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${rowVar === col ? "border-primary bg-primary/5 font-medium" : colVar === col ? "opacity-40 cursor-not-allowed border-border" : "border-border hover:border-primary/40"}`}>
                  <input type="radio" name="row_var" value={col} checked={rowVar === col} disabled={colVar === col} onChange={() => setRowVar(col)} className="accent-primary" />
                  <span className="truncate" title={col}>{col}</span>
                </label>
              ))}
            </div>
          </section>
          <section className="space-y-3">
            <h2 className="font-semibold">列变量（第二个分类变量）<span className="text-destructive text-xs font-normal">必选</span></h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {cols.map((col) => (
                <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${colVar === col ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20 font-medium" : rowVar === col ? "opacity-40 cursor-not-allowed border-border" : "border-border hover:border-amber-400/60"}`}>
                  <input type="radio" name="col_var" value={col} checked={colVar === col} disabled={rowVar === col} onChange={() => setColVar(col)} className="accent-amber-500" />
                  <span className="truncate" title={col}>{col}</span>
                </label>
              ))}
            </div>
          </section>
        </>
      )}

      {/* 单样本检验：μ + 方向 */}
      {testType === "onesample" && (
        <section className="space-y-4">
          <div className="flex flex-wrap gap-6 items-end">
            <div>
              <label className="block text-sm font-semibold mb-1">假设均值 μ₀</label>
              <input
                type="number"
                value={mu}
                onChange={(e) => setMu(e.target.value)}
                className="w-32 px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                step="any"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-1">检验方向</label>
              <div className="flex gap-2">
                {([["two-sided", "双侧"], ["less", "左侧（< μ）"], ["greater", "右侧（> μ）"]] as const).map(([val, label]) => (
                  <label key={val} className={`flex items-center gap-1.5 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${alternative === val ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                    <input type="radio" name="alternative" value={val} checked={alternative === val} onChange={() => setAlternative(val)} className="accent-primary" />
                    {label}
                  </label>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 相关分析配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface CorrelationConfigProps {
  upload: UploadResponse;
  corrVars: string[];
  setCorrVars: React.Dispatch<React.SetStateAction<string[]>>;
  corrMethod: "auto" | "pearson" | "spearman" | "kendall";
  setCorrMethod: (v: "auto" | "pearson" | "spearman" | "kendall") => void;
}

function CorrelationConfig({ upload, corrVars, setCorrVars, corrMethod, setCorrMethod }: CorrelationConfigProps) {
  const cols = upload.column_names;
  const toggleVar = (col: string) =>
    setCorrVars((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  const METHODS = [
    { value: "auto"     as const, label: "自动选择", desc: "正态→Pearson，否则→Spearman" },
    { value: "pearson"  as const, label: "Pearson r",  desc: "参数法，要求正态分布" },
    { value: "spearman" as const, label: "Spearman ρ", desc: "秩相关，无需正态" },
    { value: "kendall"  as const, label: "Kendall τ",  desc: "秩相关，适合小样本" },
  ];

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">选择变量 <span className="text-destructive text-xs font-normal">必选（至少 2 个）</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">计算所选变量间的两两相关系数</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${corrVars.includes(col) ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="checkbox" checked={corrVars.includes(col)} onChange={() => toggleVar(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">已选 {corrVars.length} 个变量</p>
      </section>

      <section className="space-y-3">
        <h2 className="font-semibold">相关方法</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {METHODS.map(({ value, label, desc }) => (
            <label key={value} className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${corrMethod === value ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}>
              <input type="radio" name="corr_method" value={value} checked={corrMethod === value} onChange={() => setCorrMethod(value)} className="accent-primary mt-0.5" />
              <div>
                <p className="text-sm font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">{desc}</p>
              </div>
            </label>
          ))}
        </div>
      </section>

      {corrVars.length >= 2 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>已选变量（{corrVars.length}）：<span className="font-medium text-foreground">{corrVars.join("、")}</span></p>
          <p>相关方法：<span className="font-medium text-foreground">{METHODS.find(m => m.value === corrMethod)?.label}</span></p>
          <p>将生成 {corrVars.length * (corrVars.length - 1) / 2} 对相关系数{corrVars.length <= 5 ? " + 散点图矩阵" : ""}</p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 线性回归配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface LinearRegConfigProps {
  upload: UploadResponse;
  outcome: string;
  setOutcome: (v: string) => void;
  predictors: string[];
  setPredictors: React.Dispatch<React.SetStateAction<string[]>>;
  mode: "both" | "univariate" | "multivariate";
  setMode: (v: "both" | "univariate" | "multivariate") => void;
}

function LinearRegConfig({ upload, outcome, setOutcome, predictors, setPredictors, mode, setMode }: LinearRegConfigProps) {
  const cols = upload.column_names;
  const predCols = cols.filter((c) => c !== outcome);
  const togglePred = (col: string) =>
    setPredictors((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  const MODES = [
    { value: "both"         as const, label: "单变量 + 多变量", desc: "输出两种分析结果" },
    { value: "univariate"   as const, label: "仅单变量",         desc: "每个自变量单独回归" },
    { value: "multivariate" as const, label: "仅多变量",         desc: "所有自变量同时进入" },
  ];

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">因变量（结局变量）<span className="text-destructive text-xs font-normal">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">连续型数值变量</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${outcome === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="lr_outcome" value={col} checked={outcome === col} onChange={() => setOutcome(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">自变量（预测变量）<span className="text-destructive text-xs font-normal">必选（可多选）</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">数值型变量；分类变量请先进行哑变量编码</p>
        </div>
        {predCols.length === 0 ? (
          <p className="text-xs text-muted-foreground">请先选择因变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {predCols.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${predictors.includes(col) ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                <input type="checkbox" checked={predictors.includes(col)} onChange={() => togglePred(col)} className="accent-primary" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="font-semibold">分析模式</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {MODES.map(({ value, label, desc }) => (
            <label key={value} className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${mode === value ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}>
              <input type="radio" name="lr_mode" value={value} checked={mode === value} onChange={() => setMode(value)} className="accent-primary mt-0.5" />
              <div>
                <p className="text-sm font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">{desc}</p>
              </div>
            </label>
          ))}
        </div>
      </section>

      {outcome && predictors.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>因变量：<span className="font-medium text-foreground">{outcome}</span></p>
          <p>自变量（{predictors.length}）：<span className="font-medium text-foreground">{predictors.join("、")}</span></p>
          <p>模式：<span className="font-medium text-foreground">{MODES.find(m => m.value === mode)?.label}</span></p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 线性回归控制混杂配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface LinearRegAdjustedConfigProps {
  upload: UploadResponse;
  outcome: string;
  setOutcome: (v: string) => void;
  exposure: string;
  setExposure: (v: string) => void;
  covariates: string[];
  setCovariates: (v: string[]) => void;
  model2Covs: string[];
  setModel2Covs: React.Dispatch<React.SetStateAction<string[]>>;
  stratifyVar: string;
  setStratifyVar: (v: string) => void;
  interactionVar: string;
  setInteractionVar: (v: string) => void;
  mode: "both" | "crude" | "adjusted";
  setMode: (v: "both" | "crude" | "adjusted") => void;
}

function LinearRegAdjustedConfig({
  upload,
  outcome, setOutcome,
  exposure, setExposure,
  covariates, setCovariates,
  model2Covs, setModel2Covs,
  stratifyVar, setStratifyVar,
  interactionVar, setInteractionVar,
  mode, setMode,
}: LinearRegAdjustedConfigProps) {
  const cols = upload.column_names;

  const toggleCov = (col: string) =>
    setCovariates(
      covariates.includes(col) ? covariates.filter((c) => c !== col) : [...covariates, col]
    );

  const toggleModel2Cov = (col: string) =>
    setModel2Covs((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );

  const MODES = [
    { value: "both"     as const, label: "粗模型 + 调整模型", desc: "输出 Model 1/2/3 逐步对比" },
    { value: "crude"    as const, label: "仅粗模型",           desc: "不调整协变量" },
    { value: "adjusted" as const, label: "仅调整模型",         desc: "仅输出全调整结果" },
  ];

  // 可用于暴露/协变量/分层/交互的列（排除已选为因变量/暴露变量的）
  const nonOutcome = cols.filter((c) => c !== outcome);
  const nonExposureOrOutcome = cols.filter((c) => c !== outcome && c !== exposure);

  return (
    <div className="space-y-6">
      {/* ── 因变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">
            因变量（结局变量）<span className="text-destructive text-xs font-normal ml-1">必选</span>
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">连续型数值变量</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label
              key={col}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                outcome === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
              }`}
            >
              <input type="radio" name="lra_outcome" value={col} checked={outcome === col}
                onChange={() => setOutcome(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* ── 暴露变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">
            暴露变量（主要研究自变量）<span className="text-destructive text-xs font-normal ml-1">必选</span>
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            关注其对因变量的效应，系数变化将被全程追踪
          </p>
        </div>
        {!outcome ? (
          <p className="text-xs text-muted-foreground">请先选择因变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {nonOutcome.map((col) => (
              <label
                key={col}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                  exposure === col
                    ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20 font-medium"
                    : "border-border hover:border-emerald-400/60"
                }`}
              >
                <input type="radio" name="lra_exposure" value={col} checked={exposure === col}
                  onChange={() => setExposure(col)} className="accent-emerald-500" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      {/* ── 协变量（混杂因素）── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">协变量（需控制的混杂因素）</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            可多选；数值型变量，分类变量请先哑变量编码
          </p>
        </div>
        {!exposure ? (
          <p className="text-xs text-muted-foreground">请先选择暴露变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {nonExposureOrOutcome.map((col) => (
              <label
                key={col}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                  covariates.includes(col)
                    ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20 font-medium"
                    : "border-border hover:border-amber-400/60"
                }`}
              >
                <input type="checkbox" checked={covariates.includes(col)}
                  onChange={() => toggleCov(col)} className="accent-amber-500" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      {/* ── Model 2 协变量（可选） ── */}
      {covariates.length >= 2 && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">Model 2 协变量（可选）</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              从上方协变量中选择进入 Model 2（部分调整）；留空则自动取前一半
            </p>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {covariates.map((col) => (
              <label
                key={col}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                  model2Covs.includes(col)
                    ? "border-violet-500 bg-violet-50 dark:bg-violet-950/20 font-medium"
                    : "border-border hover:border-violet-400/60"
                }`}
              >
                <input type="checkbox" checked={model2Covs.includes(col)}
                  onChange={() => toggleModel2Cov(col)} className="accent-violet-500" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
          <div className="flex gap-2 flex-wrap">
            <span className="text-xs px-2 py-1 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300">
              Model 2：{model2Covs.length > 0 ? model2Covs.join("、") : "自动（前一半）"}
            </span>
            <span className="text-xs px-2 py-1 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300">
              Model 3（全调整）：{covariates.join("、")}
            </span>
          </div>
        </section>
      )}

      {/* ── 分层变量（可选）── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">分层变量（可选）</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            按各水平分别拟合模型，输出亚组森林图及 Cochran Q 异质性检验
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {/* 无 / 清除 */}
          <label className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
            !stratifyVar ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
          }`}>
            <input type="radio" name="lra_stratify" value="" checked={!stratifyVar}
              onChange={() => setStratifyVar("")} className="accent-primary" />
            <span className="text-muted-foreground">不分层</span>
          </label>
          {cols.filter((c) => c !== outcome && c !== exposure).map((col) => (
            <label
              key={col}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                stratifyVar === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
              }`}
            >
              <input type="radio" name="lra_stratify" value={col} checked={stratifyVar === col}
                onChange={() => setStratifyVar(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* ── 交互项变量（可选）── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">交互项变量（可选）</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            检验 暴露 × 该变量 的交互效应（效应修饰），输出交互 p 值
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          <label className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
            !interactionVar ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
          }`}>
            <input type="radio" name="lra_interaction" value="" checked={!interactionVar}
              onChange={() => setInteractionVar("")} className="accent-primary" />
            <span className="text-muted-foreground">不检验</span>
          </label>
          {cols.filter((c) => c !== outcome).map((col) => (
            <label
              key={col}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                interactionVar === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
              }`}
            >
              <input type="radio" name="lra_interaction" value={col} checked={interactionVar === col}
                onChange={() => setInteractionVar(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* ── 分析模式 ── */}
      <section className="space-y-3">
        <h2 className="font-semibold">分析模式</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {MODES.map(({ value, label, desc }) => (
            <label
              key={value}
              className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
                mode === value ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <input type="radio" name="lra_mode" value={value} checked={mode === value}
                onChange={() => setMode(value)} className="accent-primary mt-0.5" />
              <div>
                <p className="text-sm font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">{desc}</p>
              </div>
            </label>
          ))}
        </div>
      </section>

      {/* ── 配置摘要 ── */}
      {outcome && exposure && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>因变量：<span className="font-medium text-foreground">{outcome}</span></p>
          <p>暴露变量：<span className="font-medium text-emerald-600 dark:text-emerald-400">{exposure}</span></p>
          {covariates.length > 0 && (
            <p>协变量（{covariates.length}）：<span className="font-medium text-foreground">{covariates.join("、")}</span></p>
          )}
          {stratifyVar && (
            <p>分层变量：<span className="font-medium text-foreground">{stratifyVar}</span></p>
          )}
          {interactionVar && (
            <p>交互项：<span className="font-medium text-foreground">{exposure} × {interactionVar}</span></p>
          )}
          <p>模式：<span className="font-medium text-foreground">{MODES.find((m) => m.value === mode)?.label}</span></p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Logistic 回归配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface LogisticRegConfigProps {
  upload: UploadResponse;
  outcome: string;
  setOutcome: (v: string) => void;
  predictors: string[];
  setPredictors: (v: string[]) => void;
  catVars: string[];
  setCatVars: React.Dispatch<React.SetStateAction<string[]>>;
  refCats: Record<string, string>;
  setRefCats: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  mode: "both" | "univariate" | "multivariate";
  setMode: (v: "both" | "univariate" | "multivariate") => void;
}

function LogisticRegConfig({
  upload, outcome, setOutcome, predictors, setPredictors,
  catVars, setCatVars, refCats, setRefCats, mode, setMode,
}: LogisticRegConfigProps) {
  const cols = upload.column_names;
  const predCols = cols.filter((c) => c !== outcome);

  const togglePred = (col: string) =>
    setPredictors(
      predictors.includes(col) ? predictors.filter((c) => c !== col) : [...predictors, col]
    );

  const toggleCat = (col: string) => {
    setCatVars((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
    // Reset ref category when unchecking
    if (catVars.includes(col)) {
      setRefCats((prev) => { const next = { ...prev }; delete next[col]; return next; });
    }
  };

  // 从 preview 中提取某列的唯一值
  const getPreviewValues = (colName: string): string[] => {
    const colIdx = upload.column_names.indexOf(colName);
    if (colIdx === -1) return [];
    const vals = new Set<string>();
    for (const row of upload.preview) {
      const v = row[colIdx];
      if (v !== null && v !== undefined && v !== "") vals.add(String(v));
    }
    return Array.from(vals).sort();
  };

  const MODES = [
    { value: "both"         as const, label: "单变量 + 多变量", desc: "输出两种分析结果" },
    { value: "univariate"   as const, label: "仅单变量",         desc: "每个自变量单独回归" },
    { value: "multivariate" as const, label: "仅多变量",         desc: "所有自变量同时进入" },
  ];

  return (
    <div className="space-y-6">
      {/* ── 因变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">
            因变量（二分类结局）<span className="text-destructive text-xs font-normal ml-1">必选</span>
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            支持 0/1、True/False 或两个字符串类别
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
              outcome === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
            }`}>
              <input type="radio" name="log_outcome" value={col} checked={outcome === col}
                onChange={() => setOutcome(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* ── 自变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">
            自变量（预测变量）<span className="text-destructive text-xs font-normal ml-1">必选（可多选）</span>
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            可混合连续变量与分类变量；分类变量需在下方标记
          </p>
        </div>
        {predCols.length === 0 ? (
          <p className="text-xs text-muted-foreground">请先选择因变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {predCols.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${
                predictors.includes(col) ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
              }`}>
                <input type="checkbox" checked={predictors.includes(col)}
                  onChange={() => togglePred(col)} className="accent-primary" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      {/* ── 分类变量标记 + 参考组 ── */}
      {predictors.length > 0 && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">分类变量标记</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              勾选后将自动 dummy 编码；可从预览数据中选择参考组（默认取第一个类别）
            </p>
          </div>
          <div className="space-y-2">
            {predictors.map((col) => {
              const isCat = catVars.includes(col);
              const previewVals = isCat ? getPreviewValues(col) : [];
              return (
                <div key={col} className={`flex flex-wrap items-center gap-3 px-4 py-3 rounded-lg border transition-colors ${
                  isCat ? "border-amber-400 bg-amber-50 dark:bg-amber-950/20" : "border-border"
                }`}>
                  <label className="flex items-center gap-2 text-sm cursor-pointer min-w-[140px]">
                    <input type="checkbox" checked={isCat} onChange={() => toggleCat(col)}
                      className="accent-amber-500" />
                    <span className="font-medium">{col}</span>
                    {isCat && (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200">
                        分类
                      </span>
                    )}
                  </label>
                  {isCat && previewVals.length > 0 && (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-xs text-muted-foreground">参考组：</span>
                      <select
                        value={refCats[col] ?? previewVals[0]}
                        onChange={(e) =>
                          setRefCats((prev) => ({ ...prev, [col]: e.target.value }))
                        }
                        className="text-sm px-2 py-1 border border-border rounded-lg focus:outline-none focus:ring-1 focus:ring-primary/30 bg-background"
                      >
                        {previewVals.map((v) => (
                          <option key={v} value={v}>{v}</option>
                        ))}
                      </select>
                    </div>
                  )}
                  {isCat && previewVals.length === 0 && (
                    <p className="text-xs text-muted-foreground">预览数据不足，将使用默认参考组（字母序第一）</p>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── 分析模式 ── */}
      <section className="space-y-3">
        <h2 className="font-semibold">分析模式</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {MODES.map(({ value, label, desc }) => (
            <label key={value} className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${
              mode === value ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
            }`}>
              <input type="radio" name="log_mode" value={value} checked={mode === value}
                onChange={() => setMode(value)} className="accent-primary mt-0.5" />
              <div>
                <p className="text-sm font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">{desc}</p>
              </div>
            </label>
          ))}
        </div>
      </section>

      {/* ── 配置摘要 ── */}
      {outcome && predictors.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>因变量：<span className="font-medium text-foreground">{outcome}</span></p>
          <p>自变量（{predictors.length}）：<span className="font-medium text-foreground">{predictors.join("、")}</span></p>
          {catVars.length > 0 && (
            <p>分类变量：
              <span className="font-medium text-foreground">
                {catVars.map((v) => `${v}（参考：${refCats[v] ?? "默认"}）`).join("、")}
              </span>
            </p>
          )}
          <p>模式：<span className="font-medium text-foreground">{MODES.find((m) => m.value === mode)?.label}</span></p>
        </div>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
  );
}
