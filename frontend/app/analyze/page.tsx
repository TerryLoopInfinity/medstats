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
  { value: "logistic_reg_adjusted", label: "Logistic 回归控制混杂",   available: true  },
  { value: "survival",     label: "生存分析（Kaplan-Meier）",  available: true  },
  { value: "cox_reg",      label: "Cox 回归",                 available: true  },
  { value: "psm",          label: "倾向性得分匹配（PSM）",  available: true  },
  { value: "prediction",   label: "临床预测模型",           available: true  },
  { value: "forest_plot",  label: "亚组分析 & 森林图",      available: true  },
  { value: "rcs",          label: "RCS 曲线",               available: true  },
  { value: "threshold",    label: "阈值效应分析",           available: true  },
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

  // ── Logistic 回归控制混杂 params ──────────────────────────────
  const [lrlaOutcome, setLrlaOutcome] = useState<string>("");
  const [lrlaExposure, setLrlaExposure] = useState<string>("");
  const [lrlaCovariates, setLrlaCovariates] = useState<string[]>([]);
  const [lrlaCatVars, setLrlaCatVars] = useState<string[]>([]);
  const [lrlaRefCats, setLrlaRefCats] = useState<Record<string, string>>({});
  const [lrlaModel2Covs, setLrlaModel2Covs] = useState<string[]>([]);
  const [lrlaStratifyVar, setLrlaStratifyVar] = useState<string>("");
  const [lrlaInteractionVar, setLrlaInteractionVar] = useState<string>("");
  const [lrlaMode, setLrlaMode] = useState<"both" | "crude" | "adjusted">("both");

  // ── 生存分析 params ────────────────────────────────────────────
  const [survTimeCol, setSurvTimeCol] = useState<string>("");
  const [survEventCol, setSurvEventCol] = useState<string>("");
  const [survGroupCol, setSurvGroupCol] = useState<string>("");
  const [survTimePoints, setSurvTimePoints] = useState<string>("");

  // ── Cox 回归 params ────────────────────────────────────────────
  const [coxTimeCol, setCoxTimeCol] = useState<string>("");
  const [coxEventCol, setCoxEventCol] = useState<string>("");
  const [coxPredictors, setCoxPredictors] = useState<string[]>([]);
  const [coxCatVars, setCoxCatVars] = useState<string[]>([]);
  const [coxRefCats, setCoxRefCats] = useState<Record<string, string>>({});
  const [coxMode, setCoxMode] = useState<"both" | "univariate" | "multivariate">("both");

  // ── PSM params ─────────────────────────────────────────────────
  const [psmTreatCol, setPsmTreatCol] = useState<string>("");
  const [psmCovariates, setPsmCovariates] = useState<string[]>([]);
  const [psmOutcomeCol, setPsmOutcomeCol] = useState<string>("");
  const [psmOutcomeType, setPsmOutcomeType] = useState<"continuous" | "binary" | "survival">("survival");
  const [psmTimeCol, setPsmTimeCol] = useState<string>("");
  const [psmEventCol, setPsmEventCol] = useState<string>("");
  const [psmMethod, setPsmMethod] = useState<"nearest" | "caliper" | "optimal">("nearest");
  const [psmCaliper, setPsmCaliper] = useState<string>("");
  const [psmRatio, setPsmRatio] = useState<1 | 2 | 3>(1);
  const [psmWithReplacement, setPsmWithReplacement] = useState<boolean>(false);

  // ── 临床预测模型 params ────────────────────────────────────────
  const [predModelType, setPredModelType] = useState<"logistic" | "cox">("logistic");
  const [predOutcome, setPredOutcome] = useState<string>("");
  const [predTimeCol, setPredTimeCol] = useState<string>("");
  const [predEventCol, setPredEventCol] = useState<string>("");
  const [predPredictors, setPredPredictors] = useState<string[]>([]);
  const [predCatVars, setPredCatVars] = useState<string[]>([]);
  const [predRefCats, setPredRefCats] = useState<Record<string, string>>({});
  const [predValidation, setPredValidation] = useState<"internal_bootstrap" | "split" | "cross_validation">("internal_bootstrap");
  const [predNBoot, setPredNBoot] = useState<string>("1000");
  const [predTrainRatio, setPredTrainRatio] = useState<string>("0.7");
  const [predTimePoint, setPredTimePoint] = useState<string>("");
  const [predStepwise, setPredStepwise] = useState<boolean>(false);

  // ── 亚组分析 & 森林图 params ───────────────────────────────────
  const [fpModelType, setFpModelType] = useState<"logistic" | "cox" | "linear">("logistic");
  const [fpOutcome, setFpOutcome] = useState<string>("");
  const [fpTimeCol, setFpTimeCol] = useState<string>("");
  const [fpEventCol, setFpEventCol] = useState<string>("");
  const [fpExposure, setFpExposure] = useState<string>("");
  const [fpCovariates, setFpCovariates] = useState<string[]>([]);
  const [fpSubgroupVars, setFpSubgroupVars] = useState<string[]>([]);
  const [fpCatVars, setFpCatVars] = useState<string[]>([]);

  // ── RCS 曲线 params ────────────────────────────────────────────
  const [rcsModelType, setRcsModelType] = useState<"logistic" | "cox" | "linear">("linear");
  const [rcsOutcome, setRcsOutcome] = useState<string>("");
  const [rcsTimeCol, setRcsTimeCol] = useState<string>("");
  const [rcsEventCol, setRcsEventCol] = useState<string>("");
  const [rcsExposure, setRcsExposure] = useState<string>("");
  const [rcsCovariates, setRcsCovariates] = useState<string[]>([]);
  const [rcsNKnots, setRcsNKnots] = useState<3 | 4 | 5>(4);
  const [rcsRefValue, setRcsRefValue] = useState<string>("");

  // ── 阈值效应 params ────────────────────────────────────────────
  const [thrModelType, setThrModelType] = useState<"logistic" | "cox" | "linear">("linear");
  const [thrOutcome, setThrOutcome] = useState<string>("");
  const [thrTimeCol, setThrTimeCol] = useState<string>("");
  const [thrEventCol, setThrEventCol] = useState<string>("");
  const [thrExposure, setThrExposure] = useState<string>("");
  const [thrCovariates, setThrCovariates] = useState<string[]>([]);

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
    } else if (method === "logistic_reg_adjusted") {
      if (!lrlaOutcome) { setError("请选择因变量"); return; }
      if (!lrlaExposure) { setError("请选择暴露变量"); return; }
      params = {
        outcome: lrlaOutcome,
        exposure: lrlaExposure,
        covariates: lrlaCovariates,
        categorical_vars: lrlaCatVars,
        ref_categories: lrlaRefCats,
        mode: lrlaMode,
        ...(lrlaModel2Covs.length > 0 ? { model2_covariates: lrlaModel2Covs } : {}),
        ...(lrlaStratifyVar ? { stratify_var: lrlaStratifyVar } : {}),
        ...(lrlaInteractionVar ? { interaction_var: lrlaInteractionVar } : {}),
      };
    } else if (method === "survival") {
      if (!survTimeCol) { setError("请选择时间变量"); return; }
      if (!survEventCol) { setError("请选择事件变量"); return; }
      const rawPts = survTimePoints.trim();
      const parsedPts = rawPts
        ? rawPts.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n))
        : [];
      params = {
        time_col: survTimeCol,
        event_col: survEventCol,
        ...(survGroupCol ? { group_col: survGroupCol } : {}),
        ...(parsedPts.length > 0 ? { time_points: parsedPts } : {}),
      };
    } else if (method === "cox_reg") {
      if (!coxTimeCol) { setError("请选择时间变量"); return; }
      if (!coxEventCol) { setError("请选择事件变量"); return; }
      if (!coxPredictors.length) { setError("请至少选择一个自变量"); return; }
      params = {
        time_col: coxTimeCol,
        event_col: coxEventCol,
        predictors: coxPredictors,
        categorical_vars: coxCatVars,
        ref_categories: coxRefCats,
        mode: coxMode,
      };
    } else if (method === "psm") {
      if (!psmTreatCol) { setError("请选择处理变量"); return; }
      if (!psmCovariates.length) { setError("请至少选择一个协变量"); return; }
      if (psmOutcomeType === "survival" && (!psmTimeCol || !psmEventCol)) {
        setError("生存结局需要选择时间变量和事件变量"); return;
      }
      params = {
        treatment_col: psmTreatCol,
        covariates: psmCovariates,
        outcome_type: psmOutcomeType,
        method: psmMethod,
        ratio: psmRatio,
        with_replacement: psmWithReplacement,
        ...(psmOutcomeCol ? { outcome_col: psmOutcomeCol } : {}),
        ...(psmOutcomeType === "survival" ? { time_col: psmTimeCol, event_col: psmEventCol } : {}),
        ...(psmCaliper.trim() ? { caliper: parseFloat(psmCaliper) } : {}),
      };
    } else if (method === "prediction") {
      if (predModelType === "logistic") {
        if (!predOutcome) { setError("请选择因变量"); return; }
        if (!predPredictors.length) { setError("请至少选择一个预测因子"); return; }
      } else {
        if (!predTimeCol) { setError("请选择时间变量"); return; }
        if (!predEventCol) { setError("请选择事件变量"); return; }
        if (!predPredictors.length) { setError("请至少选择一个预测因子"); return; }
      }
      const nBoot = parseInt(predNBoot, 10);
      const trainR = parseFloat(predTrainRatio);
      const tp = parseFloat(predTimePoint);
      params = {
        model_type: predModelType,
        predictors: predPredictors,
        categorical_vars: predCatVars,
        ref_categories: predRefCats,
        validation: predValidation,
        n_bootstrap: isNaN(nBoot) ? 1000 : nBoot,
        train_ratio: isNaN(trainR) ? 0.7 : trainR,
        stepwise: predStepwise,
        ...(predModelType === "logistic"
          ? { outcome: predOutcome }
          : {
              time_col: predTimeCol,
              event_col: predEventCol,
              ...(predTimePoint.trim() && !isNaN(tp) ? { time_point: tp } : {}),
            }),
      };
    } else if (method === "forest_plot") {
      if (!fpExposure) { setError("请选择暴露变量"); return; }
      if (!fpSubgroupVars.length) { setError("请至少选择一个亚组变量"); return; }
      if (fpModelType === "logistic" && !fpOutcome) { setError("请选择因变量"); return; }
      if (fpModelType === "linear"   && !fpOutcome) { setError("请选择因变量"); return; }
      if (fpModelType === "cox" && (!fpTimeCol || !fpEventCol)) { setError("请选择时间变量和事件变量"); return; }
      params = {
        model_type: fpModelType,
        exposure: fpExposure,
        covariates: fpCovariates,
        subgroup_vars: fpSubgroupVars,
        categorical_vars: fpCatVars,
        ...(fpModelType !== "cox" ? { outcome: fpOutcome } : { time_col: fpTimeCol, event_col: fpEventCol }),
      };
    } else if (method === "rcs") {
      if (!rcsExposure) { setError("请选择暴露变量（连续变量）"); return; }
      if (rcsModelType === "logistic" && !rcsOutcome) { setError("请选择因变量"); return; }
      if (rcsModelType === "linear"   && !rcsOutcome) { setError("请选择因变量"); return; }
      if (rcsModelType === "cox" && (!rcsTimeCol || !rcsEventCol)) { setError("请选择时间变量和事件变量"); return; }
      const refVal = parseFloat(rcsRefValue);
      params = {
        model_type: rcsModelType,
        exposure: rcsExposure,
        covariates: rcsCovariates,
        n_knots: rcsNKnots,
        ...(rcsModelType !== "cox" ? { outcome: rcsOutcome } : { time_col: rcsTimeCol, event_col: rcsEventCol }),
        ...(rcsRefValue.trim() && !isNaN(refVal) ? { ref_value: refVal } : {}),
      };
    } else if (method === "threshold") {
      if (!thrExposure) { setError("请选择暴露变量（连续变量）"); return; }
      if (thrModelType === "logistic" && !thrOutcome) { setError("请选择因变量"); return; }
      if (thrModelType === "linear"   && !thrOutcome) { setError("请选择因变量"); return; }
      if (thrModelType === "cox" && (!thrTimeCol || !thrEventCol)) { setError("请选择时间变量和事件变量"); return; }
      params = {
        model_type: thrModelType,
        exposure: thrExposure,
        covariates: thrCovariates,
        n_steps: 100,
        n_bootstrap: 100,
        ...(thrModelType !== "cox" ? { outcome: thrOutcome } : { time_col: thrTimeCol, event_col: thrEventCol }),
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
    if (method === "logistic_reg_adjusted") return !!lrlaOutcome && !!lrlaExposure;
    if (method === "survival") return !!survTimeCol && !!survEventCol;
    if (method === "cox_reg") return !!coxTimeCol && !!coxEventCol && coxPredictors.length > 0;
    if (method === "psm") {
      const survOk = psmOutcomeType !== "survival" || (!!psmTimeCol && !!psmEventCol);
      return !!psmTreatCol && psmCovariates.length > 0 && survOk;
    }
    if (method === "prediction") {
      if (predModelType === "logistic") return !!predOutcome && predPredictors.length > 0;
      return !!predTimeCol && !!predEventCol && predPredictors.length > 0;
    }
    if (method === "forest_plot") {
      const outOk = fpModelType === "cox" ? (!!fpTimeCol && !!fpEventCol) : !!fpOutcome;
      return outOk && !!fpExposure && fpSubgroupVars.length > 0;
    }
    if (method === "rcs") {
      const outOk = rcsModelType === "cox" ? (!!rcsTimeCol && !!rcsEventCol) : !!rcsOutcome;
      return outOk && !!rcsExposure;
    }
    if (method === "threshold") {
      const outOk = thrModelType === "cox" ? (!!thrTimeCol && !!thrEventCol) : !!thrOutcome;
      return outOk && !!thrExposure;
    }
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
      ) : method === "logistic_reg_adjusted" ? (
        <LogisticRegAdjustedConfig
          upload={upload}
          outcome={lrlaOutcome}
          setOutcome={(v) => {
            setLrlaOutcome(v);
            setLrlaExposure((e) => e === v ? "" : e);
            setLrlaCovariates((c) => c.filter((x) => x !== v));
            setLrlaCatVars((c) => c.filter((x) => x !== v));
            setLrlaModel2Covs((c) => c.filter((x) => x !== v));
            setLrlaStratifyVar((s) => s === v ? "" : s);
            setLrlaInteractionVar((i) => i === v ? "" : i);
          }}
          exposure={lrlaExposure}
          setExposure={(v) => {
            setLrlaExposure(v);
            setLrlaCovariates((c) => c.filter((x) => x !== v));
            setLrlaCatVars((c) => c.filter((x) => x !== v));
            setLrlaModel2Covs((c) => c.filter((x) => x !== v));
          }}
          covariates={lrlaCovariates}
          setCovariates={(covs) => {
            setLrlaCovariates(covs);
            setLrlaCatVars((c) => c.filter((x) => covs.includes(x)));
            setLrlaModel2Covs((m2) => m2.filter((c) => covs.includes(c)));
          }}
          catVars={lrlaCatVars}
          setCatVars={setLrlaCatVars}
          refCats={lrlaRefCats}
          setRefCats={setLrlaRefCats}
          model2Covs={lrlaModel2Covs}
          setModel2Covs={setLrlaModel2Covs}
          stratifyVar={lrlaStratifyVar}
          setStratifyVar={setLrlaStratifyVar}
          interactionVar={lrlaInteractionVar}
          setInteractionVar={setLrlaInteractionVar}
          mode={lrlaMode}
          setMode={setLrlaMode}
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
      ) : method === "survival" ? (
        <SurvivalConfig
          upload={upload}
          timeCol={survTimeCol}
          setTimeCol={(v) => { setSurvTimeCol(v); if (survEventCol === v) setSurvEventCol(""); if (survGroupCol === v) setSurvGroupCol(""); }}
          eventCol={survEventCol}
          setEventCol={(v) => { setSurvEventCol(v); if (survTimeCol === v) setSurvTimeCol(""); }}
          groupCol={survGroupCol}
          setGroupCol={setSurvGroupCol}
          timePoints={survTimePoints}
          setTimePoints={setSurvTimePoints}
        />
      ) : method === "cox_reg" ? (
        <CoxRegConfig
          upload={upload}
          timeCol={coxTimeCol}
          setTimeCol={(v) => { setCoxTimeCol(v); if (coxEventCol === v) setCoxEventCol(""); setCoxPredictors((p) => p.filter((c) => c !== v)); }}
          eventCol={coxEventCol}
          setEventCol={(v) => { setCoxEventCol(v); if (coxTimeCol === v) setCoxTimeCol(""); setCoxPredictors((p) => p.filter((c) => c !== v)); }}
          predictors={coxPredictors}
          setPredictors={(preds) => { setCoxPredictors(preds); setCoxCatVars((c) => c.filter((x) => preds.includes(x))); }}
          catVars={coxCatVars}
          setCatVars={setCoxCatVars}
          refCats={coxRefCats}
          setRefCats={setCoxRefCats}
          mode={coxMode}
          setMode={setCoxMode}
        />
      ) : method === "psm" ? (
        <PSMConfig
          upload={upload}
          treatCol={psmTreatCol}
          setTreatCol={(v) => { setPsmTreatCol(v); setPsmCovariates((c) => c.filter((x) => x !== v)); setPsmOutcomeCol((o) => o === v ? "" : o); }}
          covariates={psmCovariates}
          setCovariates={(covs) => { setPsmCovariates(covs.filter((c) => c !== psmTreatCol)); }}
          outcomeCol={psmOutcomeCol}
          setOutcomeCol={setPsmOutcomeCol}
          outcomeType={psmOutcomeType}
          setOutcomeType={setPsmOutcomeType}
          timeCol={psmTimeCol}
          setTimeCol={setPsmTimeCol}
          eventCol={psmEventCol}
          setEventCol={setPsmEventCol}
          method={psmMethod}
          setMethod={setPsmMethod}
          caliper={psmCaliper}
          setCaliper={setPsmCaliper}
          ratio={psmRatio}
          setRatio={setPsmRatio}
          withReplacement={psmWithReplacement}
          setWithReplacement={setPsmWithReplacement}
        />
      ) : method === "prediction" ? (
        <PredictionConfig
          upload={upload}
          modelType={predModelType}
          setModelType={(t) => {
            setPredModelType(t);
            setPredOutcome(""); setPredTimeCol(""); setPredEventCol("");
            setPredPredictors([]); setPredCatVars([]); setPredRefCats({});
          }}
          outcome={predOutcome}
          setOutcome={(v) => { setPredOutcome(v); setPredPredictors((p) => p.filter((c) => c !== v)); }}
          timeCol={predTimeCol}
          setTimeCol={(v) => { setPredTimeCol(v); setPredPredictors((p) => p.filter((c) => c !== v)); if (predEventCol === v) setPredEventCol(""); }}
          eventCol={predEventCol}
          setEventCol={(v) => { setPredEventCol(v); setPredPredictors((p) => p.filter((c) => c !== v)); if (predTimeCol === v) setPredTimeCol(""); }}
          predictors={predPredictors}
          setPredictors={(preds) => { setPredPredictors(preds); setPredCatVars((c) => c.filter((x) => preds.includes(x))); }}
          catVars={predCatVars}
          setCatVars={setPredCatVars}
          refCats={predRefCats}
          setRefCats={setPredRefCats}
          validation={predValidation}
          setValidation={setPredValidation}
          nBoot={predNBoot}
          setNBoot={setPredNBoot}
          trainRatio={predTrainRatio}
          setTrainRatio={setPredTrainRatio}
          timePoint={predTimePoint}
          setTimePoint={setPredTimePoint}
          stepwise={predStepwise}
          setStepwise={setPredStepwise}
        />
      ) : method === "forest_plot" ? (
        <ForestPlotConfig
          upload={upload}
          modelType={fpModelType}
          setModelType={(t) => { setFpModelType(t); setFpOutcome(""); setFpTimeCol(""); setFpEventCol(""); setFpExposure(""); setFpCovariates([]); setFpSubgroupVars([]); setFpCatVars([]); }}
          outcome={fpOutcome}
          setOutcome={(v) => { setFpOutcome(v); setFpCovariates((c) => c.filter((x) => x !== v)); setFpSubgroupVars((s) => s.filter((x) => x !== v)); }}
          timeCol={fpTimeCol}
          setTimeCol={(v) => { setFpTimeCol(v); if (fpEventCol === v) setFpEventCol(""); setFpCovariates((c) => c.filter((x) => x !== v)); }}
          eventCol={fpEventCol}
          setEventCol={(v) => { setFpEventCol(v); if (fpTimeCol === v) setFpTimeCol(""); setFpCovariates((c) => c.filter((x) => x !== v)); }}
          exposure={fpExposure}
          setExposure={(v) => { setFpExposure(v); setFpCovariates((c) => c.filter((x) => x !== v)); setFpSubgroupVars((s) => s.filter((x) => x !== v)); }}
          covariates={fpCovariates}
          setCovariates={setFpCovariates}
          subgroupVars={fpSubgroupVars}
          setSubgroupVars={setFpSubgroupVars}
          catVars={fpCatVars}
          setCatVars={setFpCatVars}
        />
      ) : method === "rcs" ? (
        <RCSConfig
          upload={upload}
          modelType={rcsModelType}
          setModelType={(t) => { setRcsModelType(t); setRcsOutcome(""); setRcsTimeCol(""); setRcsEventCol(""); setRcsExposure(""); setRcsCovariates([]); }}
          outcome={rcsOutcome}
          setOutcome={(v) => { setRcsOutcome(v); setRcsCovariates((c) => c.filter((x) => x !== v)); if (rcsExposure === v) setRcsExposure(""); }}
          timeCol={rcsTimeCol}
          setTimeCol={(v) => { setRcsTimeCol(v); if (rcsEventCol === v) setRcsEventCol(""); setRcsCovariates((c) => c.filter((x) => x !== v)); }}
          eventCol={rcsEventCol}
          setEventCol={(v) => { setRcsEventCol(v); if (rcsTimeCol === v) setRcsTimeCol(""); setRcsCovariates((c) => c.filter((x) => x !== v)); }}
          exposure={rcsExposure}
          setExposure={(v) => { setRcsExposure(v); setRcsCovariates((c) => c.filter((x) => x !== v)); }}
          covariates={rcsCovariates}
          setCovariates={setRcsCovariates}
          nKnots={rcsNKnots}
          setNKnots={setRcsNKnots}
          refValue={rcsRefValue}
          setRefValue={setRcsRefValue}
        />
      ) : method === "threshold" ? (
        <ThresholdConfig
          upload={upload}
          modelType={thrModelType}
          setModelType={(t) => { setThrModelType(t); setThrOutcome(""); setThrTimeCol(""); setThrEventCol(""); setThrExposure(""); setThrCovariates([]); }}
          outcome={thrOutcome}
          setOutcome={(v) => { setThrOutcome(v); setThrCovariates((c) => c.filter((x) => x !== v)); if (thrExposure === v) setThrExposure(""); }}
          timeCol={thrTimeCol}
          setTimeCol={(v) => { setThrTimeCol(v); if (thrEventCol === v) setThrEventCol(""); setThrCovariates((c) => c.filter((x) => x !== v)); }}
          eventCol={thrEventCol}
          setEventCol={(v) => { setThrEventCol(v); if (thrTimeCol === v) setThrTimeCol(""); setThrCovariates((c) => c.filter((x) => x !== v)); }}
          exposure={thrExposure}
          setExposure={(v) => { setThrExposure(v); setThrCovariates((c) => c.filter((x) => x !== v)); }}
          covariates={thrCovariates}
          setCovariates={setThrCovariates}
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

// ─────────────────────────────────────────────────────────────────────────────
// Logistic 回归控制混杂配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface LogisticRegAdjustedConfigProps {
  upload: UploadResponse;
  outcome: string;
  setOutcome: (v: string) => void;
  exposure: string;
  setExposure: (v: string) => void;
  covariates: string[];
  setCovariates: (v: string[]) => void;
  catVars: string[];
  setCatVars: React.Dispatch<React.SetStateAction<string[]>>;
  refCats: Record<string, string>;
  setRefCats: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  model2Covs: string[];
  setModel2Covs: React.Dispatch<React.SetStateAction<string[]>>;
  stratifyVar: string;
  setStratifyVar: (v: string) => void;
  interactionVar: string;
  setInteractionVar: (v: string) => void;
  mode: "both" | "crude" | "adjusted";
  setMode: (v: "both" | "crude" | "adjusted") => void;
}

function LogisticRegAdjustedConfig({
  upload,
  outcome, setOutcome,
  exposure, setExposure,
  covariates, setCovariates,
  catVars, setCatVars,
  refCats, setRefCats,
  model2Covs, setModel2Covs,
  stratifyVar, setStratifyVar,
  interactionVar, setInteractionVar,
  mode, setMode,
}: LogisticRegAdjustedConfigProps) {
  const cols = upload.column_names;
  const nonOutcome = cols.filter((c) => c !== outcome);
  const nonExposureOrOutcome = cols.filter((c) => c !== outcome && c !== exposure);

  const toggleCov = (col: string) =>
    setCovariates(covariates.includes(col) ? covariates.filter((c) => c !== col) : [...covariates, col]);

  const toggleCat = (col: string) => {
    setCatVars((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);
    if (catVars.includes(col)) {
      setRefCats((prev) => { const next = { ...prev }; delete next[col]; return next; });
    }
  };

  const toggleModel2Cov = (col: string) =>
    setModel2Covs((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

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
    { value: "both"     as const, label: "粗模型 + 调整模型", desc: "输出 Model 1/2/3 逐步对比" },
    { value: "crude"    as const, label: "仅粗模型",           desc: "不调整协变量" },
    { value: "adjusted" as const, label: "仅调整模型",         desc: "仅输出全调整结果" },
  ];

  return (
    <div className="space-y-6">
      {/* ── 因变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">因变量（二分类结局）<span className="text-destructive text-xs font-normal ml-1">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">支持 0/1、True/False 或两个字符串类别</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${outcome === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="lrla_outcome" value={col} checked={outcome === col} onChange={() => setOutcome(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* ── 暴露变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">暴露变量（主要研究自变量）<span className="text-destructive text-xs font-normal ml-1">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">数值型变量；OR 变化将被全程追踪</p>
        </div>
        {!outcome ? (
          <p className="text-xs text-muted-foreground">请先选择因变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {nonOutcome.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${exposure === col ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20 font-medium" : "border-border hover:border-emerald-400/60"}`}>
                <input type="radio" name="lrla_exposure" value={col} checked={exposure === col} onChange={() => setExposure(col)} className="accent-emerald-500" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      {/* ── 协变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">协变量（需控制的混杂因素）</h2>
          <p className="text-xs text-muted-foreground mt-0.5">可混合连续变量与分类变量；分类变量需在下方标记</p>
        </div>
        {!exposure ? (
          <p className="text-xs text-muted-foreground">请先选择暴露变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {nonExposureOrOutcome.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${covariates.includes(col) ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20 font-medium" : "border-border hover:border-amber-400/60"}`}>
                <input type="checkbox" checked={covariates.includes(col)} onChange={() => toggleCov(col)} className="accent-amber-500" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      {/* ── 分类变量标记 ── */}
      {covariates.length > 0 && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">分类变量标记</h2>
            <p className="text-xs text-muted-foreground mt-0.5">勾选后自动 dummy 编码；可选择参考组（默认取第一个类别）</p>
          </div>
          <div className="space-y-2">
            {covariates.map((col) => {
              const isCat = catVars.includes(col);
              const previewVals = isCat ? getPreviewValues(col) : [];
              return (
                <div key={col} className={`flex flex-wrap items-center gap-3 px-4 py-3 rounded-lg border transition-colors ${isCat ? "border-amber-400 bg-amber-50 dark:bg-amber-950/20" : "border-border"}`}>
                  <label className="flex items-center gap-2 text-sm cursor-pointer min-w-[140px]">
                    <input type="checkbox" checked={isCat} onChange={() => toggleCat(col)} className="accent-amber-500" />
                    <span className="font-medium">{col}</span>
                    {isCat && <span className="text-xs px-1.5 py-0.5 rounded bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200">分类</span>}
                  </label>
                  {isCat && previewVals.length > 0 && (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-xs text-muted-foreground">参考组：</span>
                      <select
                        value={refCats[col] ?? previewVals[0]}
                        onChange={(e) => setRefCats((prev) => ({ ...prev, [col]: e.target.value }))}
                        className="text-sm px-2 py-1 border border-border rounded-lg focus:outline-none focus:ring-1 focus:ring-primary/30 bg-background"
                      >
                        {previewVals.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── Model 2 协变量（可选）── */}
      {covariates.length >= 2 && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">Model 2 协变量（可选）</h2>
            <p className="text-xs text-muted-foreground mt-0.5">从协变量中选择进入 Model 2（部分调整）；留空则自动取前一半</p>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {covariates.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${model2Covs.includes(col) ? "border-violet-500 bg-violet-50 dark:bg-violet-950/20 font-medium" : "border-border hover:border-violet-400/60"}`}>
                <input type="checkbox" checked={model2Covs.includes(col)} onChange={() => toggleModel2Cov(col)} className="accent-violet-500" />
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

      {/* ── 分层变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">分层变量（可选）</h2>
          <p className="text-xs text-muted-foreground mt-0.5">按各水平分别拟合模型，输出各层 OR + Breslow-Day 同质性检验</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          <label className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${!stratifyVar ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
            <input type="radio" name="lrla_stratify" value="" checked={!stratifyVar} onChange={() => setStratifyVar("")} className="accent-primary" />
            <span className="text-muted-foreground">不分层</span>
          </label>
          {cols.filter((c) => c !== outcome && c !== exposure).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${stratifyVar === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="lrla_stratify" value={col} checked={stratifyVar === col} onChange={() => setStratifyVar(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* ── 交互项变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">交互项变量（可选）</h2>
          <p className="text-xs text-muted-foreground mt-0.5">检验 暴露 × 该变量 的交互效应（效应修饰），输出交互 OR 和 p 值</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          <label className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${!interactionVar ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
            <input type="radio" name="lrla_interaction" value="" checked={!interactionVar} onChange={() => setInteractionVar("")} className="accent-primary" />
            <span className="text-muted-foreground">不检验</span>
          </label>
          {cols.filter((c) => c !== outcome).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${interactionVar === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="lrla_interaction" value={col} checked={interactionVar === col} onChange={() => setInteractionVar(col)} className="accent-primary" />
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
            <label key={value} className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${mode === value ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}>
              <input type="radio" name="lrla_mode" value={value} checked={mode === value} onChange={() => setMode(value)} className="accent-primary mt-0.5" />
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
          {catVars.length > 0 && (
            <p>分类变量：<span className="font-medium text-foreground">{catVars.map((v) => `${v}（参考：${refCats[v] ?? "默认"}）`).join("、")}</span></p>
          )}
          {stratifyVar && <p>分层变量：<span className="font-medium text-foreground">{stratifyVar}</span></p>}
          {interactionVar && <p>交互项：<span className="font-medium text-foreground">{exposure} × {interactionVar}</span></p>}
          <p>模式：<span className="font-medium text-foreground">{MODES.find((m) => m.value === mode)?.label}</span></p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 生存分析配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface SurvivalConfigProps {
  upload: UploadResponse;
  timeCol: string;
  setTimeCol: (v: string) => void;
  eventCol: string;
  setEventCol: (v: string) => void;
  groupCol: string;
  setGroupCol: (v: string) => void;
  timePoints: string;
  setTimePoints: (v: string) => void;
}

function SurvivalConfig({
  upload, timeCol, setTimeCol, eventCol, setEventCol,
  groupCol, setGroupCol, timePoints, setTimePoints,
}: SurvivalConfigProps) {
  const cols = upload.column_names;

  return (
    <div className="space-y-6">
      {/* 时间变量 */}
      <section className="space-y-2">
        <div>
          <h2 className="font-semibold">时间变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">连续数值，单位可为天 / 月 / 年</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${timeCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="surv_time" value={col} checked={timeCol === col} onChange={() => setTimeCol(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* 事件变量 */}
      <section className="space-y-2">
        <div>
          <h2 className="font-semibold">事件变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">0 = 删失，1 = 事件发生</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.filter((c) => c !== timeCol).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${eventCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="surv_event" value={col} checked={eventCol === col} onChange={() => setEventCol(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* 分组变量（可选） */}
      <section className="space-y-2">
        <div>
          <h2 className="font-semibold">分组变量 <span className="text-muted-foreground text-xs font-normal">可选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">提供分组变量时将绘制多组 KM 曲线并进行 Log-rank 检验</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          <label className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${groupCol === "" ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
            <input type="radio" name="surv_group" value="" checked={groupCol === ""} onChange={() => setGroupCol("")} className="accent-primary" />
            <span className="text-muted-foreground">不分组</span>
          </label>
          {cols.filter((c) => c !== timeCol && c !== eventCol).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${groupCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="surv_group" value={col} checked={groupCol === col} onChange={() => setGroupCol(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* 指定时间点 */}
      <section className="space-y-2">
        <h2 className="font-semibold">指定时间点生存率 <span className="text-muted-foreground text-xs font-normal">可选</span></h2>
        <p className="text-xs text-muted-foreground">用逗号分隔，如 <code className="bg-muted px-1 rounded text-xs">365,1095,1825</code>（天）或 <code className="bg-muted px-1 rounded text-xs">1,3,5</code>（年）</p>
        <input
          type="text"
          value={timePoints}
          onChange={(e) => setTimePoints(e.target.value)}
          placeholder="例：365,1095,1825"
          className="w-full max-w-xs px-3 py-2 rounded-lg border border-border text-sm bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
        />
      </section>

      {/* 配置摘要 */}
      {timeCol && eventCol && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>时间变量：<span className="font-medium text-foreground">{timeCol}</span></p>
          <p>事件变量：<span className="font-medium text-foreground">{eventCol}</span></p>
          {groupCol && <p>分组变量：<span className="font-medium text-foreground">{groupCol}</span></p>}
          {timePoints && <p>时间点：<span className="font-medium text-foreground">{timePoints}</span></p>}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Cox 回归配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface CoxRegConfigProps {
  upload: UploadResponse;
  timeCol: string;
  setTimeCol: (v: string) => void;
  eventCol: string;
  setEventCol: (v: string) => void;
  predictors: string[];
  setPredictors: (v: string[]) => void;
  catVars: string[];
  setCatVars: React.Dispatch<React.SetStateAction<string[]>>;
  refCats: Record<string, string>;
  setRefCats: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  mode: "both" | "univariate" | "multivariate";
  setMode: (v: "both" | "univariate" | "multivariate") => void;
}

function CoxRegConfig({
  upload, timeCol, setTimeCol, eventCol, setEventCol,
  predictors, setPredictors, catVars, setCatVars,
  refCats, setRefCats, mode, setMode,
}: CoxRegConfigProps) {
  const cols = upload.column_names;
  const reservedCols = new Set([timeCol, eventCol]);

  const togglePredictor = (col: string) => {
    if (reservedCols.has(col)) return;
    const next = predictors.includes(col)
      ? predictors.filter((c) => c !== col)
      : [...predictors, col];
    setPredictors(next);
  };

  const toggleCatVar = (col: string) => {
    if (!predictors.includes(col)) return;
    setCatVars((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const COX_MODES: { value: "both" | "univariate" | "multivariate"; label: string; desc: string }[] = [
    { value: "both",          label: "单变量 + 多变量",   desc: "分别呈现，便于对比" },
    { value: "univariate",    label: "仅单变量",           desc: "逐一分析每个自变量" },
    { value: "multivariate",  label: "仅多变量",           desc: "所有自变量同时纳入" },
  ];

  return (
    <div className="space-y-6">
      {/* 时间变量 */}
      <section className="space-y-2">
        <div>
          <h2 className="font-semibold">时间变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">连续数值，必须 &gt; 0</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${timeCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="cox_time" value={col} checked={timeCol === col} onChange={() => setTimeCol(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* 事件变量 */}
      <section className="space-y-2">
        <div>
          <h2 className="font-semibold">事件变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">0 = 删失，1 = 事件发生</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.filter((c) => c !== timeCol).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${eventCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="cox_event" value={col} checked={eventCol === col} onChange={() => setEventCol(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* 自变量（多选） */}
      <section className="space-y-2">
        <div>
          <h2 className="font-semibold">自变量 <span className="text-destructive text-xs font-normal">必选，可多选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">不能与时间变量 / 事件变量重复</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.filter((c) => !reservedCols.has(c)).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${predictors.includes(col) ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="checkbox" checked={predictors.includes(col)} onChange={() => togglePredictor(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">已选 {predictors.length} 个</p>
      </section>

      {/* 分类变量标记 + 参考组 */}
      {predictors.length > 0 && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">标记分类变量 <span className="text-muted-foreground text-xs font-normal">可选</span></h2>
            <p className="text-xs text-muted-foreground mt-0.5">标记为分类变量的列将进行 dummy 编码</p>
          </div>
          <div className="space-y-2">
            {predictors.map((col) => {
              const isCat = catVars.includes(col);
              return (
                <div key={col} className="flex items-center gap-3 flex-wrap">
                  <label className="flex items-center gap-2 text-sm cursor-pointer min-w-[140px]">
                    <input type="checkbox" checked={isCat} onChange={() => toggleCatVar(col)} className="accent-primary" />
                    <span className="font-medium">{col}</span>
                    {isCat && <span className="text-xs bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 px-1.5 py-0.5 rounded">分类</span>}
                  </label>
                  {isCat && (
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-muted-foreground">参考组：</span>
                      <input
                        type="text"
                        value={refCats[col] ?? ""}
                        onChange={(e) => setRefCats((prev) => ({ ...prev, [col]: e.target.value }))}
                        placeholder="默认第一类"
                        className="px-2 py-1 rounded border border-border bg-background text-sm w-28 focus:outline-none focus:ring-1 focus:ring-primary/40"
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 分析模式 */}
      <section className="space-y-2">
        <h2 className="font-semibold">分析模式</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {COX_MODES.map(({ value, label, desc }) => (
            <label key={value} className={`flex items-start gap-3 px-4 py-3 rounded-lg border cursor-pointer transition-colors ${mode === value ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}>
              <input type="radio" name="cox_mode" value={value} checked={mode === value} onChange={() => setMode(value)} className="accent-primary mt-0.5" />
              <div>
                <p className="text-sm font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">{desc}</p>
              </div>
            </label>
          ))}
        </div>
      </section>

      {/* 配置摘要 */}
      {timeCol && eventCol && predictors.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>时间变量：<span className="font-medium text-foreground">{timeCol}</span></p>
          <p>事件变量：<span className="font-medium text-foreground">{eventCol}</span></p>
          <p>自变量（{predictors.length}）：<span className="font-medium text-foreground">{predictors.join("、")}</span></p>
          {catVars.length > 0 && (
            <p>分类变量：<span className="font-medium text-foreground">{catVars.map((v) => `${v}（参考：${refCats[v] ?? "默认"}）`).join("、")}</span></p>
          )}
          <p>模式：<span className="font-medium text-foreground">{COX_MODES.find((m) => m.value === mode)?.label}</span></p>
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

// ─────────────────────────────────────────────────────────────────────────────
// PSM 配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface PSMConfigProps {
  upload: UploadResponse;
  treatCol: string; setTreatCol: (v: string) => void;
  covariates: string[]; setCovariates: (v: string[]) => void;
  outcomeCol: string; setOutcomeCol: (v: string) => void;
  outcomeType: "continuous" | "binary" | "survival"; setOutcomeType: (v: "continuous" | "binary" | "survival") => void;
  timeCol: string; setTimeCol: (v: string) => void;
  eventCol: string; setEventCol: (v: string) => void;
  method: "nearest" | "caliper" | "optimal"; setMethod: (v: "nearest" | "caliper" | "optimal") => void;
  caliper: string; setCaliper: (v: string) => void;
  ratio: 1 | 2 | 3; setRatio: (v: 1 | 2 | 3) => void;
  withReplacement: boolean; setWithReplacement: (v: boolean) => void;
}

function PSMConfig({
  upload,
  treatCol, setTreatCol,
  covariates, setCovariates,
  outcomeCol, setOutcomeCol,
  outcomeType, setOutcomeType,
  timeCol, setTimeCol,
  eventCol, setEventCol,
  method, setMethod,
  caliper, setCaliper,
  ratio, setRatio,
  withReplacement, setWithReplacement,
}: PSMConfigProps) {
  const cols = upload.column_names;
  const nonTreat = cols.filter((c) => c !== treatCol);

  const toggleCov = (col: string) =>
    setCovariates(covariates.includes(col) ? covariates.filter((c) => c !== col) : [...covariates, col]);

  const OUTCOME_TYPES = [
    { value: "continuous" as const, label: "连续结局", desc: "配对 Wilcoxon 检验" },
    { value: "binary"     as const, label: "二分类结局", desc: "McNemar 检验" },
    { value: "survival"   as const, label: "生存结局", desc: "分层 Cox + Log-rank" },
  ];

  const METHODS = [
    { value: "nearest" as const, label: "最近邻（Greedy）", desc: "贪心搜索最近 PS 对，速度快" },
    { value: "caliper" as const, label: "Caliper 约束",     desc: "限制最大 PS 差（默认 0.2×SD）" },
    { value: "optimal" as const, label: "最优匹配（1:1）",  desc: "最小化总体 PS 距离，仅支持 1:1" },
  ];

  return (
    <div className="space-y-6">
      {/* ── 处理变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">处理变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">二分类变量，区分处理组与对照组（如 treatment、group）</p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {cols.map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${treatCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="psm_treat" value={col} checked={treatCol === col} onChange={() => setTreatCol(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
      </section>

      {/* ── 协变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">协变量 <span className="text-destructive text-xs font-normal">必选，可多选</span></h2>
          <p className="text-xs text-muted-foreground mt-0.5">用于估计倾向性得分（PS）的混杂变量；分类变量自动 dummy 编码</p>
        </div>
        {!treatCol ? (
          <p className="text-xs text-muted-foreground">请先选择处理变量</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {nonTreat.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${covariates.includes(col) ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20 font-medium" : "border-border hover:border-amber-400/60"}`}>
                <input type="checkbox" checked={covariates.includes(col)} onChange={() => toggleCov(col)} className="accent-amber-500" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        )}
        {covariates.length > 0 && <p className="text-xs text-muted-foreground">已选 {covariates.length} 个协变量</p>}
      </section>

      {/* ── 结局变量 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">结局变量 <span className="text-muted-foreground text-xs font-normal">可选（匹配后处理效应估计）</span></h2>
        </div>
        <div className="flex flex-wrap gap-2 mb-3">
          {OUTCOME_TYPES.map((ot) => (
            <button
              key={ot.value}
              type="button"
              onClick={() => setOutcomeType(ot.value)}
              className={`px-3 py-1.5 rounded-lg border text-sm transition-colors ${outcomeType === ot.value ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}
            >
              {ot.label}
              <span className="text-xs text-muted-foreground ml-1.5">({ot.desc})</span>
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          <label className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${!outcomeCol ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
            <input type="radio" name="psm_outcome" value="" checked={!outcomeCol} onChange={() => setOutcomeCol("")} className="accent-primary" />
            <span className="text-muted-foreground">不分析结局</span>
          </label>
          {nonTreat.filter((c) => !covariates.includes(c)).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${outcomeCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="radio" name="psm_outcome" value={col} checked={outcomeCol === col} onChange={() => setOutcomeCol(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>

        {/* 生存结局：时间 + 事件变量 */}
        {outcomeType === "survival" && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
            <div className="space-y-2">
              <p className="text-sm font-medium">时间变量 <span className="text-destructive text-xs">必选</span></p>
              <div className="flex flex-wrap gap-2">
                {cols.filter((c) => c !== treatCol).map((col) => (
                  <label key={col} className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border text-xs cursor-pointer transition-colors ${timeCol === col ? "border-teal-500 bg-teal-50 dark:bg-teal-950/20 font-medium" : "border-border hover:border-teal-400/60"}`}>
                    <input type="radio" name="psm_time" value={col} checked={timeCol === col} onChange={() => setTimeCol(col)} className="accent-teal-500" />
                    {col}
                  </label>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">事件变量 <span className="text-destructive text-xs">必选</span></p>
              <div className="flex flex-wrap gap-2">
                {cols.filter((c) => c !== treatCol && c !== timeCol).map((col) => (
                  <label key={col} className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border text-xs cursor-pointer transition-colors ${eventCol === col ? "border-teal-500 bg-teal-50 dark:bg-teal-950/20 font-medium" : "border-border hover:border-teal-400/60"}`}>
                    <input type="radio" name="psm_event" value={col} checked={eventCol === col} onChange={() => setEventCol(col)} className="accent-teal-500" />
                    {col}
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ── 匹配参数 ── */}
      <section className="space-y-4">
        <h2 className="font-semibold">匹配参数</h2>

        {/* 匹配方法 */}
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">匹配方法</p>
          <div className="flex flex-wrap gap-2">
            {METHODS.map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setMethod(m.value)}
                className={`px-3 py-1.5 rounded-lg border text-sm transition-colors ${method === m.value ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}
              >
                {m.label}
                <span className="text-xs text-muted-foreground ml-1.5">— {m.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* 匹配比例 + 放回 + caliper */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">匹配比例</p>
            <div className="flex gap-2">
              {([1, 2, 3] as const).map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRatio(r)}
                  disabled={method === "optimal" && r !== 1}
                  className={`px-3 py-1.5 rounded-lg border text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${ratio === r ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}
                >
                  1:{r}
                </button>
              ))}
            </div>
            {method === "optimal" && <p className="text-xs text-muted-foreground">最优匹配仅支持 1:1</p>}
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">放回匹配</p>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={withReplacement}
                onChange={(e) => setWithReplacement(e.target.checked)}
                className="accent-primary w-4 h-4"
              />
              允许同一对照重复使用
            </label>
          </div>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Caliper 值 <span className="text-xs">（留空 = 0.2×SD(PS)）</span></p>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={caliper}
              onChange={(e) => setCaliper(e.target.value)}
              placeholder="如 0.05"
              className="w-full px-3 py-1.5 rounded-lg border border-border bg-background text-sm focus:outline-none focus:ring-1 focus:ring-primary/40"
            />
          </div>
        </div>
      </section>

      {/* 配置预览 */}
      {treatCol && covariates.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>处理变量：<span className="font-medium text-foreground">{treatCol}</span></p>
          <p>协变量（{covariates.length}）：<span className="font-medium text-foreground">{covariates.join("、")}</span></p>
          {outcomeCol && <p>结局变量：<span className="font-medium text-foreground">{outcomeCol}（{OUTCOME_TYPES.find((o) => o.value === outcomeType)?.label}）</span></p>}
          {outcomeType === "survival" && <p>时间 / 事件：<span className="font-medium text-foreground">{timeCol || "—"} / {eventCol || "—"}</span></p>}
          <p>匹配：<span className="font-medium text-foreground">{METHODS.find((m) => m.value === method)?.label} · 1:{ratio} · {withReplacement ? "放回" : "不放回"}</span></p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 临床预测模型配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface PredictionConfigProps {
  upload: UploadResponse;
  modelType: "logistic" | "cox";
  setModelType: (t: "logistic" | "cox") => void;
  outcome: string;
  setOutcome: (v: string) => void;
  timeCol: string;
  setTimeCol: (v: string) => void;
  eventCol: string;
  setEventCol: (v: string) => void;
  predictors: string[];
  setPredictors: (p: string[]) => void;
  catVars: string[];
  setCatVars: (v: string[]) => void;
  refCats: Record<string, string>;
  setRefCats: (r: Record<string, string>) => void;
  validation: "internal_bootstrap" | "split" | "cross_validation";
  setValidation: (v: "internal_bootstrap" | "split" | "cross_validation") => void;
  nBoot: string;
  setNBoot: (v: string) => void;
  trainRatio: string;
  setTrainRatio: (v: string) => void;
  timePoint: string;
  setTimePoint: (v: string) => void;
  stepwise: boolean;
  setStepwise: (v: boolean) => void;
}

function PredictionConfig({
  upload,
  modelType, setModelType,
  outcome, setOutcome,
  timeCol, setTimeCol,
  eventCol, setEventCol,
  predictors, setPredictors,
  catVars, setCatVars,
  refCats, setRefCats,
  validation, setValidation,
  nBoot, setNBoot,
  trainRatio, setTrainRatio,
  timePoint, setTimePoint,
  stepwise, setStepwise,
}: PredictionConfigProps) {
  const cols = upload.column_names;

  const usedCols = modelType === "logistic"
    ? [outcome]
    : [timeCol, eventCol];
  const availablePredictors = cols.filter((c) => !usedCols.includes(c) || predictors.includes(c));

  const togglePredictor = (col: string) =>
    setPredictors(
      predictors.includes(col) ? predictors.filter((c) => c !== col) : [...predictors, col]
    );

  const toggleCatVar = (col: string) =>
    setCatVars(
      catVars.includes(col) ? catVars.filter((c) => c !== col) : [...catVars, col]
    );

  return (
    <div className="space-y-6">

      {/* ── 模型类型 ── */}
      <section className="space-y-3">
        <h2 className="font-semibold">模型类型</h2>
        <div className="flex gap-3">
          {(["logistic", "cox"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setModelType(t)}
              className={`px-4 py-2 rounded-lg border text-sm transition-colors ${
                modelType === t ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
              }`}
            >
              {t === "logistic" ? "Logistic 回归（二分类结局）" : "Cox 回归（生存结局）"}
            </button>
          ))}
        </div>
      </section>

      {/* ── 结局变量（Logistic） ── */}
      {modelType === "logistic" && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">因变量 <span className="text-destructive text-xs font-normal">必选，二分类</span></h2>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {cols.map((col) => (
              <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${outcome === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                <input type="radio" name="pred_outcome" value={col} checked={outcome === col} onChange={() => setOutcome(col)} className="accent-primary" />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))}
          </div>
        </section>
      )}

      {/* ── 时间 + 事件变量（Cox） ── */}
      {modelType === "cox" && (
        <section className="space-y-4">
          <div>
            <h2 className="font-semibold">时间 & 事件变量 <span className="text-destructive text-xs font-normal">必选</span></h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">时间变量</p>
              <div className="grid grid-cols-2 gap-1.5">
                {cols.map((col) => (
                  <label key={col} className={`flex items-center gap-2 px-2 py-1.5 rounded-md border text-xs cursor-pointer ${timeCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                    <input type="radio" name="pred_time" value={col} checked={timeCol === col} onChange={() => setTimeCol(col)} className="accent-primary" />
                    <span className="truncate" title={col}>{col}</span>
                  </label>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">事件变量（0=删失，1=事件）</p>
              <div className="grid grid-cols-2 gap-1.5">
                {cols.map((col) => (
                  <label key={col} className={`flex items-center gap-2 px-2 py-1.5 rounded-md border text-xs cursor-pointer ${eventCol === col ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
                    <input type="radio" name="pred_event" value={col} checked={eventCol === col} onChange={() => setEventCol(col)} className="accent-primary" />
                    <span className="truncate" title={col}>{col}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground whitespace-nowrap">预测时间点</label>
            <input
              type="number"
              placeholder="如 365（天）"
              value={timePoint}
              onChange={(e) => setTimePoint(e.target.value)}
              className="w-40 px-3 py-1.5 border border-border rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
            <span className="text-xs text-muted-foreground">留空则使用时间中位数</span>
          </div>
        </section>
      )}

      {/* ── 预测因子 ── */}
      <section className="space-y-3">
        <div>
          <h2 className="font-semibold">预测因子 <span className="text-destructive text-xs font-normal">必选，可多选</span></h2>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {availablePredictors.filter((c) => !usedCols.includes(c)).map((col) => (
            <label key={col} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer transition-colors ${predictors.includes(col) ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"}`}>
              <input type="checkbox" checked={predictors.includes(col)} onChange={() => togglePredictor(col)} className="accent-primary" />
              <span className="truncate" title={col}>{col}</span>
            </label>
          ))}
        </div>
        {predictors.length > 0 && <p className="text-xs text-muted-foreground">已选 {predictors.length} 个预测因子</p>}
      </section>

      {/* ── 分类变量标记 ── */}
      {predictors.length > 0 && (
        <section className="space-y-3">
          <div>
            <h2 className="font-semibold">分类变量标记 <span className="text-muted-foreground text-xs font-normal">可选</span></h2>
            <p className="text-xs text-muted-foreground mt-0.5">标记后自动 dummy 编码；可指定参考组</p>
          </div>
          <div className="space-y-2">
            {predictors.map((col) => (
              <div key={col} className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-sm cursor-pointer min-w-[140px]">
                  <input
                    type="checkbox"
                    checked={catVars.includes(col)}
                    onChange={() => toggleCatVar(col)}
                    className="accent-amber-500"
                  />
                  <span className={catVars.includes(col) ? "font-medium text-amber-700" : ""}>{col}</span>
                </label>
                {catVars.includes(col) && (
                  <input
                    type="text"
                    placeholder="参考组（留空=默认）"
                    value={refCats[col] ?? ""}
                    onChange={(e) =>
                      setRefCats({ ...refCats, [col]: e.target.value })
                    }
                    className="flex-1 max-w-[180px] px-2 py-1 border border-border rounded-md text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── 验证方式 ── */}
      <section className="space-y-3">
        <h2 className="font-semibold">内部验证方式</h2>
        <div className="flex flex-wrap gap-2">
          {[
            { value: "internal_bootstrap" as const, label: "Bootstrap 验证", desc: "Harrell 校正法" },
            { value: "split"              as const, label: "Split 验证",      desc: "训练集 / 测试集" },
            { value: "cross_validation"   as const, label: "5-fold 交叉验证", desc: "K-fold CV" },
          ].map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setValidation(opt.value)}
              className={`px-3 py-1.5 rounded-lg border text-sm transition-colors ${
                validation === opt.value ? "border-primary bg-primary/5 font-medium" : "border-border hover:border-primary/40"
              }`}
            >
              {opt.label}
              <span className="text-xs text-muted-foreground ml-1.5">({opt.desc})</span>
            </button>
          ))}
        </div>

        {validation === "internal_bootstrap" && (
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground">Bootstrap 次数</label>
            <input
              type="number"
              value={nBoot}
              onChange={(e) => setNBoot(e.target.value)}
              min={100}
              max={5000}
              className="w-28 px-3 py-1.5 border border-border rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        )}
        {validation === "split" && (
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground">训练集比例</label>
            <input
              type="number"
              value={trainRatio}
              onChange={(e) => setTrainRatio(e.target.value)}
              min={0.5}
              max={0.9}
              step={0.05}
              className="w-28 px-3 py-1.5 border border-border rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        )}
      </section>

      {/* ── 逐步变量筛选 ── */}
      <section className="space-y-2">
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={stepwise}
            onChange={(e) => setStepwise(e.target.checked)}
            className="accent-primary w-4 h-4"
          />
          <div>
            <span className="font-semibold text-sm">向后逐步变量筛选（AIC-based）</span>
            <p className="text-xs text-muted-foreground">自动移除不显著变量，可能延长计算时间</p>
          </div>
        </label>
      </section>

      {/* 配置预览 */}
      {predictors.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>模型类型：<span className="font-medium text-foreground">{modelType === "logistic" ? "Logistic 回归" : "Cox 回归"}</span></p>
          {modelType === "logistic" && outcome && <p>因变量：<span className="font-medium text-foreground">{outcome}</span></p>}
          {modelType === "cox" && <p>时间 / 事件：<span className="font-medium text-foreground">{timeCol || "—"} / {eventCol || "—"}{timePoint ? `  · 预测时间点：${timePoint}` : ""}</span></p>}
          <p>预测因子（{predictors.length}）：<span className="font-medium text-foreground">{predictors.join("、")}</span></p>
          {catVars.length > 0 && <p>分类变量：<span className="font-medium text-foreground">{catVars.join("、")}</span></p>}
          <p>验证：<span className="font-medium text-foreground">{validation === "internal_bootstrap" ? `Bootstrap (n=${nBoot})` : validation === "split" ? `Split (${Math.round(parseFloat(trainRatio) * 100)}%)` : "5-fold CV"}</span>
            {stepwise && <span className="ml-2 text-amber-600">+ 逐步筛选</span>}
          </p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 亚组分析 & 森林图 配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface ForestPlotConfigProps {
  upload: UploadResponse;
  modelType: "logistic" | "cox" | "linear";
  setModelType: (v: "logistic" | "cox" | "linear") => void;
  outcome: string; setOutcome: (v: string) => void;
  timeCol: string; setTimeCol: (v: string) => void;
  eventCol: string; setEventCol: (v: string) => void;
  exposure: string; setExposure: (v: string) => void;
  covariates: string[]; setCovariates: React.Dispatch<React.SetStateAction<string[]>>;
  subgroupVars: string[]; setSubgroupVars: React.Dispatch<React.SetStateAction<string[]>>;
  catVars: string[]; setCatVars: React.Dispatch<React.SetStateAction<string[]>>;
}

function ForestPlotConfig({
  upload, modelType, setModelType,
  outcome, setOutcome, timeCol, setTimeCol, eventCol, setEventCol,
  exposure, setExposure, covariates, setCovariates,
  subgroupVars, setSubgroupVars, catVars, setCatVars,
}: ForestPlotConfigProps) {
  const cols = upload.column_names;
  const ColSelect = ({ val, onChange, label, options }: { val: string; onChange: (v: string) => void; label: string; options: string[] }) => (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      <select value={val} onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-border px-3 py-2 text-sm bg-background">
        <option value="">— 请选择 —</option>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
  const multiToggle = (col: string, setArr: React.Dispatch<React.SetStateAction<string[]>>) =>
    setArr((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <h2 className="font-semibold">亚组分析 & 森林图配置</h2>
        <div className="flex gap-3">
          {(["logistic", "cox", "linear"] as const).map((t) => (
            <label key={t} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer ${modelType === t ? "border-primary bg-primary/5" : "border-border"}`}>
              <input type="radio" name="fp_model" value={t} checked={modelType === t} onChange={() => setModelType(t)} className="accent-primary" />
              {t === "logistic" ? "Logistic 回归" : t === "cox" ? "Cox 回归" : "线性回归"}
            </label>
          ))}
        </div>
      </div>

      {modelType !== "cox"
        ? <ColSelect val={outcome} onChange={setOutcome} label="因变量（Outcome）" options={cols} />
        : <div className="grid grid-cols-2 gap-4">
            <ColSelect val={timeCol} onChange={setTimeCol} label="时间变量" options={cols} />
            <ColSelect val={eventCol} onChange={setEventCol} label="事件变量（0=删失, 1=事件）" options={cols.filter((c) => c !== timeCol)} />
          </div>}

      <ColSelect val={exposure} onChange={setExposure} label="暴露变量（Exposure）"
        options={cols.filter((c) => c !== outcome && c !== timeCol && c !== eventCol)} />

      <div className="space-y-2">
        <label className="text-sm font-medium">协变量（调整用，多选）</label>
        <div className="grid grid-cols-3 gap-2">
          {cols.filter((c) => c !== outcome && c !== timeCol && c !== eventCol && c !== exposure).map((c) => (
            <label key={c} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm cursor-pointer ${covariates.includes(c) ? "border-primary bg-primary/5" : "border-border"}`}>
              <input type="checkbox" checked={covariates.includes(c)} onChange={() => multiToggle(c, setCovariates)} className="accent-primary" />
              <span className="truncate">{c}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">亚组变量（多选）</label>
        <div className="grid grid-cols-3 gap-2">
          {cols.filter((c) => c !== exposure && c !== outcome && c !== timeCol && c !== eventCol).map((c) => (
            <label key={c} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm cursor-pointer ${subgroupVars.includes(c) ? "border-primary bg-primary/5" : "border-border"}`}>
              <input type="checkbox" checked={subgroupVars.includes(c)} onChange={() => multiToggle(c, setSubgroupVars)} className="accent-primary" />
              <span className="truncate">{c}</span>
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">数值变量自动按中位数分为高/低两组；勾选「分类变量」后按实际类别分层</p>
      </div>

      {subgroupVars.length > 0 && (
        <div className="space-y-2">
          <label className="text-sm font-medium">标记为分类亚组变量（按实际类别分层）</label>
          <div className="flex flex-wrap gap-2">
            {subgroupVars.map((c) => (
              <label key={c} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm cursor-pointer ${catVars.includes(c) ? "border-primary bg-primary/5" : "border-border"}`}>
                <input type="checkbox" checked={catVars.includes(c)} onChange={() => multiToggle(c, setCatVars)} className="accent-primary" />
                <span>{c}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {exposure && subgroupVars.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>模型：<span className="font-medium text-foreground">{modelType}</span> · 暴露：<span className="font-medium text-foreground">{exposure}</span></p>
          <p>亚组变量（{subgroupVars.length}）：<span className="font-medium text-foreground">{subgroupVars.join("、")}</span></p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RCS 曲线配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface RCSConfigProps {
  upload: UploadResponse;
  modelType: "logistic" | "cox" | "linear";
  setModelType: (v: "logistic" | "cox" | "linear") => void;
  outcome: string; setOutcome: (v: string) => void;
  timeCol: string; setTimeCol: (v: string) => void;
  eventCol: string; setEventCol: (v: string) => void;
  exposure: string; setExposure: (v: string) => void;
  covariates: string[]; setCovariates: React.Dispatch<React.SetStateAction<string[]>>;
  nKnots: 3 | 4 | 5; setNKnots: (v: 3 | 4 | 5) => void;
  refValue: string; setRefValue: (v: string) => void;
}

function RCSConfig({
  upload, modelType, setModelType,
  outcome, setOutcome, timeCol, setTimeCol, eventCol, setEventCol,
  exposure, setExposure, covariates, setCovariates,
  nKnots, setNKnots, refValue, setRefValue,
}: RCSConfigProps) {
  const cols = upload.column_names;
  const ColSelect = ({ val, onChange, label, options }: { val: string; onChange: (v: string) => void; label: string; options: string[] }) => (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      <select value={val} onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-border px-3 py-2 text-sm bg-background">
        <option value="">— 请选择 —</option>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
  const multiToggle = (col: string, setArr: React.Dispatch<React.SetStateAction<string[]>>) =>
    setArr((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <h2 className="font-semibold">RCS 曲线配置</h2>
        <div className="flex gap-3">
          {(["logistic", "cox", "linear"] as const).map((t) => (
            <label key={t} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer ${modelType === t ? "border-primary bg-primary/5" : "border-border"}`}>
              <input type="radio" name="rcs_model" value={t} checked={modelType === t} onChange={() => setModelType(t)} className="accent-primary" />
              {t === "logistic" ? "Logistic" : t === "cox" ? "Cox" : "线性"}
            </label>
          ))}
        </div>
      </div>

      {modelType !== "cox"
        ? <ColSelect val={outcome} onChange={setOutcome} label="因变量" options={cols} />
        : <div className="grid grid-cols-2 gap-4">
            <ColSelect val={timeCol} onChange={setTimeCol} label="时间变量" options={cols} />
            <ColSelect val={eventCol} onChange={setEventCol} label="事件变量" options={cols.filter((c) => c !== timeCol)} />
          </div>}

      <ColSelect val={exposure} onChange={setExposure} label="暴露变量（连续变量）"
        options={cols.filter((c) => c !== outcome && c !== timeCol && c !== eventCol)} />

      <div className="space-y-2">
        <label className="text-sm font-medium">协变量（多选）</label>
        <div className="grid grid-cols-3 gap-2">
          {cols.filter((c) => c !== outcome && c !== timeCol && c !== eventCol && c !== exposure).map((c) => (
            <label key={c} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm cursor-pointer ${covariates.includes(c) ? "border-primary bg-primary/5" : "border-border"}`}>
              <input type="checkbox" checked={covariates.includes(c)} onChange={() => multiToggle(c, setCovariates)} className="accent-primary" />
              <span className="truncate">{c}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">节点数（Knots）</label>
          <div className="flex gap-2">
            {([3, 4, 5] as const).map((n) => (
              <label key={n} className={`flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm cursor-pointer ${nKnots === n ? "border-primary bg-primary/5 font-semibold" : "border-border"}`}>
                <input type="radio" name="rcs_knots" value={n} checked={nKnots === n} onChange={() => setNKnots(n)} className="accent-primary" />
                {n}
              </label>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">默认 4 节点（Harrell 推荐）</p>
        </div>
        <div className="space-y-1">
          <label className="text-sm font-medium">参考值（留空则用中位数）</label>
          <input type="number" step="any" value={refValue} onChange={(e) => setRefValue(e.target.value)}
            placeholder="例如：25"
            className="w-full rounded-lg border border-border px-3 py-2 text-sm bg-background" />
        </div>
      </div>

      {exposure && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>模型：<span className="font-medium text-foreground">{modelType}</span> · 暴露：<span className="font-medium text-foreground">{exposure}</span> · 节点数：<span className="font-medium text-foreground">{nKnots}</span></p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 阈值效应分析配置组件
// ─────────────────────────────────────────────────────────────────────────────

interface ThresholdConfigProps {
  upload: UploadResponse;
  modelType: "logistic" | "cox" | "linear";
  setModelType: (v: "logistic" | "cox" | "linear") => void;
  outcome: string; setOutcome: (v: string) => void;
  timeCol: string; setTimeCol: (v: string) => void;
  eventCol: string; setEventCol: (v: string) => void;
  exposure: string; setExposure: (v: string) => void;
  covariates: string[]; setCovariates: React.Dispatch<React.SetStateAction<string[]>>;
}

function ThresholdConfig({
  upload, modelType, setModelType,
  outcome, setOutcome, timeCol, setTimeCol, eventCol, setEventCol,
  exposure, setExposure, covariates, setCovariates,
}: ThresholdConfigProps) {
  const cols = upload.column_names;
  const ColSelect = ({ val, onChange, label, options }: { val: string; onChange: (v: string) => void; label: string; options: string[] }) => (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      <select value={val} onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-border px-3 py-2 text-sm bg-background">
        <option value="">— 请选择 —</option>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
  const multiToggle = (col: string, setArr: React.Dispatch<React.SetStateAction<string[]>>) =>
    setArr((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <h2 className="font-semibold">阈值效应分析配置</h2>
        <p className="text-xs text-muted-foreground">分段线性回归自动搜索最佳拐点，Bootstrap 估计置信区间</p>
        <div className="flex gap-3">
          {(["logistic", "cox", "linear"] as const).map((t) => (
            <label key={t} className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm cursor-pointer ${modelType === t ? "border-primary bg-primary/5" : "border-border"}`}>
              <input type="radio" name="thr_model" value={t} checked={modelType === t} onChange={() => setModelType(t)} className="accent-primary" />
              {t === "logistic" ? "Logistic" : t === "cox" ? "Cox" : "线性"}
            </label>
          ))}
        </div>
      </div>

      {modelType !== "cox"
        ? <ColSelect val={outcome} onChange={setOutcome} label="因变量" options={cols} />
        : <div className="grid grid-cols-2 gap-4">
            <ColSelect val={timeCol} onChange={setTimeCol} label="时间变量" options={cols} />
            <ColSelect val={eventCol} onChange={setEventCol} label="事件变量" options={cols.filter((c) => c !== timeCol)} />
          </div>}

      <ColSelect val={exposure} onChange={setExposure} label="暴露变量（连续变量）"
        options={cols.filter((c) => c !== outcome && c !== timeCol && c !== eventCol)} />

      <div className="space-y-2">
        <label className="text-sm font-medium">协变量（多选）</label>
        <div className="grid grid-cols-3 gap-2">
          {cols.filter((c) => c !== outcome && c !== timeCol && c !== eventCol && c !== exposure).map((c) => (
            <label key={c} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm cursor-pointer ${covariates.includes(c) ? "border-primary bg-primary/5" : "border-border"}`}>
              <input type="checkbox" checked={covariates.includes(c)} onChange={() => multiToggle(c, setCovariates)} className="accent-primary" />
              <span className="truncate">{c}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="rounded-lg bg-amber-50 border border-amber-200 px-4 py-3 text-xs text-amber-700">
        <p className="font-medium">⏱ 计算时间提示</p>
        <p className="mt-0.5">搜索 100 个候选拐点 + 100 次 Bootstrap，大样本约需 30–60 秒。</p>
      </div>

      {exposure && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs text-muted-foreground space-y-1">
          <p>模型：<span className="font-medium text-foreground">{modelType}</span> · 暴露：<span className="font-medium text-foreground">{exposure}</span></p>
          {covariates.length > 0 && <p>协变量：<span className="font-medium text-foreground">{covariates.join("、")}</span></p>}
        </div>
      )}
    </div>
  );
}
