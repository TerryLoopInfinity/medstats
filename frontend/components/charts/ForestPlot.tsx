"use client";

/**
 * ForestPlot — 可复用的横向森林图组件
 *
 * 接收后端返回的 option（含 forestData 数组），通过 ECharts custom 系列
 * 绘制带误差棒的标准森林图。后续 Cox 回归、亚组分析直接复用此组件。
 *
 * option 格式（backend 生成）：
 * {
 *   forestData: Array<{
 *     label: string;       // Y 轴标签（模型名 / 分层名）
 *     beta:  number;       // 点估计
 *     ci_lo: number;       // 置信区间下限
 *     ci_hi: number;       // 置信区间上限
 *     p:     string;       // p 值字符串
 *     n?:    number;       // 样本量（可选）
 *     is_note?: boolean;   // 是否为备注行（不绘制效应点）
 *   }>;
 *   nullLine: number;      // 参考线位置，通常为 0（β）或 1（OR/HR）
 *   xLabel:  string;       // X 轴标签
 *   title:   string;       // 图表标题
 * }
 */

import { useEffect, useMemo, useRef } from "react";

export interface ForestDataPoint {
  label: string;
  beta: number;
  ci_lo: number;
  ci_hi: number;
  p: string;
  n?: number;
  is_note?: boolean;
}

export interface ForestPlotOption {
  forestData: ForestDataPoint[];
  nullLine?: number;
  xLabel?: string;
  title?: string;
}

interface ForestPlotProps {
  /** 后端返回的 chart.option（含 forestData 等字段） */
  option: Record<string, unknown>;
  height?: number;
  className?: string;
}

export function ForestPlot({ option, height = 420, className = "" }: ForestPlotProps) {
  const divRef = useRef<HTMLDivElement>(null);

  // 将后端 option 格式转换为 ECharts option（含 renderItem 函数）
  const echartsOption = useMemo(() => {
    const forestData = (option.forestData ?? []) as ForestDataPoint[];
    const nullLine = (option.nullLine as number) ?? 0;
    const xLabel = (option.xLabel as string) ?? "效应估计（95% CI）";
    const chartTitle = (option.title as string) ?? "";

    if (forestData.length === 0) return null;

    // 过滤有效绘制点（非备注行 + 有有限数值）
    const plotPoints = forestData.filter(
      (d) => !d.is_note && isFinite(d.beta) && isFinite(d.ci_lo) && isFinite(d.ci_hi)
    );

    const labels = forestData.map((d) => d.label);
    const allValues = plotPoints.flatMap((d) => [d.beta, d.ci_lo, d.ci_hi]);
    const xMin = Math.min(...allValues, nullLine);
    const xMax = Math.max(...allValues, nullLine);
    const xPad = (xMax - xMin) * 0.12 || 0.5;

    // custom 系列数据：[ci_lo, ci_hi, y_index, beta]
    const customData = forestData.map((d, i) => {
      if (d.is_note || !isFinite(d.beta)) return null;
      return {
        value: [d.ci_lo, d.ci_hi, i, d.beta],
        // 携带 p 值供 tooltip 使用
        p: d.p,
        n: d.n,
      };
    });

    return {
      title: chartTitle
        ? { text: chartTitle, left: "center", textStyle: { fontSize: 13, fontWeight: "bold" } }
        : undefined,
      tooltip: {
        trigger: "item",
        formatter: (params: { data?: { p?: string; n?: number; value?: number[] } }) => {
          if (!params.data?.value) return "";
          const [ciLo, ciHi, yIdx, beta] = params.data.value as number[];
          const label = labels[yIdx as number] ?? "";
          const pStr = params.data.p ?? "—";
          const nStr = params.data.n != null ? `n = ${params.data.n}` : "";
          return [
            `<b>${label}</b>`,
            `β = ${beta.toFixed(4)}`,
            `95% CI [${ciLo.toFixed(4)}, ${ciHi.toFixed(4)}]`,
            `p = ${pStr}`,
            nStr,
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
      grid: {
        left: "28%",
        right: "8%",
        top: chartTitle ? "12%" : "5%",
        bottom: "10%",
      },
      xAxis: {
        type: "value",
        name: xLabel,
        nameLocation: "center",
        nameGap: 28,
        min: xMin - xPad,
        max: xMax + xPad,
        axisLine: { show: true },
        splitLine: { lineStyle: { type: "dashed" } },
        // 参考线（null line）
        markLine: undefined, // handled in series
      },
      yAxis: {
        type: "category",
        data: labels,
        inverse: true,
        axisLabel: {
          width: 160,
          overflow: "truncate",
          fontSize: 12,
        },
      },
      series: [
        // ── 参考线（vertical null line）
        {
          type: "line",
          markLine: {
            silent: true,
            symbol: ["none", "none"],
            lineStyle: { color: "#999", type: "dashed", width: 1.5 },
            data: [{ xAxis: nullLine }],
            label: { show: false },
          },
          data: [],
        },
        // ── 森林图主体（CI 棒 + 菱形点估计）
        {
          type: "custom",
          renderItem: renderForestItem,
          data: customData.filter(Boolean),
          encode: { x: [0, 1], y: 2 },
          z: 5,
        },
      ],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(option)]);

  // 直接管理 ECharts 实例（避免 renderItem 函数因 JSON.stringify 丢失）
  useEffect(() => {
    if (!divRef.current || !echartsOption) return;
    const el = divRef.current;
    let chart: import("echarts").ECharts | null = null;

    import("echarts").then((echarts) => {
      if (!el) return;
      // 复用或新建实例
      chart = echarts.getInstanceByDom(el) ?? echarts.init(el, undefined, { renderer: "canvas" });
      chart.setOption(echartsOption, true);

      const onResize = () => chart?.resize();
      window.addEventListener("resize", onResize);
      return () => window.removeEventListener("resize", onResize);
    });

    return () => {
      import("echarts").then((echarts) => {
        const existing = echarts.getInstanceByDom(el);
        existing?.dispose();
      });
    };
  }, [echartsOption]);

  if (!option.forestData) {
    return (
      <div className="flex items-center justify-center h-24 text-muted-foreground text-sm">
        暂无森林图数据
      </div>
    );
  }

  return (
    <div className={className}>
      <div ref={divRef} style={{ height, width: "100%" }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ECharts custom renderItem：绘制 CI 线 + 菱形点
// 此函数在组件模块级定义，确保引用稳定（不会被 JSON.stringify 丢弃）
// ─────────────────────────────────────────────────────────────────────────────

function renderForestItem(
  _params: unknown,
  api: {
    value: (dim: number) => number;
    coord: (point: [number, number]) => [number, number];
    style: (opts?: Record<string, unknown>) => Record<string, unknown>;
    size: (val: [number, number]) => [number, number];
  }
) {
  const ciLo = api.value(0);
  const ciHi = api.value(1);
  const yIdx = api.value(2);
  const beta = api.value(3);

  const startPt = api.coord([ciLo, yIdx]);
  const endPt = api.coord([ciHi, yIdx]);
  const betaPt = api.coord([beta, yIdx]);

  // 像素高度（固定 8px 的误差棒端帽）
  const whiskerH = 8;
  const diamondW = 8; // 菱形半宽
  const diamondH = 6; // 菱形半高

  // 颜色：使用 p 值（data 中没有直接传，用固定色）
  const lineColor = "#5470c6";
  const diamondColor = "#5470c6";

  return {
    type: "group",
    children: [
      // 水平 CI 线
      {
        type: "line",
        shape: {
          x1: startPt[0],
          y1: startPt[1],
          x2: endPt[0],
          y2: endPt[1],
        },
        style: { stroke: lineColor, lineWidth: 2 },
      },
      // 左侧端帽
      {
        type: "line",
        shape: {
          x1: startPt[0],
          y1: startPt[1] - whiskerH / 2,
          x2: startPt[0],
          y2: startPt[1] + whiskerH / 2,
        },
        style: { stroke: lineColor, lineWidth: 2 },
      },
      // 右侧端帽
      {
        type: "line",
        shape: {
          x1: endPt[0],
          y1: endPt[1] - whiskerH / 2,
          x2: endPt[0],
          y2: endPt[1] + whiskerH / 2,
        },
        style: { stroke: lineColor, lineWidth: 2 },
      },
      // 菱形点估计
      {
        type: "polygon",
        shape: {
          points: [
            [betaPt[0] - diamondW, betaPt[1]],
            [betaPt[0], betaPt[1] - diamondH],
            [betaPt[0] + diamondW, betaPt[1]],
            [betaPt[0], betaPt[1] + diamondH],
          ],
        },
        style: { fill: diamondColor, stroke: diamondColor },
      },
    ],
  };
}
