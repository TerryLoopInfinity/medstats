"use client";

import { useEffect, useRef } from "react";

export interface EChartsRendererProps {
  option: Record<string, unknown>;
  title?: string;
  height?: number;
  className?: string;
}

export function EChartsRenderer({
  option,
  title,
  height = 340,
  className = "",
}: EChartsRendererProps) {
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!divRef.current) return;
    const el = divRef.current;
    let chart: import("echarts").ECharts;

    // Dynamic import avoids SSR issues (echarts references DOM at init time)
    import("echarts").then((echarts) => {
      if (!el) return;
      chart = echarts.init(el, undefined, { renderer: "canvas" });
      chart.setOption(option);

      const handleResize = () => chart.resize();
      window.addEventListener("resize", handleResize);

      return () => {
        window.removeEventListener("resize", handleResize);
      };
    });

    return () => {
      // echarts may not be loaded yet when component unmounts quickly
      import("echarts").then((echarts) => {
        const existing = echarts.getInstanceByDom(el);
        existing?.dispose();
      });
    };
    // option は JSON なので stringify で変化検知
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(option)]);

  return (
    <div className={`space-y-1 ${className}`}>
      {title && <p className="text-xs text-muted-foreground font-medium">{title}</p>}
      <div ref={divRef} style={{ height, width: "100%" }} />
    </div>
  );
}
