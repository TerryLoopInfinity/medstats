import Link from "next/link";

const steps = [
  {
    step: "01",
    title: "上传数据",
    desc: "支持 CSV / XLSX，最大 10 MB，自动预览前 20 行",
    href: "/upload",
    cta: "开始上传",
  },
  {
    step: "02",
    title: "选择分析方法",
    desc: "从 13 种统计方法中选择，勾选需要分析的变量",
    href: "/analyze",
    cta: "配置分析",
  },
  {
    step: "03",
    title: "查看结果",
    desc: "自动生成统计表格与可视化图表",
    href: "/result",
    cta: "查看结果",
  },
];

const methods = [
  "统计描述 & 正态性检验", "三线表生成", "差异性分析", "相关 & 线性回归",
  "Logistic 回归", "生存分析 & Cox 回归", "倾向性评分匹配",
  "临床预测模型", "亚组分析 & 森林图", "RCS 曲线", "阈值效应分析",
  "中介分析", "样本量计算",
];

export default function HomePage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-12 space-y-16">
      <section className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">临床医学统计分析平台</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          上传数据集，选择分析方法，自动完成统计计算并生成图表。
        </p>
        <Link
          href="/upload"
          className="inline-block mt-2 px-6 py-2.5 bg-primary text-primary-foreground rounded-lg font-medium hover:opacity-90 transition-opacity"
        >
          立即开始
        </Link>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-6">使用流程</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {steps.map(({ step, title, desc, href, cta }) => (
            <div key={step} className="border border-border rounded-xl p-6 space-y-3">
              <span className="text-3xl font-bold text-muted-foreground/40">{step}</span>
              <h3 className="font-semibold text-lg">{title}</h3>
              <p className="text-sm text-muted-foreground">{desc}</p>
              <Link href={href} className="inline-block text-sm underline underline-offset-2 hover:text-primary transition-colors">
                {cta} →
              </Link>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-4">支持的分析方法</h2>
        <div className="flex flex-wrap gap-2">
          {methods.map((m, i) => (
            <span
              key={m}
              className={`px-3 py-1 rounded-full text-sm border ${
                i === 0
                  ? "bg-primary text-primary-foreground border-primary"
                  : "border-border text-muted-foreground"
              }`}
            >
              {m}
            </span>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-2">高亮项目已实现，其余持续开发中</p>
      </section>
    </div>
  );
}
