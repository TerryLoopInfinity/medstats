# MedStats — 临床医学统计分析平台

在线统计分析工具，用户上传 CSV 数据后选择分析方法，后端自动完成统计计算并返回结果表格与图表。

## 技术栈

- **前端**: Next.js 14+ (App Router), TypeScript, Tailwind CSS, shadcn/ui, Apache ECharts
- **后端**: Python 3.11+, FastAPI, Pydantic v2
- **统计计算**: pandas, scipy, statsmodels, lifelines, scikit-learn, rpy2 (桥接 R 包)
- **数据库**: SQLite (开发阶段) → PostgreSQL (生产环境)
- **部署**: Docker, 前端 Vercel, 后端 Railway / Fly.io

## 项目结构

```
medstats/
├── frontend/                # Next.js 应用
│   ├── app/
│   │   ├── upload/          # 数据上传页
│   │   ├── analyze/         # 分析配置页
│   │   └── result/          # 结果展示页
│   ├── components/
│   │   ├── ui/              # shadcn/ui 组件
│   │   ├── charts/          # ECharts 图表封装组件
│   │   └── data/            # 数据表格、变量选择器等
│   └── lib/
│       ├── api.ts           # 后端 API 调用封装
│       └── types.ts         # TypeScript 类型定义
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI 入口
│   │   ├── api/
│   │   │   └── routes/      # 每个分析方法一个路由文件
│   │   ├── core/
│   │   │   ├── config.py    # 配置管理
│   │   │   └── security.py  # 文件上传安全校验
│   │   ├── models/          # Pydantic 请求/响应模型
│   │   ├── services/        # 业务逻辑层
│   │   └── stats/           # 统计计算引擎（核心）
│   │       ├── descriptive.py      # 统计描述与正态性检验
│   │       ├── table_one.py        # 三线表生成
│   │       ├── ttest.py            # 差异性分析
│   │       ├── hypothesis.py       # 假设检验
│   │       ├── correlation.py      # 相关与线性回归
│   │       ├── linear_reg.py       # 线性回归（含混杂控制）
│   │       ├── logistic_reg.py     # logistic 回归（含混杂控制）
│   │       ├── survival.py         # 生存分析 (lifelines)
│   │       ├── cox_reg.py          # Cox 回归
│   │       ├── psm.py             # 倾向性得分匹配
│   │       ├── prediction.py       # 临床预测模型
│   │       ├── forest_plot.py      # 亚组分析与森林图
│   │       ├── rcs.py             # RCS 曲线（rpy2 调用 R rms 包）
│   │       ├── threshold.py        # 阈值效应分析
│   │       ├── mediation.py        # 中介分析
│   │       └── sample_size.py      # 样本量计算
│   ├── tests/               # pytest 测试
│   └── data/
│       └── examples/        # 示例数据集 (CSV)
├── docker-compose.yml
└── CLAUDE.md
```

## 关键约定

### 后端

- 每个统计模块必须是独立 Python 文件，接收 `pd.DataFrame` + 参数 dict，返回标准化 `AnalysisResult`
- `AnalysisResult` 包含: `tables`（列表，每项含 headers + rows）、`charts`（列表，每项含 chart_type + option）、`summary`（文字结论）、`warnings`（数据质量提醒）
- 所有 API 路由使用 Pydantic v2 做入参校验，不允许裸 dict 传参
- 文件上传限制 10MB，仅接受 .csv 和 .xlsx
- 对用户上传的数据做安全校验：检查行列数上限、检测异常编码、拒绝可执行内容
- 使用 `rpy2` 调用 R 时，集中在 `backend/app/stats/r_bridge.py` 统一管理 R 环境初始化，不要在每个模块里单独初始化
- 统计 p 值保留 3 位小数，置信区间用 95% 作为默认值

### 前端

- 所有 ECharts 图表封装为独立 React 组件，放在 `components/charts/`
- 图表组件接收后端返回的 `option` JSON 直接渲染，不在前端做统计计算
- 使用 `useMemo` 缓存大数据量图表的 option 计算
- 数据表格使用 shadcn/ui 的 DataTable，支持排序和导出
- 所有页面支持中文界面，但变量名和代码使用英文

### 数据流

```
用户上传 CSV → 前端预览（前 20 行）→ 用户选择分析方法和变量
→ 前端发送 POST /api/analyze/{method} (file_id + params)
→ 后端读取文件、校验、调用对应 stats 模块
→ 返回 AnalysisResult JSON
→ 前端渲染表格 + 图表
```

## 常用命令

```bash
# 前端
cd frontend && npm run dev          # 启动开发服务器 (port 3000)
cd frontend && npm run build        # 构建生产版本
cd frontend && npm run lint         # ESLint 检查

# 后端
cd backend && uvicorn app.main:app --reload    # 启动开发服务器 (port 8000)
cd backend && pytest                            # 运行测试
cd backend && pytest tests/stats/              # 只跑统计模块测试

# Docker
docker-compose up                   # 启动完整环境
```

## 开发顺序

按此顺序逐个实现功能模块，每完成一个确保前后端完整跑通：

1. 统计描述与正态性检验（最小可用版本，跑通完整数据流）
2. 三线表生成
3. 定量资料差异性分析 + 基本假设检验
4. 相关与线性回归
5. logistic 回归
6. 线性回归/logistic 回归控制混杂偏倚
7. 生存分析 + Cox 回归
8. 倾向性得分匹配
9. 临床预测模型
10. 亚组分析与森林图
11. RCS 曲线 + 阈值效应分析
12. 中介分析
13. 在线样本量计算

## 注意事项

- 不要在统计模块中 print 调试信息，使用 Python logging
- 每个新的统计模块必须附带至少一个 pytest 测试用例，使用 `data/examples/` 中的示例数据
- 前端不做任何统计计算，所有计算在后端完成
- ECharts option 由后端生成并返回，前端只负责渲染
- rpy2 相关代码需要处理 R 环境不可用的情况（graceful fallback 或明确报错）
