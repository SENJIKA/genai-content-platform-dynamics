# 生成式人工智能与内容平台的动态治理

**副标题：** 从普惠式扩量到选择性扶持  
**作者：** 朱千子、朱胜豪  
**单位：** 对外经济贸易大学经济学院

本仓库包含论文的 LaTeX 正文、参考文献、原创数值求解器、派生结果和最终 PDF。
模型求解两期均衡与无限期平稳马尔可夫均衡，并区分统一非负支持、有符号转移支付
基准和零支付配额三种政策环境。

## 主要文件

- `main.tex`：论文正文。
- `references.bib`：参考文献。
- `scripts/solve_dynamic_equilibria.py`：原创两期向后归纳与无限期值函数迭代求解器。
- `output/computation/`：均衡摘要、政策切片、过渡路径与敏感性结果。
- `figures/`：正文使用的数值图形。
- `output/pdf/`：已编译的论文 PDF。

## 复现数值结果

需要 Python 3.11 或更高版本。首次运行：

```bash
python3 -m pip install -r requirements.txt
make data
make solve
```

`make data` 只下载 KuaiRec 的类别/文本表并核验官方 MD5。该文件用于计算标准化
Shannon 类别多样性，不用于校准观看、收入或合同参数。

## 编译论文

需要 XeLaTeX、BibTeX 和常用中文 LaTeX 宏包：

```bash
make all
```

编译顺序为 `xelatex -> bibtex -> xelatex -> xelatex`。

## 公开材料边界

公开仓库不重复分发下载的论文全文、第三方复现包、第三方教程代码或原始数据。
相关来源、DOI、校验值及本文借鉴边界记录在
`literature_recent_discrete_models/replication_code_sources/README.md`。
本文求解器独立实现，仅借鉴公开代码中的一般数值方法组织。

本仓库目前未授予统一的仓库级许可证。第三方材料仍受各自许可证和使用条款约束。
