# 错误与推理记录

**2026-03-30 `streamlit_shap_three_cohorts.py`：缺失 `_matplotlib_shap_barh_plot` + 误用 `python` 启动**
- **现象**：浏览器 **Connection error**；或终端用 **`python streamlit_shap_three_cohorts.py`**（Cursor「运行」按钮）→ 无 Web 服务，仅 **missing ScriptRunContext** 警告。
- **原因**：文件中 **调用了已删除的** `_matplotlib_shap_barh_plot`，SHAP 绘图会 **NameError**；**必须用** `streamlit run ...` 才能起服务。
- **修正**：**补回** `_matplotlib_shap_barh_plot`；**SHAP 图只写磁盘** `streamlit_shap_output/`；**`main()` 首行** `st.set_page_config`；**`if __name__ == "__main__"`** 内 **`get_script_run_ctx() is None` 则 stderr 提示并 `sys.exit(2)`**；侧栏按钮 **`width="stretch"`**。

**2026-03-30 Streamlit「自动退出」排查**
- **终端现象**：`Local URL` 打印后很快回到 `PS>` 且无 Python traceback 时，多为 **进程被外部结束**（关掉终端/标签、IDE 集成终端重置、休眠杀后台、OOM、杀毒）或 **原生库崩溃**（XGBoost CUDA/CPU 混用仅见 warning，极端情况可能拖垮进程）。
- **代码**：`main()` 里曾在 **`st.set_page_config` 之前**访问 **`st.session_state`**，违反 Streamlit「首个 `st` 调用应为 `set_page_config`」约定，已 **调换顺序**。
- **配置**：`.streamlit/config.toml` 增加 **`fileWatcherType = "none"`**，减轻 Windows 下大目录 **文件监视**带来的异常。
- **启动**：新增 **`scripts/run_streamlit_three_cohorts.ps1`**，用 **独立 PowerShell 窗口** 跑 `streamlit run`，避免 Cursor 内置终端会话结束时顺带结束子进程。

**2026-03-30 Streamlit 切换队列 → CONNECTING / 无结果**
- **原因**：每切到 A/B/C 都会 **重建 SHAP TreeExplainer**（+ 变换背景矩阵），**首轮可长达 1–3 分钟**；期间浏览器易显示 **CONNECTING**，界面可能仍短暂显示上一队列内容。
- **修正**：**`@st.cache_resource` `_cohort_shap_bundle`** 缓存「模型 + transform + explainer + X_all」；**`.streamlit/config.toml`** 设 **`server.websocketPingInterval = 30`**。

**2026-03-30 Streamlit 监测：`streamlit run` 后立刻回到 `PS>`**
- **脚本**：`scripts/run_streamlit_with_log.ps1` — `Tee-Object` 把 **stdout+stderr** 写入 **`streamlit_logs/streamlit_*.log`**，并设 **`--logger.level=debug`**、`PYTHONFAULTHANDLER=1`。进程静默退出时看 **日志最后几十行**。
- **`.gitignore`**：`streamlit_logs/`。

**2026-03-30 Streamlit：`Failed to fetch … static/js/index.*.js`（与 SHAP 预览）**
- **现象**：红字 **`TypeError: Failed to fetch dynamically imported module` … `index.B….js`**；此前 SHAP 用了 **`st.image` / `components.html`**。
- **原因**：这些组件会 **懒加载** 前端 chunk；缓存损坏、扩展、网络会导致 chunk 拉取失败（**非** Python/SHAP 算错）。
- **修正**：`_streamlit_show_shap_figure` **只用** **`st.markdown` + data-URI `<img>`** + 磁盘 PNG；**去掉** `st.image` 与 **`components.v1.html`**。说明文案同步 **`shap_trouble_body`**。

**2026-03-30 Streamlit SHAP「有文字无图」：`st.image` 不渲染**
- **现象**：**SHAP check**、保存路径、因子表正常，但页面无条形图；磁盘 **PNG** 用系统看图可打开。
- **原因**：部分环境下 **`st.image`** 依赖的前端资源/MediaFile URL 异常，图片元素为空；**非** SHAP 未计算。
- **修正（已被上条替代）**：曾试 iframe / PIL；最终以 **markdown data-URI** 为主并避免 `st.image`。

**2026-03-30 Streamlit：内联 base64 SHAP 图 → Connection error / CONNECTING**
- **现象**：风险与因子正常，**SHAP 区空**或随后整页 **Connection error**；终端可能无 Python traceback。
- **原因**：`st.markdown` 里 **超大 `data:image/png;base64,...`** 会放大 **WebSocket 单条消息**，易超时/断连；部分 Streamlit 还会清理 data-URI。
- **修正**：`_streamlit_show_shap_figure` 改为 **只写磁盘** `streamlit_shap_output/shap_cohort_{A|B|C}.png`，页面仅显示 **绝对路径 + 打开说明**；`savefig` **dpi≈88**、图宽约 **9 in** 控制体积。`.gitignore` 增加 `streamlit_shap_output/`。

**2026-03-30 Streamlit SHAP 后出现 `Failed to fetch … static/js/index.*.js`（在 SHAP check 文字之后）**
- **原因**：**`st.download_button` / `st.image` / `st.pyplot`** 等会拉取 **懒加载前端 chunk**；缓存损坏、扩展或网络会导致 **`TypeError: Failed to fetch dynamically imported module`**。超大 **inline base64** 还会诱发 **Connection error**（见上条「内联 base64」）。
- **修正（当前）**：`_streamlit_show_shap_figure` **只写磁盘** `streamlit_shap_output/shap_cohort_*.png` + 页面文字路径；**不用** st.image / st.pyplot / download_button / 内联 data-URI。

**2026-03-30 Streamlit SHAP：base≈−3、无图 + Connection error（XGBoost TreeExplainer 用 margin 非概率）**
- **现象**：**SHAP check** 里 **E[f(x)]** 为**大负数**（如 **−3.2**），与 **f(x)≈0.03** 明显不在同一尺度；力图空白或异常；浏览器 **CONNECTING / Connection error**。
- **原因**：`shap.TreeExplainer(model, X)` 默认常解释 **margin（logit）**；界面 **predict_proba** 是 **概率**。轴强行夹在 [0,1] 时条带被压没或渲染异常；重计算/大图也可能拖长请求，表现为断连。
- **修正**：`_build_explainer` 优先 **`TreeExplainer(..., model_output="probability")`**（多种 call 签名 try）；`_expected_value_proba_class1` 若 **expected_value ∉ [0,1]** 则视为无效；**`_base_value_for_force_plot`** 用 **与 Σφ、f(x) 可加一致** 的 baseline。`st.image` 优先 **`width="stretch"`**，`TypeError` 时回退 **`use_container_width`**。项目根 **`.streamlit/config.toml`**：`runOnSave = false`。用户需 **重启 Streamlit** 并再点 **Start assessment** 刷新缓存。

**2026-03-30 Streamlit SHAP 图不显示：markdown 里 data-URI `<img>` 被剥离**
- **现象**：**SHAP explanation** 有标题/说明但**没有图**，或整段像「没出来」。
- **原因**：不少 Streamlit 版本对 **`st.markdown(..., unsafe_allow_html=True)`** 里的 **`<img src="data:image/png;base64,...">`** 做 **HTML 清理**，**丢弃 data URI**，图就不渲染。
- **修正**：SHAP 图改用 **`st.image(io.BytesIO(raw_png), use_container_width=True)`**；失败时再回退 markdown base64。图前加 **`st.caption`** 显示 **base / Σφ / f(x)** 便于确认计算是否跑通。`tight_layout` 警告用 **`fig.subplots_adjust(...)`** 替代。

**2026-03-30 Streamlit SHAP 图：用户要 force plot 而非条形图**
- **需求**：界面「SHAP explanation」要像 **shap.force_plot**（第二张参考图）：横轴为 **模型输出概率**，中间 **f(x)**，**base value**，红/蓝段表示推拉。
- **实现**：**不用** `shap.plots.force` / `streamlit-shap`（依赖浏览器 JS，易触发 chunk 加载失败）。在 **`streamlit_shap_three_cohorts.py`** 用 **matplotlib** 画 **单条水平累积带**：从 **E[f(x)]** 堆到 **f(x)**，标签 **特征名 = 原始输入值**；**top 12 |SHAP|** + 余项合并 **Other features**；结果仍 **PNG → base64 → `st.markdown`**。缓存增加 **`base_proba`**（`explainer.expected_value` 正类；缺失则用 `proba - sum(SHAP)`）。

**2026-03-30 Streamlit `TypeError: Failed to fetch dynamically imported module …/static/js/index.*.js`（SHAP 区红字）**
- **现象**：浏览器控制台 / 页面红字：**Failed to fetch dynamically imported module** 指向 **`localhost:8501/static/js/index.….js`**；SHAP 区空白或整页异常。
- **本质**：**前端**懒加载的 **JS chunk** 未成功拉取（缓存损坏、扩展拦截、网络、或 **Streamlit 某版本 + 大量组件** 组合触发），**不是** SHAP 数值算错。
- **已做（`streamlit_shap_three_cohorts.py`）**：SHAP 与周边尽量少用会拉额外 chunk 的组件：**`st.dataframe` / `st.image` / `st.slider`（SHAP 内）/ `st.metric` / `st.expander`** → 改为 **`st.markdown` + HTML**（表用 `_html_simple_table`，图用 **PNG base64 `<img>`**）；**Feature values** 展开区由 **`st.expander`** 改为 **`_html_details_rich`（原生 `<details>`）**；无 SHAP / 绘图失败用 **内联样式 `<p>`** 替代 **`st.warning` / `st.error`**（该块内）。
- **用户侧**：`pip install -U streamlit`；对 **`http://localhost:8501`** 清 **站点数据** 或 **InPrivate**；**Ctrl+F5**；可试 **`http://127.0.0.1:8501`**。

**2026-03-30 Streamlit SHAP 不显示：cache 与版本号不一致**
- **现象**：Assessment / factors 可能正常，但 **SHAP explanation** 整块不出现；或反复提示点 Start。
- **原因**：`run_ver == computed_ver`（认为已跑过）但 **`{key}_result_cache` 丢失**（浏览器会话、代码升级改缓存结构、`st.session_state` 里 DataFrame 序列化问题等）→ 原逻辑 `cache is None` 直接 `return`，**既不重算也不画图**。
- **修正**：若 `not need_compute and cache is None`，将 **`computed_ver` 置 0** 并 **`st.rerun()`** 强制进入 `need_compute`；缓存中 **`factors_df` 改为 `factors_records` + `factors_columns`**（纯 list/dict）；**`row_values`** 仅保留可 JSON 的 float；文件头 **`matplotlib.use("Agg")` 在 `import pyplot` 之前**；SHAP 绘图包 **`try/except`** 显示 `SHAP plot failed: …`；**Kernel SHAP** 旁注等待时间。

**2026-03-30 Streamlit 临床仪表盘式改版（仿参考布局 + 解决预测/SHAP 不刷新）**
- **布局**：侧栏 **Clinical assessment parameters** — 全部特征 `st.sidebar.slider`（min/max 为子样本范围）+ **Load cohort medians**；主区 **Start assessment**（`type="secondary"` + 红框 CSS）→ 双列 **Assessment result**（分级+概率+风险标准 expander）与 **Key contributing factors** 表 → **SHAP explanation**（红/蓝条形，近 force-plot 配色）→ **Clinical recommendations** 编号列表；展开区 **Feature values** 显示当前输入。
- **运行逻辑**：用 **`run_ver` / `computed_ver`** 取代 `pred_armed`：首次进入队列 `run_ver=1` 与 `computed_ver=0` 自动触发一次计算；点击 Start 或 Load medians 时 `run_ver += 1` 强制重算；结果写入 **`{key}_result_cache`**（`proba`、`vec` 存 list、`factors_df`、`row_values`）。侧栏改滑块后若未点 Start，显示 **stale_inputs_warn**。
- **签名**：`_params_signature_from_sliders` 与稳定排序/四舍五入，避免无谓脏标记。
- **删除**：主区 `data_editor` 纵向表与旧 `pred_armed` 流程。

**2026-03-30 Streamlit Run prediction 按钮无反应 bug**
- **根本原因**：顶部 `run_pred_top_{key}` 按钮在 `data_editor` **之前**触发，设 `pred_armed=True`；之后 signature 检查发现 prev_sig ≠ sig，把 `pred_armed` 重置为 **False**，导致预测不执行。
- **修正**：删除顶部按钮，**仅保留 `data_editor` 之后的底部 `run_pred_{key}` 按钮**。该按钮在 signature 检查之后渲染，点击时能正确覆盖 `pred_armed=True`，无竞争条件。
- **规则**：凡是影响 `pred_armed` 的按钮，必须放在 signature 检查**之后**，否则检查会覆盖按钮的赋值。

**2026-03-30 Streamlit UI 优化（`streamlit_shap_three_cohorts.py`）**
- **主标题**：`.pd-main-title` 从 **5.6rem → 1.55rem**，不再占满半屏。
- **布局**：`max-width` 960px → **1120px**；风险/SHAP 列比 **1:1 → 1:2**，SHAP 图更宽。
- **风险输出**：改为 `.risk-pill` HTML + CSS（`risk-low/med/high` 三色：**绿/橙/红**，阈值 20%/40%），替代默认 `st.metric`。
- **Run prediction 按钮**：在特征表**上方**增加一个（`run_pred_top_{key}`），无需滚动到底部再触发；下方原按钮（`run_pred_{key}`）保留。
- **特征数显示**：表格上方用 inline HTML badge 显示特征总数。
- **SHAP 条形图**：bars 旁边附数值标注（`+0.xxx` / `-0.xxx`，7.5pt）；左边距根据特征名长度自适应（`left_margin = min(0.48, max(0.28, 0.04 + max_name_len * 0.012))`）。
- **侧栏**：顶部加 `.pd-sidebar-top` 品牌名；移除顶部多余 `---`；底部 caption 去掉 `streamlit run …` 行。
- **`_pd_flow_step`**：`desc` 为空时不再渲染空 `<p>` 标签。
- **py_compile 通过**：Exit code 0。

**2026-03-30 Streamlit 侧栏：连续变量被当成「分类」（超长 selectbox）**
- **现象**：年龄、BMI 等本应 **slider**，却变成 **selectbox**（唯一值极多）。
- **原因**：(1) **`if item and item[0] == c or (...)`** 应加括号，避免与 **`total_cognition` 别名** 混用时的歧义；(2) 仅靠 Table-1 元数据不足以覆盖「高基数数值列」；(3) **`lifestyle_binary`** 等未并入 binary 循环时，个别列可能被误标。
- **修正**：拆成 **`_bps_kind_from_metadata`** + **`_bps_ui_kind(col, X_all)`**；对 **`pd.to_numeric` 后 `nunique > 25`** 的列 **强制 `continuous`**（slider）；侧栏与 **Load medians** 一律传入 **`X_all`**。

**2026-03-30 Streamlit 界面改为仅英文（弃用中英切换）**
- **变更**：`streamlit_shap_three_cohorts.py` 移除 `STRINGS[LANG_ZH]` / 侧栏语言 radio / `_init_lang_from_url_once`；**单字典英文** `STRINGS` + `t(key)`；`page_title` 固定英文；`COHORT_META` 仅保留 `title_en`（删未用 `title_zh`）。
- **文档**：`docs/SHAP_Streamlit_三队列使用说明.md`「界面语言」与页面结构表改为**仅英文**说明；`.remember/memory/project.md` 中 Streamlit 侧栏表述同步。
- **历史**：此前「URL `?lang=` 每轮覆盖侧栏」类问题随语言切换一并移除。

**2026-03-30 Streamlit 选 English 界面仍中文**
- **原因**：曾在 **每次** `main()` 开头调用 **`_apply_lang_query_param()`**，若地址栏长期带 **`?lang=zh`**（或 Streamlit 保留 query），会**每轮覆盖** `st.session_state["ui_lang"]`，与侧栏 **`radio(key="ui_lang")`** 冲突，表现为选了 English 仍中文。
- **修正**：改为 **`_init_lang_from_url_once()`**（`_lang_url_initialized`）：**仅本会话首次**读取 URL 的 `lang`；之后完全由侧栏 radio 控制。`set_page_config` 仍靠前；`st.query_params` 不再在每次 rerun 强行写 `ui_lang`。

**2026-03-30 Streamlit 英文界面不明显**
- **原因**：语言原为侧栏 **selectbox**，易被忽略；部分用户未注意到可切 **English**。
- **已做**：侧栏顶部改为横向 **`st.radio`（中文 | English）**；URL 语言见上条「仅首次」；**`cohort_kicker`** 随语言；说明见 **`docs/SHAP_Streamlit_三队列使用说明.md`**。

**2026-03-30 Streamlit 特征表无法改数值：多 Tab × 多 `data_editor`**
- **现象**：用户反馈 localhost 页面上数值无法修改。
- **原因**：在 **`st.tabs`** 内为每个 BPS 分组各放一个 **`st.data_editor`** 时，部分 Streamlit 版本下非当前标签或嵌套布局会导致表格**不可编辑**或状态异常。
- **修正**：改为**单个** `st.data_editor` + 上方 **`selectbox`** 按 BPS 筛选（含「全部特征」）；`key` 使用 `f"{vedk}_editor_{scope}"` 避免切换筛选时组件状态串台；界面增加「单击数值格 → 表格外/Enter 刷新」说明；`docs/SHAP_Streamlit_三队列使用说明.md` 同步。

**2026-03-30 Streamlit 与投稿稿表号/结局口径对齐（`streamlit_shap_three_cohorts.py` / `docs/SHAP_Streamlit_三队列使用说明.md`）**
- **主文**：CPM 冠军为 **Table 3**（非 Table 2）；结局为 **next-wave incident DCC**，不宜写死「2 年」单一表述。
- **已做**：界面与 `warn_mismatch` 等 i18n 改为 **Table 3** + 说明 **`table2_*_main_performance.csv`** 为流水线文件名；风险/SHAP 文案改为 **next-wave DCC**；首页 **hero** 改为双语 **`hero_main_title`**（原仅英文长句）；`_load_cohort_subsample` 注释与说明文档修正 **`had_comorbidity_before`**（特征排除、须走 `load_df_for_analysis` 非裸 CSV）。

**2026-03-29 预测网页与投稿稿对齐**
- **事实**：三队列 **`streamlit_shap_three_cohorts.py`** 一直在仓库；**`PAPER_Manuscript_Submission_Ready.md`** 与根 **`README.md`** 曾未写 Streamlit，易被误认为「缺网页」。
- **已做**：标题页 **Data availability**、**§2.14**、**§2.15**（`streamlit` 依赖一句）、**Main ↔ supplementary index** 表一行、**补充目录** 增 **Replication code (interactive)**；**`README.md`** 增 **交互式预测演示** 小节；**`requirements.txt`** 增 **`streamlit>=1.28.0`**。旧 **`streamlit_app.py` / `charls_streamlit_app.py`** 在 README 中标为可能过期，**以三队列为准**。

**2026-03-30 补表脚本与主流程挂钩（`scripts/fill_paper_tables_extras.py` / `run_all_charls_analyses.py` / 稿件 S16·S8·表4）**
- **`fill_paper_tables_extras.py`**：`--psw-only` 对 **m1–m5** 仅跑 **`run_causal_methods_comparison`**（取 **PSW** 行）→ **Rubin 合并** → **`results/tables/rubin_pooled_psw_exercise.csv`**；`--age-only` 对 **A/B/C** × **age&lt;70 / ≥70** 调用 **`estimate_causal_impact_xlearner`**（`run_auxiliary_steps=False`，**200** bootstrap）→ **`subgroup_age70_xlearner_exercise.csv`**。Windows 下临时目录前缀**禁止**含 **`<`**（已改用 **`lt70`/`ge70`**）。
- **`run_all_charls_analyses._run_multi_imputation_rubin_analysis`**：在写出 **`rubin_pooled_auc/ate.csv`** 后 **`importlib` 加载** 上脚本并 **`run_psw_rubin_mi()`**，保证全流程 Rubin 后 **S16 用 PSW 合并** 有文件。
- **稿件**：**S16** 填入 **PSW** 合并列；**§3.8**、**§4.7** 与 **Table 4 / S8** 增加 **&lt;70/≥70** 行并指回复现 CSV。

**2026-03-29 审稿人式修订：表2/ E-value/ S16/ 亚组/ 局限/ 图注/ 参考文献/ S10（`PAPER_Manuscript_Submission_Ready.md`）**
- **表2**：脚注明确 **Incident cases**、**粗比例**（例 B 426/3123=13.6%）与 **人年发病率**（84.4/1000 PY）关系；**§3.2** 补一句分母区别。
- **E-value**：**§2.6**、**§3.5**、**表5** 列名与脚注标明 **point** vs **CI-based conservative**（**最接近 null 的 CI 界**）。
- **S16**：用锁定 **`rubin_pooled_auc.csv` / `rubin_pooled_ate.csv`** 数值填满 **Rubin pooled AUC** 与 **XLearner exercise ARD**（**M=5**、**SE**、**95% CI**）；**PSW** 行保留 **—** 并正文/**§3.8**/**§4.7** 说明 **MI 敏感性导出未做 PSW 的 Rubin 合并**（主估计仍为单次完成集）。正文 **不写** `results/...` 路径。
- **亚组年龄**：**Table 4 / S8** 诚实说明 **预设 <70/≥70** 与 **三分类导出** 不一致；**可 proof 阶段补行**，避免谎称 S8 已有二元年龄行。
- **§4.7**：在既有二分类运动局限上补 **效应量可能低估/错定** 与 **客观测量** 方向。
- **Figure 3**：主文明确 **仅 Cohort B SHAP**；**A/C** → 新增 **Supplementary Figure S6**（原 **S3** 仍为 Love plot，避免冲突）。
- **Table 7** 脚注：**ARD** 标明 **percentage points**，非相对变化。
- **目录 S10**：改为 **intentionally skipped**（内容在主文 Table 1），**S11–S16** 编号与引用对齐。
- **参考文献**：统一 **doi:10.** 形式（**2, 11–12, 20–27** 等）；**22** 保留全文作者列表；**24** 不编造卷期页。

**2026-03-29 投稿稿去中文（`PAPER_Manuscript_Submission_Ready.md`）**
- **用户要求**：全文不出现中文（CJK）。
- **已做**：将此前为双语读者加入的中文标签（精准预防窗口、直接干预窗口、可行动窗口、IPW/PSW/ARD/ATE 旁注等）全部改为**纯英文**（如 *precision prevention window*、*direct intervention window*、*canonical causal estimand: average treatment effect*）；**Supplementary glossary** 与 **index / TOC** 同步。**验证**：对稿件 `[\u4e00-\u9fff]` 无匹配。

**2026-03-29 补充材料编号：撤销「去 S」统一为数字（`PAPER_Manuscript_Submission_Ready.md`）**
- **用户要求**：撤销此前将 **Supplementary Table S1**→**Supplementary Table 1**、**Supplementary Text S1**→**Supplementary Material 1** 的全稿替换。
- **已恢复**：正文、表题、清单与 Mermaid 中 **Supplementary Table S1–S16**（含 **S6b**）、**Supplementary Figures S1–S5**、**Supplementary Text S1–S3**；章节标题 **## Supplementary Text S3** 等。标题页 **Supplementary Materials**（复数、指附录总称）保留不变。

**2026-03-30 标题页/摘要/引言/§2.1 冗余删除与 PSW 数值去重（`PAPER_Manuscript_Submission_Ready.md`）**
- **删除**：Word counts 下编辑用斜体备注；Highlights 内嵌 *n*/≈24/84/107/表号堆叠；引言 **H1–H3** 编号与括号解释 → 改为 **Pre-specified expectations** 融入正文；**Registration** 段删除 **OSF/ClinicalTrials/PROSPERO** 未注册辩解；**§2.1** 删 **`pipeline_trace/README`**、事后功效 **nQuery/手算**、**Fisher-exact** 免责、**A 组 PSW −0.7%** 兼容句、**MCAR** 主动否认句；**插补**段删 **max_iter/tol/50树/20特征** 细目，改为 **Bayesian ridge / RF / 迭代链** 概述，细节指 **S1+复现**。
- **PSW 3.1%**：**Abstract Results** 与 **§3.4 等 Results** 保留精准值；**Highlights / What adds / Contributions / §3.0 / §4.1 / §4.8** 等改为 **显著/绝对风险尺度** 指代 **Results/Table 6**；**Abstract Conclusions** 压缩重复数字。
- **全局**：**§2.8** **hypothesis H2** → **pre-specified primary readout**；补充表 **Triangulation flag** 括号改为逗号衔接。

**2026-03-30 标题/摘要/引言/方法/讨论「去冗余」与顾问意见对齐（`PAPER_Manuscript_Submission_Ready.md`）**
- **勿编造**：顾问示例中的 **45–59 岁**、**吸烟边际收益**、**中年/老年双色 CATE 图**与本研究 **CHARLS ≥60、A/B/C 表型、四生活方式暴露**不符；**未**写入虚假年龄分层或新图。
- **已做**：标题去掉括号 **(Stratified Causal ML)**，改为 **Leveraging Causal Machine Learning… Life-Course Cohort Analysis of CHARLS**；摘要 **Results** 强调 **表型间干预优先级转变**；引言压缩 DCC 共识、强化 **when/for whom** 与 **Cox/回归+少量交互** 对 **HTE** 的不足；**§2.2** 补充 **interval censoring + midpoint** 表述；**§2.10** 增加 **CausalForestDML** 的 **doubly robust**（Neyman 正交、[6]）与 **被动语态** 稳健性句；**§2.2** 增加 **失访/选择性随访** 透明声明（**未** IPW 续访）；讨论增 **FINGER [27]**、**Lancet Commission [26]** 对比段；**Figure 4** 图注 **multi-panel 合并**编辑提示；语言 **important→principal / pivotal**，**Importantly→In this framework**；参考文献新增 **[26][27]**。

**2026-03-30 Cohort C 防误读「运动有害」：摘要/结论/通俗总结/§3–4 措辞（`PAPER_Manuscript_Submission_Ready.md`）**
- **原则**：C 层 **正 ARD / 选择+反向因果** 表述须 **并列** **非因果、非临床减活动建议**；**勿**让读者把 **统计关联** 读成 **physical activity harmful**。
- **已改**：**Highlights**、**Abstract Results/Conclusions**、**What this study adds**、**Plain-language**、**§3.0 / §3.3.1 / §3.4 / §3.6**、**§4.3 / §4.7 / §4.8**；**§2.8 PSW** 早前已区分 **non-stabilized IPW**（另条）。

**2026-03-30 §2.1 MICE 明细 + §2.6 XLearner/PSW 与 PSW 非稳定化修正（`PAPER_Manuscript_Submission_Ready.md` / `archive/charls_imputation_npj_style.py` / `causal/charls_causal_methods_comparison.py`）**
- **§2.1**：与 **`charls_imputation_npj_style.py`** 对齐写明 **排除变量**（**VARS_NO_IMPUTE**、结局与分层量表）、**连续 / 序次 / 二分类 / 名义** 处理、**辅助列**、**个体历史 ffill/bfill**、**连续块 NRMSE 选法**（含 **MissForest 100 树**）、**`IterativeImputer` `max_iter=5`、`tol=10⁻³`**（sklearn 默认）。
- **§2.6**：**XLearner** **`RandomForestRegressor` n_estimators=200, max_depth=4, min_samples_leaf=15, bootstrap=True**；**B=200** 聚类 bootstrap；**PSW** **LogisticRegression C=1e-2, max_iter=5000, lbfgs**，**ps clip [0.01,0.99]**，权重 **`1/ps` 与 `1/(1-ps)`** → **非稳定化 IPW**。
- **文稿修正**：原写 **stabilised IPW** 与 **`_ate_psw`** 不符；已改为 **non-stabilized Horvitz–Thompson–type**。

**2026-03-30 摘要 Methods 与 §2.1 主分析 / M=5 Rubin 对齐（`PAPER_Manuscript_Submission_Ready.md`）**
- **需求**：摘要 **Methods** 与 **§2.1** 一致写明 **主分析 = 预设 bulk 完成表 `step1_imputed_full`**；**敏感性 = M=5 + Rubin（S16）**；避免 **`config.RANDOM_SEED` + *m*** 在 Markdown 中反引号断裂 → 改为 **`study RANDOM_SEED + m`（`config.py`）**。
- **其他**：**迭代收敛图** 指 **`pipeline_trace` 中 imputation convergence figure + Supplementary Text S1**；**PSW 权重处理** 写 **prespecified estimation scripts**，勿写 **§2.8 code**。

**2026-03-30 `table7` XLearner CI 与主文不同源（`causal/charls_causal_methods_comparison.py`）**
- **原因**：`run_all_cohorts_comparison` 曾用 `CAUSAL_ANALYSIS` 里 `causal_impact_*` 列的 **2.5/97.5% 分位数**当「95% CI」，实为 **ITE 在样本中的离散度**，不是 **ATE 的抽样区间**。
- **修正**：新增 `load_ml_ate_ci_from_summary_txt`，**优先**从同目录 `ATE_CI_summary_{T}.txt` 读 ATE+CI（与 `charls_recalculate_causal_impact` / **table4** 同源）；仅当缺少该 txt 时才回退分位数并 **WARNING**。
- **验证**：`Cohort_B exercise XLearner` 由 **(−0.08, 0.0582)** 对齐为 **(−0.0416, 0.0034)**；已将 `LIU_JUE_STRATEGIC_SUMMARY/causal_methods_comparison_summary.csv` 复制到 **`results/tables/table7_psm_psw_dml.csv`**。

**2026-03-29 ARR/ARD 符号、Table 7、§3.4–§4.2、阴性对照逻辑、Table 6 脚注（`PAPER_Manuscript_Submission_Ready.md`）**
- **ARR vs ARD**：**ARR** 叙述**降低幅度**用**正数**（如 **3.1%**）；**ARD** 为**暴露减对照**的**有符号差**（保护性为**负**，如 **−3.1%**）。**禁止**写 **ARR −3.1%**；应写 **ARD −3.1%（ARR 3.1%）**。
- **已扫**：**§3.4、§3.5、§3.8、§4.1、§4.2、§4.8、Table 7** 与 **Table 6** 脚注；**Cohort A PSW** 非显著处标 **ARD −0.7%**。
- **§4.1**：**Cross-estimator spread** 不写 **identifies the estimand**；改为 **rebalancing scheme / separation from null**；**estimator** vs **estimand** 分句。
- **阴性对照**：**跌倒**定义指 **Supplementary Table S3**；补全 **generic healthy-responder** 与 **DCC+falls 双终点** 的**反驳链条**（**null falls** 削弱**唯一**健康应答者解释，**不**排除 **DCC 特异**混杂）。
- **Table 6 脚注**：**comorbidity** → **DCC**；**ATE** 与 **ARD** 关系指 **§2.7**（风险尺度前提）。
- **勿编造**：用户曾建议 **Supplementary Text S4** 写 **B=1000 bootstrap** 敏感性；若**无**该补充文件，应写 **replication materials** 或**略去**，**勿**虚构 **Text S4**。
- **参考文献 [11]**：**VanderWeele & Ding E-value** 已在 **## References** 第 **11** 条；若用户称「缺 [11]」→ **核对 References 节**。

**2026-03-29 DCC 命名、因果假设可读性、CATE/ITE、时序、NNT、表题、DML 与 *P*（`PAPER_Manuscript_Submission_Ready.md`）**
- **DCC**：摘要 **Background** 首句 **操作化定义** + 全文高频 **Depression–cognition comorbidity (DCC)** / **incident DCC**；**Keywords** 增 **DCC**；**短标题** 写明 DCC。
- **§2.6**：**强可忽略性 / 重叠** 扩展为可读表述 + **Table 1 / Supplementary Figure S2**；另设 **Plain-language check** 段。
- **§2.9 / §3.7**：区分 **ITE** 与文内报告的 **队列平均 ARD** 与 **亚组 CATE**；**precision prevention** 与亚组关系一句。
- **§4.6.5**（Limitations 前）：**时序 / 滞后 / 排除基线共病** 论证减轻反向因果；**不**断言 4 年、7 年（与 CHARLS 波次一致写 **next wave**）。
- **§4.2**：**NNT ≈ 32**（ARR 3.1%）+ 观察性限定。
- **Table 3/4/6**、**Table 2**、**Table 1 一行**、**Figure 2/4** 标题或脚注强化 **DCC** 自明性。
- **DML**：首现 **double machine learning (DML)**，**Highlights / §3.0 / S7 脚注** 等与 **CausalForestDML** 对齐；**XLearner vs DML** 段标题调整。
- ***P***：**0.064** 统一为 ***P* = 0.064**（三位小数）；*P* < 0.001 保留。
- **References** 上增加 **Vancouver 体例说明**（按目标刊微调）。

**2026-03-29 投稿稿去「工程痕迹」（`PAPER_Manuscript_Submission_Ready.md`）**
- **问题**：正文/补充出现 **仓库路径**（`results/figures/`、`results/tables/*.csv`）、**代码变量名**（`is_fall_next`、`Cohort_A`/`drinkev`）、**管道/TBD/audit** 口吻，易被审稿人视为**未完成稿**。
- **修正**：改为**学术表述**（complete cases、prespecified analysis program、replication materials）；**删** `.csv` 与 **锁定运行日期**；**Supplementary Table S13/S14/S15** 表头与 **S13** 行内暴露名改为 **可读标签**；**S16** 合并为一条 **Note**，去掉 **`results/tables/`** 与 **Replication note/TBD** 话术，**Cohort B XLearner** 行填入文内 **Rubin −2.13%**；索引表去掉 **`cohort=`** 代码筛选说明；**Figure** 规格段去 **`./results/`** 与 **export scripts** 依赖表述。

**2026-03-29 投稿稿去中文（`PAPER_Manuscript_Submission_Ready.md`）**
- 已删除全文 **CJK 汉字**块：**中文关键词**、中文 **核心发现**、摘要后 **本研究的新增信息**、**通俗总结（中文）**；**Keywords** 去 *中文关键词见标题页* 附注。
- 恢复英文 **Highlights** 条列与 **Plain-language summary** 标题（无 “English” 后缀）。

**2026-03-29 摘要/Highlights/要点段「去机器感」英文化（`PAPER_Manuscript_Submission_Ready.md`）**
- **范围**：**Highlights**、结构化 **Abstract**、**What this study adds**、**§3.0**、**§3.3.1**；**不改动**任何**数值、CI、P、E-value、表号结论**。
- **原则**：完整主谓宾句；**PSW ARR 与 ARD** 用自然语言解释等价关系；**E-value** 写为 **point vs CI-based conservative**；少用 **加粗碎片+嵌套括号** 堆叠。

**2026-03-29 正文标点偏好（`PAPER_Manuscript_Submission_Ready.md`，# MAIN TEXT 至 ## References）**
- **需求**：除非必要，少用 **em dash（—）**、括号与冒号。
- **做法**：主文内 **U+2014 全部去除**，改为 **逗号、分号、句号** 或 **First/Second** 枚举；**Figure/Table** 引用由 `(Figure 1)` 改为 **`as in Figure 1`** / **`per Table 3`** 等；部分 **加粗段首 `**Label:**`** 改为 **`**Label.**`**。
- **勿**对全文做**无脑**全局 `—`→`; `** 替换，会破坏 **which group—if any—**、**MICE…—imputation…—to** 等结构；需**逐段**核对 **`. they` / `. so` / `; while`** 类断裂。术语列表 **§2.9** 仍保留 **`ARD:`** 式定义冒号以利扫读。

**2026-03-29 审稿「必须修改」9 条（`PAPER_Manuscript_Submission_Ready.md`）**
- **§3.8**：**禁止**正文出现内部仓库路径（如 **`results/tables/rubin_pooled_ate.csv`**）→ 改为 **locked replication environment** + **主文数值 + 同行评议存档** 表述。
- **术语**：§2.11 **XLearner** 主推断句用 **contrast (ARD, %)**；**Figure 4(A)** 用 **risk differences (ARD, %)**，避免 **ATE** 与 **ARD** 叠用。
- **Funding**：**不得**保留 **`[To be inserted]`** → 无资助时用 **The authors received no specific funding for this work.**（有资助则替换为真实声明）。
- **文献 [22]**：补全 **Marafino 等** **NPJ Digit Med 2025;8(1):571** + DOI（Crossref 核对）；**[1] WHO**：**2025 fact sheet** + **URL + accessed date**。
- **Rubin 数值**：摘要 **Rubin-pooled** 与 **§3.8** 统一为 **−2.13%**（**95% CI −4.43%～+0.17%**）；**single-draw / Table 6** 仍为 **−2.1%**。
- **§2.1 事后功效**：拆分 **ARD ≤1 pp 需极大样本** 与 **~5 pp 为说明性功效阈值**，避免与 **≤1 pp** 混读。
- **§4.7**：补 **二元自报运动** 的 **剂量/时长/类型缺失** 与 **错分（非差向→零偏；差向不可排除）** 一句。

**2026-03-29 严谨性补强：E-value / SUTVA / MI Rubin / 功效 / ARD 符号 / 阴性对照数值（`PAPER_Manuscript_Submission_Ready.md`）**
- **§2.1**：**person-waves ↔ participants**（**14,386** / **~7,027**，**≈2.05** waves）；**功效**扩展为 **B 为主因、A 低发病率与功效不足解释、C 非功效瓶颈**；**非 Fisher** **前瞻**口径写明。
- **§2.4/2.6/2.7**：**grouped CV** 按 **ID** **防泄漏**；**SUTVA** **个体自报运动** **工作假设**；**E-value** **表意**（**XLearner** 路径 + **PSW** 可 **[11]** **自行换算**）；**RF** **`n_estimators=200,max_depth=4,min_samples_leaf=15`**。
- **§2.9 / §2.7 causal chain / Table 6 脚注**：**ARD 符号**、**ATE≡ARD**（**Y** 二分类）；**§3.5–3.8**：**B** **comorbidity** **E-value ~1.59/~1.19**；**跌倒阴性对照** **A/B/C** **XLearner** + **B PSM/PSW** **点值与 CI**（对齐 **`negative_control_results.csv`** / **`negative_control_psm_psw.csv`**）；**§3.8** **Rubin pooled XLearner B** **−2.13%**（**−4.43%~+0.17%**）；**投稿正文不写**内部 **`results/...`** 路径。
- **摘要/亮点**：**CPM** **全称**；**PSW ARR −3.1%** **与 ARD** **并列**；**E-value**、**Rubin**、**跌倒 B** **数值**；亮点 **英文标点**。**摘要词数** 终稿需 **Word** 再压（Markdown 粗计可能 **>300**）。

**2026-03-29 降「AI 味」与学术表述对齐（`PAPER_Manuscript_Submission_Ready.md`）**
- **引言**：**Furthermore/While/Conversely** 段改为短句 + **However**；**Standard regression** 段改为 **masking heterogeneity** + **XLearner** **separate outcome distributions**；**PSM/PSW** 与 **cross-check**。**Contributions (iv)**：**falsifies** → **does not support universal healthy-responder bias as sole explanation**。
- **摘要**：压至 **~300 词**（Markdown 粗计）；**Methods** 删 **Platt/嵌套插补细目**（指回 **Methods**）；补 **单套 MICE 主分析 + Rubin 敏感性（supplements）**；**Results B** 加 **临床解读** 与 **estimand** 句；**Conclusions** 用 **triangulated**。**关键词**与标题页对齐 **Older adults; China**。
- **方法 §2.1**：补 **单套主分析 / Rubin 敏感性** 一句 + **Cohort B 事后效力**（~80% 检 **≥2.5% ARR**，与 **−3.1%** 一致，**illustration**）。**§2.7**：**nuisance** → **auxiliary**；固定超参 **CHARLS 常见实践 + 透明度**。**§2.8/2.10/2.11/4.1/4.4/4.7/4.8**：**stress test** → **cross-check / cross-estimator**；**omnibus** → **universal**；**§4.4** 标题与阴性对照语气审慎。**§3.7–3.8**、表脚注、补充 **S6b** 说明同步术语。
- **未自动完成**：**温哥华参考文献** 全书逐条终检（**Funding** 已改为 **no specific funding** 占位句；若有资助需作者替换）。

**2026-03-29 摘要去图表编号（`PAPER_Manuscript_Submission_Ready.md`）**
- **需求**：从 **TITLE PAGE**（含 **Highlights**、图表数量说明）经 **# ABSTRACT** 至 **Plain-language summary** 止，**零**对主文/补充 **Figure *n* / Table *n* / Table S*** / Figure S*** / Supplementary Text S*** / §*.* 的**交叉引用**（避免门户粘贴与短摘要堆编号）。**Introduction 第一段**（开篇段）本身无 **Table/Figure**；紧接的**研究设计概述句**去掉 **Figure 2 / Table 7**，改为 **later in the main text**、**tabular results presented below**。
- **已改**：**Highlights** 去 **Table 2/7/S6/S6b、Figure 5**；**标题页**「图表数量」改为**叙述性**（不列 **Tables 1–7**、**Table 3**、**Figures 1–5**、**S1–S16**）；**Word counts** 中 **§§1–4** 改为 **sections 1–4**；结构化摘要 **Methods** 中 **Supplementary Text S1** → **supplementary appendix on missing data**；**§2.8** → **full definitions in Methods**。此前摘要 **Results/Conclusions** 已去 **Table/Figure** 编号。

**2026-03-29 Table S6b 数值化 + 流水线同步 + 封面/清单 + 文献 23–25**
- **S6b**：用 `run_causal_methods_comparison(..., target_col=is_fall_next)` 对 **A/B/C** 在 **complete-case** `is_fall_next` 上跑出 **PSM/PSW**，写入 **`results/tables/negative_control_psm_psw.csv`**；稿内 **Table S6b** 与 **§3.5 / §4.4 / 摘要 Results / Highlights** 对齐为 **完成时** 三角验证表述。
- **`run_negative_control_outcome_summary`** 末尾增加 **同逻辑** 写出 **`negative_control_psm_psw.csv`**，避免下次只更新 **XLearner CSV** 而漏 **S6b**。
- **新增文件**：`COVER_LETTER_snippets.md`（阴性对照 **S6** 主报告 + **S6b** 一句）；`PAPER_Checklists_STROBE_TRIPOD.md`（**STROBE** + **TRIPOD** 勾选指引）。**补充材料** 增加 **Mermaid** 源码块（**Figure 2** 伴侣图）。
- **参考文献 [23]–[25]**：**Baduanjin** RCT（Ye *et al.* **Front Public Health** 2024）、**太极 vs 有氧**（Huang *et al.* **JSEP** 2024）、**中国老年抑郁 ML**（Song YLQ *et al.* **Front Public Health** 2025）。**勿**凭记忆编造 **BibTeX 作者**：**JSEP** 文曾误写作者，已按检索改为 **Huang C, Yi L, Luo B, Wang J**；**Baduanjin** 文改为 **Ye Y, Wan M, …**。
- **标题页字数**：脚本 `scripts/_count_words_md.py` — 结构化摘要 **~540** 词（多刊 **>300** 上限，需再压）；主文叙事（去表格行）**~7560** 词量级 — **Word** 终检为准。

**2026-03-29 投稿瘦身与规范：删 Mermaid / 摘要去 Rubin 细目 / 弱化 “未完成” 叙事**
- **Title / Short title / Highlights / Abstract / Plain-language / Intro / §2.1–2.4 / §2.7 / §2.10 / §4.7 / §4.8**：按用户清单删 **pending**、**内部脚本路径**、**Mermaid 代码块**（改一句 replication 存档说明）；摘要 **Methods** 去掉 **Rubin/S16** 长句（改结论末句轻点 **S16**）；**What this study adds** 聚焦设计与科学贡献；**§4.7** 纵向段改为 **ID 聚类 bootstrap + 分组 CV** 一句。
- **S6b 与正文一致性**：主文 **§3.5 / §4.4 / 结论** 在 **S6b 仍为占位行** 时不断言 **PSM/PSW 数值已证 null**；**S6b** 表头保留 **投稿前替换斜体格** 的作者提示。**Highlights** 仍指向 **S6b** 作 **parallel checks**。

**2026-03-29 审稿第二轮扫尾：S6b 诚实表述 + 索引 + S15/S13 可读性**
- **问题**：正文 **§4.4 / Highlights / §2.10** 曾暗示 **PSM/PSW 跌倒阴性对照**已填满 **S6b**，与流水线 **`negative_control_results.csv`**（仅 **XLearner**）不一致 → **选择性夸大**、不可复现风险。
- **修正**：**Highlights**、**§4.4**、**§4.8**、**§2.10** 统一为 **Table S6 = XLearner** 已锁定；**S6b** **预注册模板**、数值 **待合并导出**。**S6b** 表头增加 **Status** 段说明 **`run_negative_control_outcome_summary`** 行为。
- **其他**：**§4.7** 已含 **纵向聚类 / 未用混合效应** 与 **文献时效** 计划句；**S5** **行映射** 改为 **九行 exercise** 与 **S13** 对齐；**主↔补充索引** 行改为 **`exercise` 过滤 S13**；**S15** 增加 **Reader guide**（前 15 行 = 主情景 + 列释义）；**S13/S14** 增加 **`ate_lb/ate_ub`** 等 **glossary**。

**2026-03-29 审稿回应稿：术语 / 索引 / TBD 标注 / SHAP / Mermaid（不改嵌图工作流）**
- **§2.9** 新增 **Effect size and terminology**（ARD / ARR / ATE / CATE）；原 **§2.9–2.14** 顺延为 **§2.10–2.15**。**§2.8** 内原有效应量段并入 **§2.9** 以免重复。**Table 6** 脚注改指 **§2.9**。
- **§2.1** 预测–因果关系段后增加 **Mermaid** 流程图（无像素路径；Pandoc→Word 可能显示为代码块，可改期刊排版时替换为图）。
- **§3.3.2** 增加 **SHAP** 与老年医学衔接及与 **§2.7** 协变量对齐；修正 **Table 7** 一句加粗错误。
- **Table 7** 脚注补充 **PSW CI/P** → **S5**，全估计器 **CI/P** → **S13/S14**。**补充材料** 在目录前增加 **Main ↔ Supplementary 索引表**；**S6b**、**S16** 增加 **TBD/待填** 的 **replication** 说明（路径写 `results/tables/`、`save_reproducibility_snapshot`，避免写死仓库根相对路径 `./replication/...` 除非实际存在）。
- **FIGURE LEGENDS** 末增加 **Figure file specifications**（`./results/figures/`、矢量/300 dpi、终稿转换说明）。

**2026-03-29 投稿稿为何 Markdown 中未嵌入图**
- **原因**：主要是**工作流与期刊投稿习惯**（正文与高清图常分开上传），以及**避免图路径与仓库不同步**；非 Pandoc/Word 技术限制。稿内亦曾**刻意**以表代图（如 ROC）。若需在 Word 中内嵌图，在 `.md` 对应图注处添加 `![](相对路径)` 并保证图文件随稿一并复制即可。

**2026-03-29 主文表号重组 + 全文交叉引用扫尾**
- **结构**：原 **Supplementary Table S10**（全变量基线）→ **主文 Table 1**；原 **Table 1b**（发病密度）→ **主文 Table 2**；原主文 **Tables 2–6** 顺延为 **Tables 3–7**（CPM、亚组、假设、XLearner、策略）。
- **正文**：**Table 1b**→**Table 2**，**Tables 1–1b**→**Tables 1–2**；**CPM/ROC/校准** 指向 **Table 3**；**亚组** **Table 4**、**假设诊断** **Table 5**、**XLearner** **Table 6**、**双重可行动窗口/策略** **Table 7**。**勿**将 **Table 6**（XLearner）与 **Table 7**（策略）全局互换。
- **易错**：用正则把 `**Table 6**` 批量改为 **7** 时，会误伤 **XLearner** 引用并**吃掉**闭合 `**`（出现 `**Table 7).`）；需逐段核对并恢复 **Table 6** / `**Table 7**).` 等。**S16** 模板行：Test AUC→**(Table 3)**；XLearner B/C→**(Table 6)**；PSW→**(Table 7 / S5)**。
- **补充目录**：**S10** 改为占位说明（内容已进主文 **Table 1**）；标题页与 checklist 写 **Tables 1–7**。

**2026-03-29 审稿意见大修：`PAPER_Manuscript_Submission_Ready.md`**
- **摘要**：补中国老龄化/基层资源本土化负担；方法明确 **运动定义**（中等强度、每周≥1次）；结论收紧 **观察性、单套插补主分析** 对临床应用的含义；**ARD/ARR** 与 **PSW/XLearner** 表述对齐；**Plain-language summary** 去专业化（避免 IPW estimand、DCA 术语堆砌）。
- **引言**：中国分层干预证据缺口；**XLearner vs PSM/PSW** 与老年生活方式不平衡的适配性；**H1–H3 工作假说**。
- **方法**：**MICE *M*、收敛、变量集** + **主分析单套完成集**（算力/复杂度权衡）；**暴露缺失→0** 的保守编码 + **S2/S15** 审计；**预测 vs 因果** 同风险集、不同缺失处理路径；**CES-D/认知截断** 依据 CHARLS 常用文献非本研究 ROC；**CATE 亚组** 探索性说明；**§2.14 软件**（Python、sklearn、econml 等）；**§2.8 ARD/ARR**；阴性对照 **PSM/PSW**（**Table S6b** 模板，终投前用复制材料填满）。
- **结果**：**Table 2** 增 **校准斜率**（A≈1.0，B/C 与 §3.10 一致）；**西部验证** 病例混合解释 + **S4** 脚注；**Cohort C** 选择偏倚 **描述性证据**（复制交叉表，避免误指 S10 已含分层）；**PSM/PSW fall** 文字 + **S6b**；**Table 5/6** 与正文 **ARD/ARR** 统一。
- **讨论**：国际对照与中国落地路径；**未测量混杂** 列表；**Table S16 Rubin** 与主结论关系；**结论收敛** + **RCT 设计要点**；**AUC 0.72 局限**；**§4.9 未来方向**（方法/临床/公卫）。**图注**：Fig3 缩写表；**Supp Fig S5** CI + 平滑方法说明。**参考文献** 增 **[20]–[22]**（BMC/NPJ 方法与应用）；**[20] 第一作者**若与正式出版不一致，终投前以 PubMed 为准。

**2026-03-29 MD → Word（投稿稿）**
- **`PAPER_Manuscript_Submission_Ready.md` → `PAPER_Manuscript_Submission_Ready.docx`**：本机 **Pandoc 3.9**，`markdown+smart`、`--standalone --toc --toc-depth=6`、`-M lang=en`；在文稿目录下执行（PowerShell 用 `Set-Location` + `;`，勿用 `&&`）。当前稿内**无** `![](...)` 嵌入图、**无** `$...$` 公式，图仅为文字图注；表格与多级标题由 Pandoc 映射为 Word 样式。若 Word 中目录未自动刷新：**引用 → 更新目录**（或右键目录 → 更新域）。

**2026-03-29 投稿稿去内审化 + 单次插补辩护与 S16**
- **`PAPER_Manuscript_Submission_Ready.md`**：已删除 **Data lock-in** 整节；全文去除 **`.csv` 路径、`step1_imputed_full`、仓库路径、脚本再生命令** 等内部元数据；表格脚注与补充表导语改为 **科学表述 + 表号引用**；文末 **Data source** 改为 **Replication** 一句话。
- **单次插补**：§2.1 / §2.7 / 摘要 Methods 明确 **主分析单套 MICE 完成集**、**主文 CI 不含 between-imputation 方差**；**§4.7** 增加 **Rubin 规则、反保守、Type I** 的防御性讨论；新增 **Supplementary Table S16**（**Rubin's rules** 合并 MI 敏感性模板，**TBD** 需作者在终投前用流水线 **Rubin pooled** 输出填满）。
- **对内**：数值仍以仓库 `table7` / `table4` 等为权威；**README 或内部附件**可保留路径对照表，**勿**再写入投稿 PDF/主 `.md` 正文。

**2026-03-29 「双重可行动窗口」终稿对齐**
- **叙事**：**Cohort A** 最优 **AUC 0.72** → **精准预防**（筛高危、靶向一级预防），**≠**「健康人群最值得全民干预」；**PSW** 运动 **A** 为 **−0.7%**（95% CI **−1.6% ~ +0.2%**，NS，`table7` 锁定 **−0.0069 / −0.0156 / 0.0017**）。**Cohort B** **AUC 0.64** + **PSW −3.1%**（Sig）→ **直接干预 / 务实 RCT 锚点**。**Cohort C** **PSW +2.2%**（CI 含零）；**XLearner +3.5%** 脚注标明 **选择偏倚、非处方依据**。
- **稿件**：新增 **Table 6**（预测+PSW+策略）；讨论插入 **§4.2 Dual actionable windows**，原 **§4.2–4.7** 顺延为 **§4.3–4.8**；**Figure 5** 临床效用引用 **§4.5**；**Limitations** 补 **Cohort A 精准预防成本效果与未来 RCT**；标题页 **7 张主表（1, 1b, 2–6）**。

**2026-03-29 叙事升维：Triangulation / 阴性对照 / DCA**
- **Triangulation**：讨论 **§4.1** 从「透明度」改为 **stress test / 不同 estimand**，**PSW −3.1%** 作为 **IPW 伪总体下** 的 **ARR** 证据，**XLearner 边缘** 与 **失衡+异质性** 一致；**锚定 Cohort B 务实 RCT**。方法 **§2.8**、结果 **§3.4/3.8**、摘要/Highlights/引言 Contributions 同步。
- **阴性对照**：**§3.5**、**§4.3**、**Table S6** 脚注强化 **healthy-responder** → 若纯属偏倚则 **跌倒应同降** → **ATE≈0、CI 含零** → **排除最简单 omnibus 解释**，**路径特异性**表述；**未**写成可单独支撑「处方」级断言（留 RCT）。
- **DCA**：**§3.10** 扩写 **net benefit vs treat-all/none**；新 **§4.4** + **Figure 5** 图注；摘要 **Results** 一句。讨论结构 **4.1–4.7**（原 4.3–4.5 后移）。
- **Lancet 式收紧**：引言、Limitations、结论 **短句**；**慢性病负担** 重申 **仅基线协变量**。

**2026-03-29 正文 vs 附录：全量数据入稿**
- **判断**：主文保留 **Table 1 摘要、Table 2 冠军、Table 3 B 队亚组摘录、Table 5 四项干预**；**完整网格**放 **Supplementary Tables S10–S15**（基线全表、STROBE 流失、全算法 CPM、全 PSM/PSW/XLearner、`table4` 含 chronic 审计行、`table5` 敏感性 151 行）。**S8** 扩为 **A/B/C 全部亚组行**；**S9** 内联 **超参数搜索表**；**S2** 标明缩写、**S15** 为全表；**S4** 补 **n_val**；**S5** 指向 **S13**。
- **再生**：`python scripts/_md_appendix_from_csv.py` → 生成 `_tmp_s10.md` 等；**若 S10 已在稿中**，更新时需先 **手动删除** 旧 `## Supplementary Table S10`–`S15` 块再运行 `python scripts/inject_supplementary_tables.py`（脚本遇重复会 abort）。

**2026-03-29 终稿叙事：`chronic_low` 不作为暴露 / Table 5 与全文一致**
- **问题**：若将 **low chronic disease burden** 与运动等并列为主文 **ATE 表** 中的“干预”，易被误读为可操纵治疗，且在认知受损层出现“正向关联”时引发 **悖论式** 临床误读。
- **主文处理**：**暴露族**固定为 **四项**生活方式（运动、饮酒、隔离、BMI）；**慢病负担**仅作 **基线协变量**（§2.3）。**Table 5**、**Figure 4** 图注、**Supplementary Text S3** 中 **five → four**；讨论 **§4.2** 改为 **Cohort C 反向因果/选择** + **非干预化慢病**，删除原 **“Paradox of Chronic Disease”** 作为独立暴露叙事。**Figure 2**、**Table 1** 脚注与之一致。
- **数据文件**：`table4_ate_summary.csv` 等仓库导出仍可能含历史 **chronic** 行；主文以 **Methods + Table 5** 为准，必要时在补充材料注明 **审计用全表**。

**2026-03-29 投稿稿「亮点」续写：`PAPER_Manuscript_Submission_Ready.md`**
- 在摘要关键词后增加 **`What this study adds`**（3 条短 bullet，供期刊 novelty/significance 字段粘贴，注明可按门户字数再截断）与 **`Plain-language summary`**（非专业读者可读）。
- **投稿 checklist** 中补充材料表项由 **S1–S7** 改为 **S1–S9**（与正文补充目录 Table S8/S9 一致）。
- 若此前未保存：**标题页 `Highlights`（5 条）**、引言 **`Contributions`**、结果 **`§3.0 Key findings at a glance`**、讨论 **`Strengths in summary`** 已用于突出方法与因果审计亮点；**数值**仍以 **2026-03-28 lock** 与 `table2_prediction_cohort*`、`table4_ate_summary.csv` 为准。

**2026-03-27 扩展分析重跑：`MAIN_MINIMAL_CAUSAL_RERUN=False`**
- 为执行 **Rubin 合并、截断敏感性、多暴露、扩展干预、XLearner 全干预、PSM 全队列交叉、低样本优化、生理因果**，须 **`MAIN_MINIMAL_CAUSAL_RERUN=False`**（`run_all_charls_analyses` 中 `if not _minimal_causal:` 块）。
- **副作用**：同时恢复 **Table1b 发病密度与组合发病图**、**概念图/Table1/流失图/队列前插补敏感性**；且 **`_minimal_causal` 为 False 时会调用 `_maybe_run_npj_imputation_first()`**（若 `RUN_IMPUTATION_BEFORE_MAIN=True` 则先跑 bulk MICE，耗时长）。仅重扩展、不重插补时设 **`RUN_IMPUTATION_BEFORE_MAIN=False`**。
- 若**只要扩展、不要前置 Table1 等**：可临时 **`MAIN_SKIP_STEPS_BEFORE_COHORTS=True`**（仍跑三队列与 Rubin 等，视 `MAIN_COHORT_CAUSAL_ONLY` 而定）。

**2026-03-25 投稿补充：阴性对照 / Love plot / DML 敏感性 / CATE 密度**
- **阴性对照**：`estimate_causal_impact_xlearner(..., outcome_col=...)`；默认对照结局 `config.NEGATIVE_CONTROL_OUTCOME_COL='is_fall_next'`（CHARLS 无统一 `accidental_injury` 列时常用跌倒下期作弱相关对照）。`run_negative_control_outcome_summary(df_clean)` → `results/tables/negative_control_results.csv`；`run_auxiliary_steps=False` + 临时目录，**不覆盖**各队列 `03_causal/`。主流程：`config.RUN_NEGATIVE_CONTROL_OUTCOME` + `run_all_charls_analyses` 在队列跑完后调用。
- **DML 鲁棒性**：`estimate_causal_impact_dml_sensitivity` 与 XLearner **同源** PS 修剪 + `exercise_x_adl` 交互；`run_ate_method_sensitivity_cohort_b` → `results/tables/ate_method_sensitivity.csv`。开关 `config.RUN_ATE_METHOD_SENSITIVITY`。
- **Love plot**：`scripts/plot_love_plot.py` 合并 `assumption_balance_smd_{T}.csv` 与 `*_weighted.csv`（非 txt）；默认 Cohort B → `results/figures/love_plot_cohort_B.png`；竖线 |SMD|=0.1。
- **CATE 密度**：`scripts/plot_cate_density.py` 增加 q25/q50/q75 竖线与 **P(τ̂>0)** 文本框。

**2026-03-27 因果轻量重跑：`config.MAIN_MINIMAL_CAUSAL_RERUN`**
- **`True`**：跳过 **本轮 npj 前置插补**、**概念图/Table1/流失/插补敏感性**、**Table1b 发病密度与组合发病图**、队列后 **Rubin/截断敏感性/多暴露/扩展干预/XLearner全干预/PSM全队列交叉/低样本优化/生理因果**；仍 **读磁盘插补表** 并跑 **三队列 causal_only**（与 `MAIN_COHORT_CAUSAL_ONLY` 或关系）。汇总图 **AUC** 从已有 `table2_*_main_performance.csv` 回填。全流程请 **`False`**；建议轻量时 **`RUN_IMPUTATION_BEFORE_MAIN=False`** 以免误以为要刷新插补（代码在 minimal 下已不跑前置插补，但若该项 True 会打 WARNING）。

**2026-03-27 主流程仅跑因果：`config.MAIN_COHORT_CAUSAL_ONLY`**
- **`True`**：`run_all_charls_analyses` → `run_cohort_protocol(..., causal_only=True)`，跳过 **compare_models / SHAP / 临床评价 / 决策 / 外部验证 / 剂量反应**；仍跑 **XLearner 等因果**及 **ITE/列线图/时序/插补敏感性/亚组**；不删队列目录（与 `pool_only` 类似保留已有输出）。数据加载与 `df_for_causal` 与主流程一致（插补表切片）。

**2026-03-27 PS 修剪敏感性 + ITE 密度图**
- **修剪实现**：`causal/charls_recalculate_causal_impact.py` 中 **`_apply_ps_overlap_trim`**（PS LogisticRegression + `in_band` + `force_trim`）；**XLearner** 在 **`estimate_causal_impact_xlearner`** 步骤 2 调用；**TLearner** 同源调用。日志含 **`N_before_trim` / `N_after_trim`**。成功返回的 `df_sub.attrs["ps_trim_info"]` 供脚本读修剪样本量。
- **敏感性汇总**：`scripts/run_ate_ps_trim_sensitivity.py` → **`results/tables/ate_sensitivity_trimming.csv`**（循环 `config.PS_TRIM_SENSITIVITY_SCENARIOS` × 三队列；`--bootstrap-n` 默认 80 以加速，终稿可与主分析对齐为 200）。`estimate_causal_impact_xlearner(..., bootstrap_replicates=..., bootstrap_min_for_ci=...)` 控制聚类 bootstrap。CSV 含 **`applied_subset`、`pct_in_band_before_subset`、`overlap_trimmed_pct`、`ate_ci_source`、`bootstrap_ate_successful_draws`、`p_value_footnote`**；`df_sub.attrs['ps_trim_info']` 同步上述诊断（及 bootstrap 计划次数）。
- **CATE 密度**：主流程 XLearner 辅助步写出 **`results/tables/ite_xlearner_{treatment}_cohort_{A|B|C}.csv`**；`scripts/plot_cate_density.py` 默认读 **Cohort B** 的 `tau_hat`，**红色竖线 τ=0**，输出 **`results/figures/cate_density_cohort_B_{treatment}.png`**。

**2026-03-27 合并预测 Table 2（A+B+C）**
- **`utils/charls_table2_combine.write_combined_table2_prediction`** → **`results/tables/table2_prediction_combined_ABC.csv`**（首列 `Cohort`，按队列与 AUC 降序）。
- **`scripts/merge_table2_prediction_ABC.py`** 可单独运行；**`run_all_charls_analyses`** consolidate 复制三队列 table2 后自动调用。

**2026-03-27 发病组合图（累积发病率 + 人年密度）**
- **`viz/fig_incidence_cumulative_and_density.py`**：`draw_incidence_combined_figure(df, ...)` — 左栏 person-wave 粗发病率（%）+ 正态近似 95% CI；右栏与 **`charls_incidence_density`** 一致的中点法 **per 1,000 PY** + Poisson 正态近似 CI。
- **`scripts/plot_incidence_combined_figure.py`**：默认读 `preprocessed_data/CHARLS_final_preprocessed.csv`，输出 **`results/figures/fig_incidence_cumulative_and_density.png`**，附 **`results/tables/fig_incidence_combined_source_stats.csv`**。
- **`run_all_charls_analyses`**：在成功写出 `table1b_incidence_density.csv` 后同数据调用上图（与 `df_for_table1b` 一致）。

**2026-03-26 Person-level Table 1（STROBE / entry wave）**
- **`scripts/build_table1_person_level.py`**：读 `preprocessed_data/CHARLS_final_preprocessed.csv`（或现场 `preprocess_charls_data`）；`sort_values(['ID','wave'])` + `drop_duplicates('ID', keep='first')` → 每人一行（约 7,027 ID，与 14,386 person-waves 一致）。
- **与主 Table 1 同结构**：调用 **`data.charls_table1_stats.tabulate_baseline_table_bps`**（`BPS_SECTIONS`）；`prepare_exposures` 补 **`sleep_adequate`**；P 值与主表一致为 **Kruskal-Wallis + Chi-square**；列名为 **A/B/C + P-value**。
- **输出**：`results/tables/table1_person_level.csv`。若预处理宽表无 **`smokev`**，则不会出现「Current smoking」行（需在 `charls_complete_preprocessing` 的 `semantic_features` 中纳入该列后重跑预处理）。

**2026-03-25 Code Freeze（冠军 / Table1b / Bootstrap 透明）**
- **`_select_champion`**：`df_main`（Table 2）为唯一权威（Recall≥0.05 后最高 Test AUC）；非权威时 `logger.error` 并回退 `perf_df`；`rewrite_model_performance_full_csv` 将真实冠军置顶。
- **Table 1b**：若存在 `df_pre`，发病率密度与 Table 1 一致基于 **`df_pre`**，否则 `df_clean`。
- **Bootstrap**：`charls_causal_multi_method` 中 T/XLearner 行 bootstrap 打印 planned/success/fail；`charls_recalculate_causal_impact` 中 cluster bootstrap 在 `failed_or_skipped>0` 时用 **`logger.warning`**（否则 `info`）+ `print`。
- **`compare_models`**：若使用 `return_xy_test=False` 或 `save_roc_path`，**`logger.warning`** 提示冠军为 AUC 首行、与 Table2 可能不一致；正式管线走 `run_cohort_protocol`。

**2026-03-26 全流程分析说明文档**
- 用户索要「与插补说明同等粒度」的全文数据分析说明：已写入 **`docs/全流程数据分析详细说明.md`**（数据源双层表、主流程顺序、`run_cohort_protocol` 逐步、Rubin/敏感性/consolidate、config 速查、脚本 `load_supervised_prediction_df` vs `load_df_for_analysis`）。

---

**2026-03-26 预测插补方案 A（消除 bulk+Pipeline 自相矛盾）**
- **compare_models**：`SimpleImputer` 改为 `utils/charls_sklearn_preprocess_pipelines.build_numeric_column_transformer`（**IterativeImputer** + Scaler，CV 训练折 fit）。
- **run_all_charls_analyses**：`df_pre` 存在时 CPM/SHAP 用 **预处理缺失** 队列切片（`df_pa/b/c`）；因果/时序/剂量/亚组用 **`df_for_causal`**（插补后 `df_a/b/c`）。`run_cohort_protocol(..., df_for_causal=...)`。
- **load_supervised_prediction_df()**：脚本与 `run_prediction_only`、`run_cpm_table2_only`、`compare_auc_with_without_drop`、`test_three_mwaist_configs` 与主流程预测步对齐；因果仍用 `load_df_for_analysis()`。
- **config**：`ITERATIVE_IMPUTER_MAX_ITER`；`USE_IMPUTED_DATA` 注释更新。**文稿**：`PAPER_Manuscript_Submission_Ready.md` §2.1/2.4、摘要 Methods；`docs/MEDICAL_CODE_CAUTIONS.md`、`project.md`、`attrition_flow_readme.txt` 说明。

---

**2026-03-25 审稿清单对照（插补 / cluster bootstrap / 文稿）**
- **插补层次**：主分析仍可能先读全样本 MICE 的 `step1_imputed_full.csv` 再划分；Pipeline 内 Imputer 仅在训练折 fit，但**不能**等同于「所有缺失仅在训练折插补」。顶刊若要求 IterativeImputer 全程在 Pipeline 且仅用原始缺失表，需改数据入口与叙事。
- **Cluster bootstrap（已实现 2026-03-25）**：`utils/charls_ci_utils` 新增 `cluster_bootstrap_indices_once`；`get_metrics_with_ci(..., groups=测试集ID)`；`charls_cpm_evaluation._bootstrap_ci_at_threshold` / `evaluate_and_report` / `evaluate_single_model` 支持 `groups_test`；`compare_models` 对测试折传 `groups` 并 `return_xy_test` 增加 **`groups_test`**；`run_all_charls_analyses`、`scripts/run_cpm_table2_only` 传入 CPM；`charls_recalculate_causal_impact` 中 TLearner/XLearner fallback bootstrap 按 **`df_sub['ID']`** 聚类（无 ID 时警告并退化为行级）。`groups=None` 时 CI 仍为旧分层行级 bootstrap（兼容）。
- **已对齐清单项**：`CALIBRATE_CHAMPION_PROBA` + `CalibratedClassifierCV`+GroupKFold；XLearner **ATT** 已算并写入 `ATE_CI_summary_*`；`PSM_DOUBLE_ADJUST_LOGIT` + `_psm_double_adjust_logit`。
- **论文（已改 2026-03-25）**：`PAPER_Manuscript_Submission_Ready.md` §2.4 CPM 改为「整体 AUC 选模 + 内验 Youden 定阈」；性能 CI 与 §2.7 XLearner bootstrap 写明 **cluster bootstrap by participant ID**；Table 2 脚注与之一致。可补 **TRIPOD+AI 2024**；Cohort C max SMD 与 assumption 输出对齐仍须在改稿时核对。

---

**2026-03-24 终稿 config + Pipeline 插补 + 冠军校准**
- **`config.py`**：`CALIBRATE_CHAMPION_PROBA=True`，`USE_TEMPORAL_SPLIT=True`，`N_MULTIPLE_IMPUTATIONS=5`，`USE_RUBIN_POOLING=True`；注释区分 bulk MICE 与 sklearn 训练子集 fit。
- **`utils/charls_train_only_preprocessing.py`**：`get_train_indices_for_preprocessor` / `fit_transform_numeric_train_only`（与 `compare_models` 同款 ColumnTransformer+Pipeline，仅训练子集 fit）。
- **因果/PSM/假设检验**：`charls_recalculate_causal_impact`、`charls_causal_methods_comparison`、`charls_causal_assumption_checks` 已弃用「全表 `SimpleImputer.fit_transform`」，改用上述工具。
- **`run_all_charls_analyses.run_cohort_protocol`**：`compare_models` 现返回 `X_train,y_train,grp_train`；若 `CALIBRATE_CHAMPION_PROBA`，用 `CalibratedClassifierCV(..., GroupKFold)` 在训练子集上拟合后保存 `champion_model.joblib`。
- **`scripts/run_cpm_table2_only.py`**：`--full` 解包需含训练集返回值（未校准时仍为旧 table2 逻辑）。
- **`archive/charls_imputation_npj_style.py`**：`N_MULTIPLE_IMPUTATIONS` 默认 `from config import`。

---

**2026-03-14 人年发病密度脚本**
- **`scripts/compute_incidence_density_person_time.py`**：长表 incident 样本；每 interval 未发病 2 PY、发病 1 PY（中点）；首发后不再累计；分层取首条 person-wave；**Total N = unique ID**（14,386 行约对应 7k+ 人）。性别编码与 **`charls_table1_stats`** 一致：**Female=1, Male=0**；**rural=1** 为农村。输出 **`results/tables/table_incidence_density_person_time.csv`** 与 `*_person_level.csv`。

**2026-03-14 与代码一致的纳排图**
- **`python -m viz.draw_attrition_flowchart`**：读 `preprocessed_data/attrition_flow.csv`，CONSORT 主列 + 右侧 Excluded n 与理由（对齐 `charls_complete_preprocessing.py`）；默认另存 **`results/figures/fig1_attrition_flow_code_aligned.png`**（300 dpi）。

**2026-03-14 纳排图 vs 主流程**
- **现象**：手绘 CONSORT（如 2011/2013 受访者 n=20,284 → 失访/ADL-IADL/残疾/BPS 完整病例 → n=10,362）与仓库主预处理**不一致**。
- **主流程**：`data/charls_complete_preprocessing.py` 以 **person-wave** 为行；`preprocessed_data/attrition_flow.csv` / `results/tables/table1_sample_attrition.csv`：**96,628 → 49,015 → 43,048 → 31,574 → 16,983 → 14,386**（入射、基线无共病）；无图中四条减员与 10,362；ADL/IADL/BPS 非该链式列表删除叙事，缺失主要在 Pipeline/插补阶段处理。
- **文稿**：`PAPER_Manuscript_Submission_Ready.md` §3.1 与 **14,386** 对齐；若保留 10,362 图须注明**另一分析定义**，避免与主文冲突。

**2026-03-14 Streamlit 特征表：按论文 Table1 BPS 子 Tab（无「类别」列）**
- **`utils/bps_feature_groups.py`**：解析 `data/charls_table1_stats.BPS_SECTIONS`，`order_columns_for_editor` 得展示顺序 + `bps_sec_*` i18n 键；未映射列 → `bps_sec_other`。
- 表列：**variable | value**；BPS 模块名由顶部子 Tab 显示，表中不再重复「类别」列；`pd_step1_desc` 说明与 Table 1 一致。

**2026-03-14 Streamlit 特征表：三列 + 子样本 min–max + 裁剪**
- 表列为 **variable | range_ref | value**；`range_ref` 为子样本 `X_all` 各列 min/max 文本；初值为**中位数**（已裁剪）；合并回写时 **`np.clip` 到 bounds**。
- 已删：manual_hint、manual_n_feat、行索引滑条、载入行/中位数按钮；`pd_step1_desc` 单行说明。

**2026-03-14 Streamlit 精简：仅手动特征 + 无模型元数据 UI**
- **移除**：`st.expander` 模型元数据（冠军/table2 校验卡片）；**子样本选行**整段（radio、种子、随机、slider、右侧提示）。
- **保留**：纵向 `data_editor`；**载入滑条行 / 载入队列中位数**（仍用子样本与中位数）；预测 + SHAP；图标题固定用手动版。
- **删除代码**：`_model_stat_cards`、`model_complexity_efficiency`/table2 解析冠军、`_tags_compatible` 等整块（仅本文件内）。
- **文档**：`SHAP_Streamlit_三队列使用说明.md`、`PAPER_Web_App_论文表述建议.md` 已同步。

**2026-03-14 Streamlit 论文附录取向：扁平 + 短文案**
- **动机**：导师/投稿截图要简洁、少「产品感」。
- **CSS**：主区 **max-width 900px**；白底卡片、**无 box-shadow**、无渐变按钮/预测卡；圆角 4–6px；侧栏灰底扁平；队列顶栏仅 **3px 左色条 + 小字 Cohort kicker**（去掉胶囊徽章）。
- **步骤**：`_pd_flow_step` 改为 **`h5`「1. 标题」+ `paper-step-desc`**，无彩色数字块。
- **文案**：`page_title`/`brand_short`/`home_*`/三步标题与说明**缩短**；导航「说明与向导」→「说明」；按钮「进入」→「打开队列」；英文明同步。
- **图**：matplotlib 脸与轴区改为 `#f5f4f2` / `#ffffff`。
- **文档**：`docs/PAPER_Web_App_论文表述建议.md`（中英一句式 + 截图建议）。

**2026-03-14 Streamlit 产品设计向 UI 改版（tokens + 步骤徽章）**
- **CSS**：`:root` 设计 tokens（ink/surface/line/shadow/radius）；首页 Hero `pd-hero` + eyebrow；分区 `pd-section-label`；队列顶栏 `pd-cohort-hero` + COHORT 徽章（`_hex_to_rgba`）；三步 `_pd_flow_step`（数字色块 + 标题 + 一句 desc）；侧栏 `pd-sidebar-brand`；卡片/按钮/Expander/DataEditor 统一圆角与轻阴影。
- **文案**：`brand_short`、`home_eyebrow`、`pd_step1..3_title/desc`、`btn_enter_cohort`；移除旧 `flow_step_*` 长标题。
- **结构**：首页关于/快速入口/向导用 section label + `pd-card` / `pd-quick`；快捷按钮文案改为 i18n「进入/Open Cohort」。

**2026-03-14 Streamlit 队列页版式：三步流 + 折叠模型区**
- **诉求**：主区信息平铺、抓不住重点。
- **做法**：**模型与校验**收入 **默认折叠** `expander`；主流程改为 **① 输入 → ② 输出（预测单独卡片 + 放大 metric）→ ③ SHAP**；`st.divider` 分段；子样本模式右侧用 **浅底提示框** 说明「改左侧则下方更新」。`st.container(border=True)` 包裹预测（旧版 Streamlit 无参数时回退普通 container）。CSS：`flow-step-h`、`input-hint-box`、`stVerticalBlockBorderWrapper` 渐变卡、`stMetricValue` 略放大。

**2026-03-14 Streamlit 手动模式：横向单行表易漏看 → 纵向清单**
- **现象**：`st.data_editor` 单行宽表列多、需横向滚动，英文变量名截断，用户反馈不美观、易看漏。
- **做法**：手动模式改为 **`variable`（只读）+ `value`（可编辑）** 纵向表，固定高度约 440px 垂直滚动；可选 **变量名筛选**（`contains`，不区分大小写），编辑结果合并回全量列再 `predict_proba`；载入滑条行/中位数时 `bump` 刷新编辑器 key，避免旧 widget 缓存。旧键 `de_feat_{cohort}` 在手动分支 `pop` 防冲突。
- **文件**：`streamlit_shap_three_cohorts.py`（`_wide_one_row_to_vertical` / `_vertical_to_wide_row`）；说明见 `docs/SHAP_Streamlit_三队列使用说明.md`。

**2026-03-23 投稿稿与 2026-03-22 全量跑批结果对齐**
- **数据源**：`results/tables/table2_prediction_cohort*.csv`、`table4_ate_summary.csv`、`table6_external_validation_cohort*.csv`、`table3_subgroup_cohort*.csv`、`table5_sensitivity_summary.csv`、`Cohort_*/03_causal/ATE_CI_summary_exercise.txt` 与 `assumption_*`。
- **相对旧稿主要变化**：CPM 冠军 **A=XGB、B=AdaBoost、C=SVM**；**XLearner 运动在 B 的 CI 含 0**；**PSM/PSW 在 B 仍显著保护方向**；**C 运动 XLearner 为正且 CI 排除 0**（文稿中作混杂解释）；重叠 **B 约 1.6% trim**（非 50%）；外部验证 AUC **约 0.56–0.70**；方法学改为**单次插补**主分析叙事。
- **已更新文件**：`PAPER_Manuscript_Submission_Ready.md`（摘要、方法 §2.1/2.4、结果 §3.3–3.10、讨论 4.1–4.5、Table 2–5、Supp S2/S4/S5/S6/S7、Data lock-in）。

**2026-03-23 三队列 SHAP Streamlit 演示（导师「在线分析」需求）**
- **入口**：项目根 `streamlit_shap_three_cohorts.py`；说明 `docs/SHAP_Streamlit_三队列使用说明.md`；`README_运行与输出说明.md` § 运行顺序 6。
- **依赖**：`step1_imputed_full.csv`（含 `baseline_group`）+ 各队列 `01_prediction/champion_model.joblib`；运行 `streamlit run streamlit_shap_three_cohorts.py`。
- **逻辑**：与 `charls_shap_analysis` 一致解包 Pipeline；Tree/Linear/Kernel Explainer；SVM 可能较慢。公网部署勿上传可识别个体数据。
- **2026-03-23 更新**：侧边栏 **中/英** 切换；冠军展示 **不以 table2 首行为准**，优先 `model_complexity_efficiency.txt` 的 `Champion Model:`，否则 **按 AUC 排序 table2**；**joblib 按文件 mtime 缓存**；与记录不一致时 **黄色警告** 并提示重跑预测步（避免三队列都显示旧 LR 却与论文 Table 2 不符）。
- **2026-03-23 导航精简**：**删除「说明」首页**；侧栏仅 **三选一**（索引 0/1/2），默认 **基线健康队列**（原 A）。中文标签：**基线健康队列 / 基线抑郁队列 / 基线认知障碍队列**；英文：`nav_a/b/c` 为 Baseline healthy / depression / cognitive impairment cohort。旧 session 四 tab 索引迁移：`1–3→0–2`，原首页 `0→0`。
- **2026-03-23 had_comorbidity_before 排除**：该变量为队列定义（incident 样本恒为 0），非预测特征。已加入 **`utils/charls_feature_lists.EXCLUDE_COLS_BASE`**。预测模型现为 **34 特征**。**重要**：此前训练的 `champion_model.joblib` 为 35 特征；修改后需**重跑预测步**（`run_all_charls_analyses.py` 或 `scripts/run_cpm_table2_only.py`）以生成新的 34 特征模型，否则 Streamlit 会因列数不匹配报错。

**2026-03-14 Windows PermissionError 写 model_performance_full + 续跑**
- **现象**：`compare_models` 写 `Cohort_B/01_prediction/model_performance_full_*.csv` 时 `[Errno 13] Permission denied`，常因 **Excel/WPS 打开该 CSV**。
- **修正**：① 关闭占用程序；② `modeling/charls_model_comparison.py` 对关键 `to_csv` 增加重试；③ **`python scripts/resume_charls_cohorts.py B,C`**（或 `config.RUN_COHORTS_ONLY` + `MAIN_SKIP_STEPS_BEFORE_COHORTS`）只重跑 B/C，A 从磁盘恢复汇总。

**2026-03-14 主分析须用最新插补**
- **需求**：主分析不用「以前保留的」旧 `step1_imputed_full`，而用本轮最新插补。
- **做法**：保持 `RUN_IMPUTATION_BEFORE_MAIN=True`，主流程先跑 `run_full_experiment` 再读 `IMPUTED_DATA_PATH`（覆盖写入）。若前置插补失败，打 ERROR 提示可能仍读旧文件。
- **辅助**：`utils/imputation_data_provenance.py` — 加载时打印插补 CSV 修改时间；`WARN_IMPUTED_OLDER_THAN_PREPROCESSED` 在关前置插补且预处理表新于插补文件时 WARNING。

**2026-03-14 主分析：单次插补（关闭 Rubin / 不生成 m1..mN）**
- **决策**：主文与主流程以 **`step1_imputed_full.csv`** 为准；不将 Rubin 合并作为主估计。
- **配置**：`config.N_MULTIPLE_IMPUTATIONS=0`、`USE_RUBIN_POOLING=False`；`archive/charls_imputation_npj_style.py` 中 **`N_MULTIPLE_IMPUTATIONS=0`**（`run_all_charls_analyses` 前置插补时会从 config 同步 `_npj.N_MULTIPLE_IMPUTATIONS`）。恢复多重插补 + Rubin 时两处同时改回（如 5 + True）。
- **代码**：`_run_multi_imputation_rubin_analysis` 在 `not use_rubin` 或 `n_mi < 2` 时直接跳过。

**2026-03-22 前置插补数据入口修复**
- **问题**：`RUN_IMPUTATION_BEFORE_MAIN` 曾把 **RAW `CHARLS.csv`** 传给 `run_full_experiment`，无 **`baseline_group`** → 插补立即失败。
- **修正**：`_maybe_run_npj_imputation_first` **优先** `preprocessed_data/CHARLS_final_preprocessed.csv`；若无则 **`preprocess_charls_data(RAW_PATH)`** 写出后再插补。

**2026-03-22 config：RUN_IMPUTATION_BEFORE_MAIN=True**
- 用户选择「一条命令先插补再主分析」；`config.py` 已设为 True。若仅调模型不重算插补，可临时改 False 以节省时间。

**2026-03-22 主流程可选「前置插补」**
- **误解**：用户以为已改主代码即可每次自动重跑插补；原先主流程**只读** `step1_imputed_full`，不调用插补脚本。
- **实现**：`config.RUN_IMPUTATION_BEFORE_MAIN`（默认 `False`）+ `run_all_charls_analyses._maybe_run_npj_imputation_first()`，为 True 且 `USE_IMPUTED_DATA` 时调用 `archive.charls_imputation_npj_style.run_full_experiment`，并将脚本内 `N_MULTIPLE_IMPUTATIONS` 同步为 `config.N_MULTIPLE_IMPUTATIONS`。`IMPUTATION_OUTPUT_ROOT` 与 `IMPUTED_DATA_PATH` 对齐说明写在 `config` / `README_运行与输出说明.md`。

**2026-03-22 初稿对齐 Cohort_* + LIU_JUE（与 results table2 可能不一致）**
- **现象**：`results/tables/table2_prediction_axisA.csv` 与 `Cohort_A/01_prediction/table2_A_main_performance.csv` 中 AUC/冠军不一致（后者 CPM 冠军 A：**MLP ~0.744**；B：**ExtraTrees ~0.673**；C：**LR ~0.661**）。
- **处理**：`PAPER_Manuscript_Submission_Ready.md` 摘要、§3.3、§3.5、§3.9–3.10、Table 1 年龄、Table 2、Supp Table S7、Data lock-in / footer 已按 **Cohort_* + `table4_ate_summary.csv` + `LIU_JUE` table1 / attrition** 更新；**校准斜率** B≈4.42、C≈1.02（`04_eval/calibration_brier_report.txt`）。**E-value/SMD/修剪比例** 按各队列 `03_causal/assumption_checks_summary.txt` 与 B 的 `ATE_CI_summary_exercise.txt`。
- **清单**：`PAPER_图表文件核对清单.md` 增加「当前磁盘快照」与 Table1/2/3/S4 的 **Cohort 首选路径**。

**2026-03-14 可复现快照 + README Cohort 对齐 + project.md**
- **快照**：`scripts/save_reproducibility_snapshot.py` — 默认 `runs/repro_snapshots/时间戳/`，含 `git_*`、`config_snapshot`、`pip freeze`、`conda env export`、`python_version`；用法见 `README_运行与输出说明.md` §环境与复现。
- **README**：文首增加 Cohort ↔ Axis 目录说明，链到 `config.COHORT_*` 与 `run_cohort_protocol`；主流程表述改为三队列；输出表改为 `Cohort_*` 主名并注旧名 `Axis_*`。
- **.gitignore**：增加 `Cohort_*` 输出目录；`runs/repro_snapshots/` 默认忽略（防 pip freeze 过大）。
- **project**：新建 `.remember/memory/project.md`（语言、图表 cohort 优先、数据勿改原文件、快照命令）。

**2026-03-14 论文 MD 路径同步 cohort 命名**
- **已更新**：`PAPER_图表文件核对清单.md`、`PAPER_Manuscript_Submission_Ready.md`（Data lock-in + 正文引用）、`PAPER_完整版_2026-03-20.md`、`PAPER_Main_Figures_Mapping.md`、`PAPER_框架与图表索引_代码逻辑版.md`、`PAPER_写作前最终检查清单.md`、`PAPER_绝对路径索引.md`、`PAPER_附录全部绝对路径.md`、`PAPER_目录框架与修订清单_2026-03.md`、`PAPER_代码优化建议_论文写作视角.md`、`PAPER_Multiplicity_FDR_Code_Alignment.md`、`运行结果简报_2026-03-20*.md`、`数据分析逻辑流程_导师汇报.md`。统一 **首选 `*_cohort*`，注明 `*_axis*` 为 consolidate 兼容副本**；生理学「HPA axis」等未改。

**2026-03-14 术语：axis→Cohort（代码）+ 医学易错备忘**
- **config**：`COHORT_A/B/C_DIR`、`COHORT_STEP_DIRS`、`PARALLEL_COHORTS` 为主名；保留 `AXIS_*` / `PARALLEL_AXES` 别名仅兼容旧 import。
- **主流程**：`run_axis_protocol` → `run_cohort_protocol`；`run_all_cohorts_comparison`（旧名 `run_all_axes_comparison` 仍可用）；consolidate 对 `results/*` **双写** `*_cohort*` 与 `*_axis*` 文件名。
- **汇总表列名**：干预/XLearner/physio/multi_exposure/sensitivity/causal comparison 等 CSV 中 `axis` → `cohort` 或 `cohort_id`+`cohort_label`（敏感性）；`_add_consistency_column` 仍可读旧列名 `axis` 并自动映射。
- **文档**：新增 `docs/MEDICAL_CODE_CAUTIONS.md`（插补层次、因果失败、可讨论点）；勿改动 pandas/matplotlib 的 `axis=0/1` 及论文中「HPA axis」生理学用语。

**2026-03-14 脚本数据加载统一（SHAP/DCA/AUC 对比/mwaist 测试）**
- **改动**：`utils/charls_script_data_loader.load_df_for_analysis(apply_config_drop=True)` 新增参数；`False` 时不删 `COLS_TO_DROP`，供 `compare_auc_with_without_drop` 的「全列」臂与 `test_three_mwaist_configs`（自定义 BASE_DROP）使用。
- **脚本**：`run_shap_on_saved_models`、`run_dca_on_saved_models` 去掉重复插补/预处理逻辑，改为 `load_df_for_analysis()`；`compare_auc_with_without_drop` 两臂分别 `apply_config_drop=False/True`；`test_three_mwaist_configs` 用 `load_df_for_analysis(False)` 后按方案 `drop` + `reapply_cohort_definition`。
- **清理**：`run_all_physio_causal` 移除未使用的 `preprocess_charls_data` / `RAW_DATA_PATH` / `AGE_MIN` import。

**2026-03-14 收尾：敏感性 + CPM Table2 与主流程对齐**
- **问题**：`run_sensitivity_scenarios.run_one_scenario` 因果失败时仍可能把 `(0,0,0)` 写入汇总；`run_cpm_table2_only` 自建 `_load_df_clean` 与 `load_df_for_analysis` 重复。
- **修正**：`run_one_scenario` 在 `res_df is None` 或 ATE 为 None/NaN 时统一写 NaN；`run_cpm_table2_only` 的 `_load_df_clean` 改为调用 `load_df_for_analysis()`，并精简不再使用的 config/import。

**2026-03-14 代码修正：脚本数据源 + 因果失败占位**
- **问题**：`scripts/run_all_interventions_analysis.py` 的 `main()` 仅用 `preprocess_charls_data('CHARLS.csv')`，与 `USE_IMPUTED_DATA=True` 时主流程不一致；且因果失败返回 `(None,(0,0,0))` 时用 `ate is None` 判断，**从未成立**，失败被当成 ATE=0 写入汇总。
- **修正**：新增 `utils/charls_script_data_loader.load_df_for_analysis()`；`run_all_interventions`、`run_all_physio_causal`、`run_xlearner_all_interventions`（无传入 df 时）统一调用；干预/physio 循环以 `res_df is None` 置 NaN（physio 原先把失败记成 ATE=0）。`get_estimate_causal_impact` docstring 注明失败约定。

**2026-03-14 论文「插补仅在训练折」表述与代码矛盾 — 已修正**
- **错误**：主稿/补充写「imputation fitted strictly within training folds」，但 `USE_IMPUTED_DATA` 时 MICE 在**全样本**上生成 CSV 后再划分 train/test → 审稿可判为方法学不实。
- **修正**：`PAPER_完整版_2026-03-20.md`（§2.2 + Supplementary Text S1）、`PAPER_Manuscript_Submission_Ready.md`、`PAPER_npj_style_polished.md` 改为与实现一致：bulk MICE 在全分析队列先于分组划分；Pipeline 预处理仅在训练折 fit；CPM 阈值不用测试集。`config.py` 增加「方法学事实」注释防再写错。
- **二次排查（同日复扫）**：补同步 `PAPER_Supplementary_Materials.md`（Text S1 Missing data）、`PAPER_npj_style_polished_Full.md`、`imputation_npj_results/pipeline_trace/README_溯源说明.txt`、`PAPER_写作前最终检查清单.md`（新增 §1b）；仓库内已无 “strictly within … training folds” 类错误英文句。

**2026-03-14 多重插补 + Rubin 规则流程实现**
- **流程**：① 6 种方法单次插补 → ② NRMSE 选优 → ③ 最优方法重复运行 5 次（m1..m5）→ ④ Rubin 规则合并 AUC/ATE。
- **实现**：`utils/rubin_pooling.py`（Rubin 合并）、`config.py`（N_MULTIPLE_IMPUTATIONS、IMPUTED_MI_DIR、USE_RUBIN_POOLING）、`run_all_charls_analyses._run_multi_imputation_rubin_analysis`（加载 m1..m5、pool_only 模式跑预测+因果、合并并保存 rubin_pooled_auc.csv、rubin_pooled_ate.csv）。
- **输出**（启用 Rubin 时）：AUC/ATE 可写 `table2_rubin_pooled_auc.csv`、`table4_rubin_pooled_ate.csv`；SHAP/图表仍用 `step1_imputed_full`。**2026-03-14 起默认** `N_MULTIPLE_IMPUTATIONS=0` + `USE_RUBIN_POOLING=False`，主表以 Cohort 单次插补跑出的 Table 2/4 为准。

**2026-03-21 插补数据多变量异常 — 根因与修复**
- **根因**：sklearn 丢弃全 NaN 列，宽表列错位。
- **根本修复（已实现）**：`charls_imputation_npj_style.py` 新增 `_prefill_all_nan_wide_cols`，在宽表插补前对全 NaN 列用同变量其他 wave 中位数占位，避免被丢弃。6 种方法、NRMSE 选优、宽表纵向插补、多重插补等设计均保留。
- **临时兜底**：run_all_charls_analyses 对异常变量用预处理覆盖（插补脚本修复后仍可保留作双重保险）。

**2026-03-21 Table 1 BMI 错误修复**
- **现象**：三队列 BMI 均值分别为 13.74、13.30、12.83 kg/m²，显著低于正常范围 18.5–24，与「正常 BMI」干预定义矛盾。
- **原因**：1) 插补后数据 `step1_imputed_full.csv` 中 bmi 列可能被污染（均值 ~13.5）；2) 预处理未对 bmi 做合理化校验。
- **修正**：① `charls_complete_preprocessing.py`：若存在 mweight/mheight（CHARLS 单位：米、kg），用 bmi=weight/height² 重算并 clip(15,50)；② `run_all_charls_analyses.py`：若插补后 bmi 均值 <16 或 >35，用预处理数据覆盖；③ `charls_table1_stats.py`：bmi clip 改为 15–50，均值异常时 logger.warning。
- **注意**：CHARLS mheight 单位为**米**（非 cm），公式为 `mweight / (mheight**2)`。

**2026-03-14 方法学与结果审稿意见补充**
- **样本量**：新增 §2.1.1 Sample Size，说明 14,386 由入组标准确定、无先验效能分析，事后说明事件数足够 bootstrap。
- **MICE 插补**：§2.2 补充插补变量集、连续/有序/二分类/名义的插补方法、Rubin 规则；新增 **Supplementary Text S1** 完整插补策略。
- **CPM 可操作性**：§2.2 补充「In code: rank by AUC descending, top = champion」。
- **基线 Table 1**：表注补充中位数(IQR)、两两比较见复现输出。
- **Table 2**：表注补充 14 模型排序、雷达图路径、ECE/Brier 分解输出。
- **§3.7 亚组**：补充 CATE 未做交互检验、运动剂量反应因二分类未分析、charls_dose_response 可做连续暴露。
- **§3.9 外部验证**：补充 AUPRC/Brier 95% CI、校准曲线路径。
- **Table S4**：表注补充 95% CI 与校准曲线路径。

**2026-03-14 审稿意见综合回应**
- 新增 **`REVIEWER_RESPONSE_COMPREHENSIVE.md`**：按因果假设、人群、结果、讨论分类，逐项标注代码现状、建议补充、优先级。
- **PAPER_完整版** 已补充：§2.1 CHARLS 波浪（1–4，2011–2018）、基线/随访对应、既往共病定义；§2.5 XLearner RF 固定超参、200 次非分层 bootstrap；§2.6 PSM 卡尺依据（Austin 2011）、PSW 权重修剪（Sturmer 2020）、SMD 明细路径；§2.7 偏倚敏感性模拟（fig_bias_sensitivity.png）。

**2026-03-14 CPM 公式补充（审稿意见）**
- **审稿意见**：CPM 作为冠军选择依据，但未定义 CPM 的具体计算方式（如综合 AUC、Recall、Youden 的加权公式）。
- **澄清**：CPM 在本项目中**不是**加权公式，而是**两阶段选择流程**：(1) 各模型在验证集上 Youden 最优阈值 τ_m；(2) 冠军 = 在各自 τ_m 下 AUC 最高的模型。代码：`_select_champion` 按 `df_main.sort_values('AUC')` 取首行。
- **已做**：在 `PAPER_完整版_2026-03-20.md` §2.2 补充 CPM 正式定义：champion = argmax_m AUC_m(τ_m)，τ_m = argmax_τ [Sensitivity + Specificity − 1] on validation split；引用 TRIPOD [14]；说明「not a weighted composite formula」。

**2026-03-19 P0 论文写作优化**
- consolidate 增加 Table 2（CPM table2_*_main_performance）、Table 3（subgroup）、Table 5（sensitivity_summary）复制到 results/tables
- PAPER_写作前最终检查清单：Table 2 改为 CPM 主表，冠军选择改为 CPM 逻辑，CAUSAL_METHOD 表述放宽
- PAPER_框架与图表索引：Table 2 数据来源改为 CPM

**2026-03-19 持续优化：AXIS_STEP_DIRS + 冠军选择抽取**
- config 新增 AXIS_STEP_DIRS 统一轴线步骤子目录，run_all_charls_analyses 用 _path() 引用
- 抽取 _select_champion()、_save_cpm_champion_outputs() 简化 run_axis_protocol
- consolidate、draw_roc_combined、charls_extra_figures、run_dca_on_saved_models、run_shap_on_saved_models 改用 AXIS_STEP_DIRS

**2026-03-19 CPM 评估集成主流程**
- 将 CPM 的 evaluate_and_report（验证集 Youden 阈值、Bootstrap CI）集成到 run_axis_protocol
- compare_models 新增 return_xy_test 参数，返回 (X_test, y_test) 供 CPM 使用
- 冠军选择改为 CPM 逻辑：按 AUC 排序（Recall 已在最优阈值下提升，无需 Recall>=0.05 门槛）
- CPM df_main 增加 Precision/F1/Accuracy 以兼容 draw_performance_radar
- ROC 数据在确定 CPM 冠军后保存，确保与冠军模型一致

**2026-03-19 调优加强**
- n_iter_tuning: 40→80；n_iter_slow_models: 25→50
- 全部模型统一使用 RandomizedSearchCV（无 GridSearchCV）

**2026-03-19 sleep_adequate 仅用于因果、sleep 用于预测（全量同步）**
- sleep_adequate：因果干预分析专用（EXCLUDE_COLS_BASE 排除，不作为预测特征）
- sleep：预测模型使用连续睡眠时长（小时）
- 已修改：charls_feature_lists.EXCLUDE_COLS_BASE、config 注释、export_prediction_model_baseline（docstring+exclude_rows）、generate_ai_audit_package、compare_auc_with_without_drop、test_three_mwaist_configs（BASE_DROP 移除 sleep）、run_all_interventions_analysis、run_sensitivity_scenarios 注释

**2026-03-19 变量恢复（除 puff 外全部补充）**
- 用户反馈 AUC 效果不佳，恢复 systo/diasto/mwaist/lgrip/wspeed/sleep 以提升预测
- COLS_TO_DROP 现仅含：rgrip, grip_strength_avg, psyche, puff
- charls_feature_lists、charls_table1_stats、run_all_physio_causal、charls_cate_visualization、charls_streamlit_app 已同步更新

**2026-03-16 变量一致性收尾（方案 B：删 mwaist/systo/diasto）**
- charls_visual_enhancement.py：systo 不存在时用 pulse 作为生理平滑图示例；lgrip/rgrip 已移除则跳过握力图
- charls_cate_visualization.py：continuous_cols 更新为 age/bmi/pulse/income_total/family_size/adlab_c/iadl（与 charls_feature_lists 一致）
- charls_streamlit_app.py：移除 sleep/systo/lgrip 滑块，改为 sleep_adequate 单选；干预文案改为“睡眠充足 (≥6h)”
- streamlit_app.py：key_features 仅用于模型实际特征，动态适配，无需修改

**2026-03-16 wspeed 一并移除（高缺失率 ~37-40%）**
- COLS_TO_DROP、physio 因果、基线表 physical 移除 wspeed

**2026-03-16 lgrip 一并移除（高缺失率 ~35-40%）**
- COLS_TO_DROP 新增 lgrip；physio 因果移除握力暴露；基线表 physical 移除 lgrip

**2026-03-16 变量调整：smokev 回归、puff 移除、sleep→sleep_adequate**
- COLS_TO_DROP: 移除 smokev，新增 puff、sleep；保留 smokev、sleep_adequate
- prepare_exposures 在 COLS_TO_DROP 之前调用，构造 sleep_adequate、puff_low
- charls_feature_lists: CONTINUOUS_FOR_SCALING 移除 puff、sleep
- 基线表: lifestyle 改为 smokev、sleep_adequate（binary）
- 插补数据需含 smokev；若 step1_imputed_full.csv 无 smokev，需重新运行插补或从原始数据补充

**2026-03-16 X-Learner 全干预分析**
- 新增 run_xlearner_all_interventions.py：7 类干预（exercise, drinkev, is_socially_isolated, bmi_normal, chronic_low, sleep_adequate, puff_low）× 3 队列
- sleep_adequate：sleep≥6h=1；puff_low：puff≤中位数=1（吸烟包年低）
- 每干预同步执行假设检验（重叠、SMD、E-value、PSM/PSW）
- charls_feature_lists：puff_low 排除 puff，sleep_adequate 排除 sleep

**2026-03-16 新增 X-Learner 因果方法**
- 用户反馈 TLearner 未得出显著效果，尝试 X-Learner（Künzel et al.，治疗/对照组不平衡时通常优于 T-Learner）
- 已添加：charls_causal_multi_method（7 种方法含 XLearner）、charls_recalculate_causal_impact.estimate_causal_impact_xlearner、config.CAUSAL_METHOD='XLearner'（默认）
- 切换回 TLearner：config.CAUSAL_METHOD='TLearner'

**2026-03-16 移除 90% CI**
- 因果相关代码去除 90% CI：charls_recalculate_causal_impact（删除 ate_interval alpha=0.10、日志、ATE_CI_summary 写入）、generate_table_s4_full（删除 90% CI 列与 calc_90_ci）、clhls_full_pipeline 注释

**2026-03-16 图表与代码英文化**
- 要求：所有分析中涉及图表及代码输出统一使用英文
- 已修改：visualize_causal_forest_concrete（轴标签、标题、图例）、run_multi_method_causal（Exercise 标签）、run_all_physio_causal（EXPOSURES 英文标签、cut_label）、test_grip_walking_causal（exposures、cut_label）、charls_imputation_audit（Figure S3 Method_CN）、charls_shap_analysis（验证报告）、run_all_interventions_analysis（logger 输出）、viz/charls_extra_figures（docstring）
- config.INTERVENTION_LABELS_EN 已用于 run_sensitivity_scenarios、run_all_interventions_analysis 的图表

**2026-03-18 亚组分析 BUG 修复（致命）**
- 现象：06_subgroup 始终未生成，subgroup_analysis_results.csv 不存在
- 原因：run_axis_protocol 用 df_sub 检查 causal_impact，但 causal_impact 在 res_df（estimate_causal 返回值）中，df_sub 无此列
- 修正：用 res_df 作为 run_subgroup_analysis 的输入；causal_col 从 df_for_subgroup（res_df 或 df_sub）获取

**2026-03-18 三队列输出统一**
- run_subgroup_and_joint_causal：原仅 B/C，现扩展为 A/B/C 三队列；Axis_* 改为 Cohort_*

**2026-03-18 路径与异常处理修正**
- visualize_causal_forest_concrete.py：Axis_* 改为 config.AXIS_*_DIR（Cohort_*）
- charls_external_validation __main__：Axis_{label} 改为 axis_dirs[label]/04b_external_validation
- archive/check_data_quality.py：裸 except 改为 except Exception

**2026-03-18 Cohort_B 无 SHAP 图（TreeExplainer 模型列表不完整）**
- 现象：Cohort_B_Depression_to_Comorbidity 下无 02_shap 图
- 原因：charls_shap_analysis 仅对 RandomForest/XGB/LGBM/CatBoost 使用 TreeExplainer；若 B 冠军为 ExtraTrees/GBDT/HistGBM/DT，会降级至 KernelExplainer，易失败或超时
- 修正：扩展 TreeExplainer 列表，加入 ExtraTrees、GradientBoosting、HistGradientBoosting、DecisionTree（charls_shap_analysis、charls_shap_stratified 均已更新）
- 增强：SHAP 失败时 logger.error 增加 exc_info=True 便于排查

**2026-03-16 subgroup_analysis_results.csv 不存在**
- 现象：`Cohort_A_Healthy_Prospective\06_subgroup\subgroup_analysis_results.csv` 等文件不存在
- 原因：`06_subgroup` 由主流程 `run_all_charls_analyses.py` 的 `run_subgroup_analysis` 生成，需存在 `causal_impact_*` 列
- 解决：在项目根目录运行 `python run_all_charls_analyses.py` 生成
- 备选：`subgroup_and_joint_causal\subgroup_ate_exercise.csv`（三队列，格式 ate/ate_lb/ate_ub）
- 已更新：PAPER_绝对路径索引.md、PAPER_附录全部绝对路径.md 增加说明与备选路径

**2026-03-16 终极审查与全量修复**
- 时间划分：config.USE_TEMPORAL_SPLIT，compare_models 支持 train=wave<max
- Overlap 修剪：TLearner fit 前自动 trim PS 超出 [0.05,0.95] 且 >10%
- SMD 加权：check_balance_smd 新增 ps_weights 参数
- 选择偏倚：exercise×adlab_c 交互项（T=exercise 时）
- 详见 archive/docs/CODE_REVIEW_ULTIMATE_FINAL_2026-03-16.md

**2026-03-16 STAR 生物统计审查与 P0/P1 修复**
- E-value 保守端：保护效应用 ate_ub，有害效应用 ate_lb（VanderWeele & Ding 2017）
- income_total 预处理：移除全量中位数插补，留待 Pipeline Imputer 在训练折内 fit，避免泄露
- 插补敏感性：GroupShuffleSplit 按 ID 分组划分，替代 train_test_split
- 详见 archive/docs/CODE_REVIEW_STAR_BIOSTAT_2026-03-16.md

**2026-03-16 全项目 95% CI 格式统一**
- 统一格式：95% CI (lower, upper)，四位小数
- 已修改：charls_ci_utils（连字符→逗号）、charls_cpm_evaluation._fmt_ci、charls_imputation_audit 柱顶标注、charls_low_sample_optimization（CrI→CI）、archive/generate_table_s6_full、archive/generate_table_s4_full

**2026-03-17 变量移除（与论文基线表一致）**
- 已移除：smokev（是否吸烟）、rgrip、grip_strength_avg、psyche
- 预处理：charls_complete_preprocessing 中 chronic_cols 不含 psyche，不生成 grip_strength_avg，semantic_features 不含上述四列
- 加载插补数据时：config.COLS_TO_DROP 在 run_all_charls_analyses、run_interventions_linear_tlearner、run_all_physio_causal、run_multi_method_causal 中 drop
- 干预分析：INTERVENTIONS 由 7 个减为 6 个（移除 smokev）
- 基线表：charls_table1_stats 移除 rgrip、psyche、grip_col、smokev
- 2026-03：archive/charls_imputation_npj_style 同步更新：VARS_CONTINUOUS 移除 rgrip/grip_strength_avg，VARS_CATEGORICAL 移除 psyche/smokev，CORE_INTERVENTION_VARS 改为 exercise/drinkev，Figure S1 缺失热图将基于最新变量

**2026-03-16 逻辑/泄露/保存审计**
- 数据泄露：compare_models Pipeline 合规；敏感性分析用训练子集；因果 DML 全量 fit 符合惯例
- 保存：CHARLS_final_preprocessed.csv encoding 改为 utf-8-sig
- 临床评价：docstring 注明 DCA/校准基于传入 df 全量（略偏乐观），主 AUC 来自测试集
- 详见 archive/docs/CODE_AUDIT_LOGIC_LEAKAGE_SAVE_2026-03-16.md

**2026-03-16 SHAP 交互 "All dimensions of input must be of equal length"**
- run_shap_interaction：清理列名后可能产生重复，导致 summary_plot 报错；增加列名唯一化 + 维度校验，不匹配时跳过绘图但保留 CSV

**2026-03-16 分层 SHAP 索引错误**
- charls_shap_stratified：循环内 `idx = np.where(m)[0]` 覆盖了子图索引 idx，导致 `axes[idx]` 用行索引访问 axes 越界
- 修正：循环变量改为 plot_idx，行索引改为 row_idx

**2026-03-16 全量代码检查与修正**
- 导入路径：charls_policy_forest_plot、charls_did_analysis、charls_cate_visualization、charls_ablation_study、charls_visual_enhancement、charls_case_study 的 `from charls_feature_lists` 改为 `from utils.charls_feature_lists`
- 裸 except：charls_policy_forest_plot 的 ate/ate_interval 兼容性获取改为 `except Exception`；charls_subgroup_analysis 的 float 解析改为 `except (ValueError, IndexError)`
- 静默异常：charls_causal_methods_comparison、charls_clinical_evaluation、charls_clinical_decision_support 的 `except Exception: pass` 改为 `except Exception as ex: logger.debug(...)`
- 硬编码路径：config 新增 RAW_DATA_PATH='CHARLS.csv'，主流程与 run_sensitivity_scenarios 统一引用
- 详见 archive/docs/CODE_FULL_AUDIT_2026-03-16.md

**2026-03-16 性能优化（三处）**
- 轴线并行：config.PARALLEL_AXES=True 时 A/B/C 三轴线 joblib 并行
- 插补敏感性：config.PARALLEL_IMPUTATION_BOOTSTRAP=True 时 Bootstrap 循环并行
- 预处理复用：使用插补数据时 preprocess(write_output=True) 一次，attrition + df_for_imp_sens 共用，避免重复计算

**2026-03-16 每次运行全量更新**
- 移除 Table1、插补敏感性、流失图的跳过逻辑，所有表和图每次运行均更新
- 使用插补数据时单独运行 preprocess(write_output=True) 以生成 attrition_flow
- 移除 FORCE_REBUILD_TABLE1 配置

**2026-03-16 四项潜在问题修复**
- 流失流程图：USE_IMPUTED_DATA 时写入 attrition_flow_readme.txt 说明与当前队列差异
- 剂量反应：写入 dose_response_readme.txt 注明 sleep 仅插补数据有、预处理无
- run_ite_validation_for_axes：output_root 默认 '.' 与主流程一致，并 fallback 到 OUTPUT_ROOT

**2026-03-16 分析代码检查与修正**
- 插补敏感性：preprocess 返回 None 时保持 df_clean，打 warning
- 因果 DML：Y 缺失时 dropna(subset=[Y]) 并打日志
- 详见 archive/docs/CODE_ANALYSIS_AUDIT_2026-03-16.md

**2026-03-16 代码自查与修正**
- 插补敏感性：主分析用插补数据时，单独 preprocess 得带缺失 df 再传入，避免五种方法比较无差异
- 临床决策支持：反事实「改善睡眠」改为 sleep_adequate=1（与预处理一致，sleep 已移除）
- to_csv：主流程相关脚本补 encoding='utf-8-sig' 防 Windows 中文乱码
- 详见 archive/docs/CODE_SELF_AUDIT_2026-03.md

**2026-03-16 分类变量标准化防护**
- charls_feature_lists：新增 CATEGORICAL_NO_SCALE（edu/gender/rural/marry 等），CONTINUOUS_FOR_SCALING 仅含真正连续变量；运行时 assert 防止误将分类变量加入缩放列表

**2026-03-16 剂量反应 sleep 缺失处理**
- charls_dose_response.py：sleep 缺失不再 fillna(0) 归入 <5h；改为 dropna 排除 + 单独标记 "Missing (excluded)" 写入 dose_response_summary.csv，避免扭曲剂量反应

**2026-03-16 方法学深度加固（6 项）**
- 因果时序：preprocessing 显式检查 wave+1，不符时预警；注释滞后变量用法
- DML：min_samples_leaf=15；ITE 分布直方图 fig_ite_distribution_*.png
- E-value：基于 ATE 与 95% CI 下限计算 point/conservative
- 剂量反应：4 节点 RCS（sleep）；exercise 二分类→定序+趋势检验
- 亚组：Sample_Size_Warning 列（n_events<30 标记 Caution: Underpowered）
- 校准曲线：2x3 布局，下排增加预测概率分布直方图

**2026-03-16 方法学风险点落地**
- P0 因果时序：preprocessing、charls_recalculate_causal_impact docstring 明确 Wave(t) vs Wave(t+1)
- 剂量反应：exercise 二分类时用定序分析，非 RCS
- 重叠假设：PS 超出 [0.05,0.95] 的 trimming 比例写入 ATE_CI_summary
- 方法学风险点与改进回应.md 汇总回应

**2026-03-16 因果 DML nuisance 模型改为 RF**
- charls_recalculate_causal_impact: CatBoostRegressor/CatBoostClassifier 改为 RandomForestRegressor/RandomForestClassifier
- 移除 train_dir 与 finally 清理逻辑；cleanup_temp_cat_dirs 保留用于清理历史 CatBoost 临时目录

**2026-03-16 文档归档**
- 审稿回复、代码审计报告、草稿模板等 28 个 .md 移至 archive/docs/

**2026-03-16 文件夹管理代码**
- 创建 data/modeling/causal/evaluation/interpretability/viz/scripts/external/utils 目录
- 按功能移动模块并更新所有导入路径；主入口 run_all_charls_analyses.py 保持不变

**2026-03-18 Table 7 一致性评价**
- charls_causal_methods_comparison.run_all_axes_comparison：新增 _add_consistency_column，为 causal_methods_comparison_summary.csv 增加 Consistency 列
- 取值：Consistent（方向一致且 95% CI 重叠）、Direction_only（方向一致但 CI 不重叠）、Inconsistent（方向不一致）

**2026-03-18 代码精简（冗余与归档）**
- 删除 run_forest_plot_only.py（主流程 run_all_interventions 已生成森林图）
- 归档至 archive/scripts/：run_interventions_linear_tlearner、run_multi_method_causal、run_xlearner_causal_only、run_subgroup_and_joint_causal、test_grip_walking_causal（主流程已有替代）
- 归档脚本需从项目根目录运行：python archive/scripts/run_xxx.py；已修正 sys.path（多一层 dirname）

**2026-03-16 代码精简**
- 删除 9 个一次性脚本：compare_threshold, eval_single_model_mvp, fix_fig_s1, final_fix_fig_s1, fix_ci_axis_causal, emergency_replot, patch_missing_plots, run_final_completion, charls_evalue_plot
- 删除 main.py、generate_table2.py
- 创建 archive/ 归档：实验性、审计、表格/图表整理脚本（含 README 说明运行方式）

**2026-03-16 方法学加固（TRIPOD + 因果防守）**
- charls_model_comparison: pass_cols 的 Imputer 也封装进 Pipeline；docstring 增加论文表述
- charls_recalculate_causal_impact: 倾向评分重叠图（fig_propensity_overlap_*.png）+ E-value 写入 ATE_CI_summary

**2026-03-16 外部 AI 审稿意见落地（P0/P1）**
- **P0-1 数据泄露**：charls_model_comparison 已正确将 Imputer/Scaler 封装在 Pipeline 内，仅 CV 训练折 fit；已补充注释说明。
- **P0-2 IPW 权重修剪**：charls_causal_methods_comparison._ate_psw 增加 1–99 分位数 + [0.1,50] 兜底，对应 Sturmer et al. 2020。
- **P1-1 阈值选择**：charls_cpm_evaluation 改为在验证集（训练集 80/20 再分）上确定最优阈值，evaluate_single_model 新增 opt_threshold 参数。
- **P1-2 分层 Bootstrap**：charls_ci_utils、charls_cpm_evaluation 的 Bootstrap 改为 Stratified，n_bootstraps 提升至 1000。

---

**Mistake: [Fatal Error] Undefined variable `group_ids` in `charls_recalculate_causal_impact.py`**
**Wrong**:
```python
dml_causal_forest.fit(Y=Y_series, T=T_series, X=X_scaled, W=None, groups=group_ids)
```
**Correct**:
```python
dml_causal_forest.fit(Y=Y_series, T=T_series, X=X_scaled, W=None, groups=df['ID'])
```

**Mistake: [Logical Inconsistency] Axis naming vs. Task in Phase 1**
**Wrong**:
In `run_all_charls_analyses.py`, Phase 1 labels the tasks as "Depression Axis" and "CI Axis", which implies predicting the state itself. However, `compare_models` is hardcoded to predict `is_comorbidity_next`.
**Correct**:
Clarify that Phase 1 is predicting *future comorbidity* within the Depression-only and CI-only subgroups. Alternatively, make `compare_models` accept a target column parameter.

**Mistake: [Pre-processing/Causal Inference] Standardizing Label-Encoded Categorical Features**
**Wrong**:
Label-encoding categorical variables in `preprocess_charls_data` and then applying `StandardScaler` in `estimate_causal_impact`.
**Correct**:
Categorical variables should be handled via One-Hot encoding or kept as-is for models that support them (like CatBoost), and should not be standardized if they are categorical labels.

---

**Mistake: [Leakage Risk] Typo `memeory` in leakage_keywords**
**Wrong**:
```python
leakage_keywords = ['cesd', 'total_cog', 'cognition', 'memeory', 'executive', ...]
```
**Correct**:
```python
leakage_keywords = ['cesd', 'total_cog', 'cognition', 'memory', 'executive', ...]
```
Column names containing "memory" (e.g. cognitive/memory tests) would not be excluded with "memeory", causing potential data leakage. Fix in all modules: charls_model_comparison, charls_recalculate_causal_impact, charls_clinical_evaluation, charls_clinical_decision_support, charls_ablation_study, charls_methodology_audit, charls_cate_visualization, charls_sensitivity_analysis, charls_validation_suite, charls_visual_enhancement, replot_main_figures.

---

**Mistake: [Logical Inconsistency] Clinical decision support counterfactual vs. pipeline treatment**
**Wrong**:
In `charls_clinical_decision_support.py`, counterfactual uses `is_depression` / `is_cognitive_impairment` as "treatment" and sets them to 0; with `target_col='is_comorbidity_next'` the code always picks `is_cognitive_impairment`. The rest of the pipeline uses `exercise` as the causal treatment.
**Correct**:
Align counterfactual with intervention: either simulate "increase exercise" / "improve sleep" scenarios, or explicitly choose which variable to manipulate by axis (e.g. baseline_group or axis label) so B/C axes are semantically consistent.

---

**Mistake: [Robustness] confusion_matrix().ravel() in get_metrics_with_ci when only one class present**
**Wrong**:
```python
tn, fp, fn, tp = confusion_matrix(y_t, (y_p > 0.5).astype(int)).ravel()
```
If a bootstrap sample has predictions with only one class, the confusion matrix may not be 2×2 and ravel() yields fewer than 4 elements → ValueError.
**Correct**:
Check that both y_t and y_p have two unique values (or that confusion_matrix shape is 2×2) before unpacking; otherwise return a default (e.g. 0 or np.nan) and handle in the bootstrap loop.

---

**Mistake: [Robustness] Bootstrap CI when too few valid iterations**
**Wrong**:
In `charls_ci_utils.get_metrics_with_ci`, when many bootstrap iterations are skipped (e.g. `len(np.unique(y_true_b)) < 2`), `bootstrapped_stats[k]` can be empty or very short; `np.percentile(bootstrapped_stats[k], 2.5)` then fails or is unreliable.
**Correct**:
Before computing percentiles, check `len(bootstrapped_stats[k])`; if below a threshold (e.g. 10), use the point estimate as both lower and upper CI to avoid crash and document in code comment.

---

**Mistake: [Robustness] Table1 categorical levels .astype(int) on non-integer-like values**
**Wrong**:
In `charls_table1_stats.py`, `levels = sorted(df[col].dropna().unique().astype(int).tolist())` can raise ValueError/TypeError if the column has non-numeric or float-with-NaN semantics.
**Correct**:
Wrap in try/except and skip that categorical variable if conversion fails; use `pd.Series(...).astype(int)` for consistent handling.

---

**2025-03-02 代码审查总结**
完成 CHARLS 因果机器学习项目全面代码审查，输出 `CODE_REVIEW_REPORT.md`。主要发现：
- 高：预处理缺少 cesd10/认知列时第 64 行 KeyError；因果失败时调用方逻辑可更清晰；硬编码路径
- 中：JSON 写入缺 encoding；attrition 数值转换需防护；冗余逻辑
- 低：未使用 import；magic number；异常捕获过宽
- 综合评分 7.4/10，建议配置集中化、预处理健壮性、异常分层、统一 UTF-8 编码、增加单元测试

---

**2026-03-02 论文实验部分全面更新（以实际输出为准）**
- 数据来源：attrition_flow、table1_baseline、model_performance、causal_methods、sensitivity_summary、external_validation_summary、bias_sensitivity、ite_stratified_validation、subgroup_analysis_results
- 附录S2表6：按Axis_*/04b_external_validation/external_validation_summary.csv 更新（B区域0.679、C区域0.604等）
- 附录S4 ITE分层：按ite_stratified_validation.csv 更新（低ITE 1.0%/29.9%、高ITE 23.4%/2.7%）
- 3.7亚组、附录S3偏倚：按subgroup_analysis_results、bias_sensitivity 更新

---

**2026-03-05 审稿意见 P0 落地**
- 校准斜率+Brier 分解：`charls_clinical_evaluation.py` 已输出 calibration_brier_report.txt（含 Calibration slope、Brier、Reliability、Resolution）
- 内部时间/区域验证：新增 `charls_external_validation.py`，按 wave 划分时间验证、按东中西部划分区域验证，输出 AUC/AUPRC/Brier 及校准曲线；已集成至 `run_all_charls_analyses.py` 的 run_axis_protocol

---

**2026-03-02 论文按实验输出全面更新（非记忆文件）**
- 数据来源：Axis_*/01_prediction/model_performance_full_*.csv、table1_baseline_characteristics.csv、causal_methods_comparison_summary.csv
- 预测冠军：A MLP 0.7484，B MLP 0.7095，C NB 0.6446；表2 规律运动 53.2%/52.1%/49.8%；附录S1 完整15种模型
- 已更新：PAPER_完整版_审稿修订.md、PAPER_完整版_双语版.md（摘要、1.3、表3、3.3–3.10、4.1、5.1、图注、附录S1/S3/S6）

---

**2026-03-02 论文根据当前结果更新**
- 表1流失步骤名与 attrition_flow.csv 对齐；2.4 增加干预缺失按0处理说明
- 3.5/3.9/4.1/附录B：轴线B运动 DML -0.037→-0.034，轴线C运动 0.024→0.032；表7 DML CI (-0.103,0.030)→(-0.094,0.028)
- 已更新 PAPER_完整版_双语版.md、PAPER_完整版_审稿修订.md

---

**2026-03-02 CHARLS 代码系统性审计**
完成按「数据处理→预测建模→因果推断→敏感性分析→规范性」五模块的全面审计，输出 `LIU_JUE_STRATEGIC_SUMMARY/CHARLS代码系统性审计报告.md`。
- 致命/高危：0
- 低危：4（已全部修正：exercise.fillna(0)、attrition 步骤名、偏倚方法说明、RANDOM_SEED 统一）
- 整体可用性评分：8.5/10
- 核心逻辑（队列筛选、变量定义、GroupKFold、DML/PSM/PSW、截断值敏感性）均正确

---

**2026-03-10 全量分析代码复查**
- **已修正**：`charls_imputation_npj_style.py` 入口 `__main__` 仍检查已删除的 `result['summary']`，导致汇总永不打印 → 改为打印 `best_method`、`best_nrmse`、溯源路径
- **已修正**：主流程关键脚本 `to_csv` 缺 encoding，Windows 下中文可能乱码 → 为 charls_external_validation、charls_shap_stratified、charls_ite_validation、charls_temporal_analysis 的 to_csv 增加 `encoding='utf-8-sig'`
- **已确认无问题**：confusion_matrix ravel 有 2×2 检查；np.percentile 有 len<10 防护；table1 categorical astype(int) 有 try/except；预处理 cesd/cog 列用 next() 查找；JSON 写入有 encoding
- **设计说明**：主流程 `run_all_charls_analyses` 使用 `preprocess_charls_data` 直接输出，未加载 `charls_imputation_npj_style` 的插补结果；若需用插补数据做主分析，需在 main 中改为加载 `imputation_npj_results/pipeline_trace/step1_imputed_full.csv`
- **2026-03-15 已实现**：当 `USE_IMPUTED_DATA=True` 且 `IMPUTED_DATA_PATH` 存在时，主流程加载插补数据、`reapply_cohort_definition` 得主队列、`run_sensitivity_scenarios_analysis` 传入 `df_base=df_imputed`，全流程基于插补后数据。见 `run_all_charls_analyses.py`。

---

**2026-03-15 审稿关键修正（CRITICAL_FIXES_REVIEWER_RESPONSE.md）**
1. **数据泄露**：run_sensitivity_scenarios 默认 train_only=True，仅用 80% 训练集，与 compare_models 划分一致。
2. **多重检验**：compare_models 输出 multiple_testing_note.txt，说明比较次数与 FDR/Bonferroni 建议。
3. **亚组效能**：charls_subgroup_analysis 要求 n_events≥30 且 n_total>30 才报告亚组，输出 N_events 列。
4. **Causal Forest**：charls_recalculate_causal_impact 增加 honest=True。
5. **config**：新增 ANALYSIS_LOCK=True。

---

**2026-03-10 插补脚本全面代码审查（imputation_code_review_report.md）**
- 整体结论：符合研究设计，核心逻辑正确（先全量插补→后划分队列）
- 已修复：1) 核心干预变量完整病例子集 step1b_complete_case_core.csv（附录 S2）；2) MCAR 协变量 aux 不足时跳过；3) 队列样本量偏差>100 时 logger.warning；4) 分布图优先包含 bmi/sleep/exercise

---

**Mistake: [LossySetitemError] 插补结果 float 赋给 int64 列**
**Wrong**:
```python
df_long_out.loc[idx, v] = val  # val 为 float，目标列为 int64 时触发 LossySetitemError
out[c] = X_imp[:, i]           # _safe_assign_imputed 同理
out[cols] = imp.fit_transform(df[cols])  # impute_mean/median/mode 同理
```
**Correct**:
赋值前检查目标列 dtype，若为整数则 `np.round(vals).astype(np.int64)` 再赋值。
**已修复位置**：`impute_wide_and_map_back`、`_safe_assign_imputed`、`impute_mean`、`impute_median`、`impute_ordinal`、`impute_mode`。

---

**2026-03-10 敏感性验证核查（table4 大差异）**
- **原因**：Original_mean = 插补前「有观测」均值，Imputed_mean = 插补后「非缺失」均值；宽表部分列被跳过导致插补后仍缺失，两均值基于不同样本。
- **已做**：1) sensitivity_validation 增加 N_observed、N_valid_after_imputation、N_still_missing、Mean_of_imputed_cells_only 及 Flag(OK/remaining_NaN/moderate_shift/large_shift)；2) 输出 table4_diagnostics.csv 与 table4_sensitivity_readme.txt；3) 宽表回填后对仍含缺失的连续列用长表再插补一次，减少 N_still_missing，便于可比；4) table4 主表增加 N_* 列。

---

**2026-03-11 插补后的敏感性分析逻辑闭环**
- **问题**：主分析使用插补数据时，`run_imputation_sensitivity_preprocessed(df_clean)` 收到的是已插补数据，X 无缺失，五种插补方法比较无差异。
- **正确**：插补敏感性分析必须使用「带缺失」的预处理数据。主流程已改：当 `USE_IMPUTED_DATA` 且插补文件存在时，单独调用 `preprocess_charls_data` 得到 `df_for_imputation_sensitivity` 再传入；否则传 `df_clean`。
- **charls_imputation_audit.py**：1) 入口处 `df.dropna(subset=[target_col])` 避免 y 含 NaN；2) `_add_result` 过滤 nan、len(auc_arr)<10 时 CI 用点估计；3) `_run_single_imputation_auc` 单类/异常返回 np.nan；4) 空 results 与绘图 NaN 防护。详见 `SENSITIVITY_IMPUTATION_AUDIT_REPORT.md`。

---

**Mistake: [Robustness] Specificity else-branch in charls_cpm_evaluation when cm non-2×2**
**Wrong**:
```python
specificity_val = 0.0 if (y_true == 1).any() else 1.0  # 仅看 y_true，忽略 y_pred
```
当 y_true 含两类、y_pred 全为 0 时，spec 应为 TN/(TN+FP)=1.0，但原逻辑因 (y_true==1).any() 为 True 错误返回 0.0。
**Correct**:
```python
if (y_true == 0).sum() == 0:
    specificity_val = 0.0  # 无负类
elif (y_pred_opt == 1).all():
    specificity_val = 0.0   # 全预测 1，TN=0
else:
    specificity_val = 1.0   # 全预测 0
```
**已修复位置**：`charls_cpm_evaluation.py` 的 `evaluate_single_model` 与 `_bootstrap_ci_at_threshold`。

---

**2026-03-11 全部分析代码逻辑检查**
- **范围**：run_all_charls_analyses 调用的预处理、表1、打擂、SHAP、因果、敏感性、插补敏感性、多暴露、扩展干预、因果方法比较、低样本优化、各类绘图与 consolidate。
- **已修正**：1) 轴线路径与 config 统一：run_all 复制外部验证改用 AXIS_*_DIR；draw_roc_combined 使用 OUTPUT_ROOT、AXIS_*_DIR；charls_extra_figures.draw_combined_subgroup_cate 使用 AXIS_*_DIR。2) 其余模块参数传递、空值/边界防护与 self.md 记录一致。
- **约定**：因果模块 Y 硬编码为 is_comorbidity_next；若改 TARGET_COL 须同步改 charls_recalculate_causal_impact。详见 `ANALYSIS_CODE_LOGIC_REPORT.md`。
- **2025-03 逻辑与衔接复核**：主流程、敏感性（插补+截断值+完整病例）、因果（95%+90% CI）、亚组、临床评价、路径与 config 一致性已核查；未发现新逻辑错误或衔接断点。90% CI 为增量输出，不替换 95% CI。详见 `ANALYSIS_LOGIC_AUDIT_2025.md`。

---

**2026-03-16 投稿稿补充：Text S3（多重检验/FDR）+ Figure 6 综合图注**
- **所做**：在 `PAPER_Manuscript_Submission_Ready.md`、`PAPER_完整版_2026-03-20.md` 附件目录新增 **Supplementary Text S3**（近似 *p*、Bonferroni、FDR 列、`multiplicity_correction.py`、主推断仍为 bootstrap *P*/CI）；§2.10 交叉引用改为 **Supplementary Text S3**；Figure 6 图注写明 **DCA + calibration + PR** 综合面板并指回 S5 单独校准；投稿 checklist 改为 Text **S1–S3**。
- **`PAPER_Supplementary_Materials_Detailed.md`**：目录在 Table S5 后接 Text S3 摘要与同路径交叉引用（已取消「14 算法」补充表 S6）。
- **`PAPER_npj_style_polished.md`**：Methods 中临床效用一句与 Figure 6 综合面板对齐。
- **`PAPER_Multiplicity_FDR_Code_Alignment.md`**：增加「英文补充材料定稿位置」三文件指针。
- **注意**：`project.md` 在工作区未找到，仅更新 `self.md`。

---

**2026-03-16 CPM Table2：Precision / F1 / Accuracy 的 Bootstrap 95% CI**
- **`modeling/charls_cpm_evaluation.py`**：`_bootstrap_ci_at_threshold` 已在分层 bootstrap 中收集 Precision、F1、Accuracy；补全 `evaluate_single_model` 的 `pe` 传入三项点估计；返回并写入 CSV 列 **`Precision_95CI`、`F1_95CI`、`Accuracy_95CI`**；`evaluate_and_report` 的 `df_main` 同步输出。**重新跑预测/CPM 步后** `table2_*_main_performance.csv` 与 `results/tables/table2_prediction_axis*.csv` 才会更新。

---

**2025-03-16 主文删除 ROC 图后的稿件同步**
- **用户操作**：主文不再放三队列合并 ROC；判别与全模型表以主文 **Table 2** 为准，**不再设**「14 算法」**Supplementary Table S6**。
- **主文图号**：Figure 1–2 不变；原 ROC 后的图顺延为 **Figure 3（ATE/亚组）**、**Figure 4（SHAP）**、**Figure 5（DCA/校准/PR）**。
- **已对齐文件**：`PAPER_Manuscript_Submission_Ready.md`、`PAPER_完整版_2026-03-20.md`、`PAPER_npj_style_polished.md`、`PAPER_npj_style_polished_Full.md`（SHAP→Fig4、DCA→Fig5）、`PAPER_Main_Figures_Mapping.md`、`PAPER_DRAFT_FINAL_SUBMISSION.md`、`PAPER_DRAFT_完整版.md`、`PAPER_DRAFT_2026-03-20_基于最新运行结果.md`、`PAPER_框架与图表索引_代码逻辑版.md`。
- **流水线**：`run_all_charls_analyses.py` 仍可能生成 `fig3_roc_combined.png`，仅作可选补充材料。
- **`project.md`**：工作区仍无该文件。

---

**2025-03-16 取消补充表 S6（14 算法全表）**
- **原因**：用户表示全模型性能已在主文 Table 2 体现，补充材料中不再重复 **Table S6**。
- **已改**：`PAPER_Manuscript_Submission_Ready.md`、`PAPER_完整版_2026-03-20.md`（删附录 S6 大块表格与目录项）、`PAPER_npj_style_polished.md`、`PAPER_npj_style_polished_Full.md`、`PAPER_DRAFT_2026-03-20_基于最新运行结果.md`、`PAPER_Main_Figures_Mapping.md`、`PAPER_Supplementary_Materials_Detailed.md`、`PAPER_框架与图表索引_代码逻辑版.md`；投稿 checklist 改为 Tables **S1–S5**。
- **数据文件**：`results/tables/table2_prediction_axis*.csv` 仍可作 reproducibility，不必称为「Table S6」。

---

**2025-03-16 图表核对清单**
- 新增 **`PAPER_图表文件核对清单.md`**：按 `PAPER_Manuscript_Submission_Ready.md` 逐项列出主文 Table 1–5、Figure 1–5、补充 Table S1–S5、Figure S1–S5 的 `results/` 与 `Cohort_*` 路径，并注明 Figure S1/S3/S4 等与主流程可能不一致处（Love plot、缺失热图、figS3 文件名）。

---

**2025-03-16 投稿稿与 `results/tables` 对齐（导师版）**
- **`PAPER_Manuscript_Submission_Ready.md`**：增加 **Data lock-in** 段（权威 CSV/路径）；修正与 `table4_ate_summary.csv` 不一致的叙述（原写「其余干预均不显著」——饮酒与正常 BMI 在 Cohort A 的 95% CI 不含 0）；E-value conservative **1.26**；亚组 CATE 范围与 `table3_subgroup_axisB.csv` 一致（含 SRH_1 小样本）；补充 **Table S2** 脚注与 `table5_sensitivity_summary.csv` 一致；**Table S5** 脚注说明 XLearner 主分析 CI 以 `table4` 为准、与 `table7` 中 XLearner 宽 CI 并存；摘要 §3.4 *P* 改为 bootstrap *P* ≈ 0.018。
- **`PAPER_完整版_2026-03-20.md`**：同步上述所有数字与脚注（摘要、§3.4–3.9、Table 4、Table 5、Table S2、Table S5）。

---

**2026-03-25 `run_cohort_protocol` ATE 三元组不完整时的清洗**
- 若 `res_df is not None` 但 **ATE / lb / ub 任一非有限**：**`logger.warning`** 标明缺失项；**删除** `causal_impact_{TREATMENT_COL}`；**汇总三元组置 NaN**；下游 **`run_sensitivity_analysis`**（无因果列即返回）、**ITE/列线图**（无列不进）、**亚组**（无 `causal_impact_*` 则跳过）均不崩溃。

---

**2026-03-25 因果失败返回值与 bootstrap / readme 强化**
- **`CAUSAL_FAILURE_ATE_TRIPLET = (np.nan, np.nan, np.nan)`**：DML / TLearner / XLearner 所有失败与早退路径统一返回，**不再** `(0,0,0)`；`get_estimate_causal_impact` 文档已更新。
- **TLearner / XLearner bootstrap**：移除静默 `pass`；统计 **成功/失败/idx 过短**；**`logger.info` + `print(..., flush=True)`** 终端透明；XLearner 在 **`ate_interval` 成功**时单独打印说明未跑 cluster bootstrap。
- **`run_all_charls_analyses`**：`intervention_benefit_comparison` **try/except**；柱高 **NaN→0** + **NA** 标注；errorbar 仍仅有限值时绘制。
- **`sensitivity_analysis_readme.txt`**：重写为 **A/B 两段 + 中文摘要**，显式引用 **`config.USE_TEMPORAL_SPLIT`、`config.RANDOM_SEED`、GroupShuffleSplit 0.2、`_get_train_subset(df_imputed)`** 与 **`compare_models` 时间外推** 的差异。

---

**2026-03-25 代码/稿件一致性修补（自查后落地）**
- **`run_cohort_protocol`**：因果 `res_df is None` 时 **ATE 三元组改为 `np.nan`**，避免汇总图误读为 0；**空队列**续跑同理；**`intervention_benefit_comparison.png`** 对非有限 ATE/CI **跳过 errorbar**。
- **`_load_skipped_cohort_metrics`**：解析不到 `ATE_CI_summary` 时 **NaN** 而非 0。
- **`charls_recalculate_causal_impact` XLearner bootstrap**：统计失败次数并 **`logger.warning`**，替代静默 `pass`。
- **`run_sensitivity_scenarios.py`**：重写 **`sensitivity_analysis_readme.txt`**，区分 **ID 80% 截断敏感性基表** vs **`USE_TEMPORAL_SPLIT` 下 CPM 末波测试**。
- **`PAPER_Manuscript_Submission_Ready.md`**：§2.4 **时间外推 + 内层 GroupKFold**、**Platt 校准**、**Rubin 补充说明**、**exercise fillna(0)**、摘要/结论 **2–3 年波距**、主表数量 **6（含 Table 1b）**、§3.8 / Table S5 **XLearner B 与 −0.021 对齐为 −2.1%**。
- **`docs/全流程数据分析详细说明.md`**：步骤表补充 **sensitivity readme** 说明。

---

**2026-03-25 Table 1b 发病密度（人年）与旧脚本区分**
- **主分析表**：`data/charls_incidence_density.py` — 对主分析 `df_clean` 的 **每一 person-wave 行** 按波次赋间隔（wave 1/2→2.0 y，wave 3→3.0 y，wave 4→0），`is_comorbidity_next==1` 用 **中点法** `interval/2`，否则全区间；按 `baseline_group` + **Total** 汇总；`run_all_charls_analyses.py` 在 `OUTPUT_ROOT` 与 `results/tables/` 写出 **`table1b_incidence_density.csv`**。
- **勿混淆**：`scripts/compute_incidence_density_person_time.py` 为 **另一套** 长表/每人首条 person-wave 逻辑，输出 **`table_incidence_density_person_time.csv`**；投稿稿 §3.2 / Table 1b 以 **`table1b_incidence_density.csv`** 为准。

---

**2026-03-25 TRIPOD：CPM 冠军 / 外部验证 / EPV 对齐**
- **`CPM_MIN_RECALL_THRESHOLD`** + **`select_champion_from_perf_df`**：`compare_models` 与 **`_select_champion`** 共用规则；擂台表增加 **`Recall_at_opt_t_raw`**（测试集上 **内部 CV Youden 阈值** 下的 Recall），与 Table2 主 Recall 一致，**不再**用 `get_metrics_with_ci` 的 0.5 阈值 Recall 做筛选。
- **`compare_models`**：训练池划分后 **`[TRIPOD] EPV`** 日志（`n_events` / `n_predictors` / EPV）。
- **`charls_external_validation.py`**：废弃 SimpleImputer+ExtraTrees 重训；**`joblib.load(champion_model.joblib)`** 冻结预测；特征列按 **`feature_names_in_`** 对齐（支持 **`CalibratedClassifierCV`** 解包）；时间/地域子集仅 **`predict_proba`**。
- **`run_all_charls_analyses`**：`run_external_validation(..., champion_model_path=pred_dir/champion_model.joblib, model=best_model)` 兜底。

---

**2026-03-25 CPM 冠军门槛：禁止 0.5 阈值 Recall_raw**
- **`select_champion_from_perf_df`**：**仅** `Recall_at_opt_t_raw`（OOF Youden → 测试集 Recall）；**已移除** 对 `Recall_raw` / 解析 `Recall` 字符串 / 全放行 `np.ones` 等回退；缺列则 **ERROR** 日志并退回全表 AUC 最高。
- **`get_metrics_with_ci`**：docstring 标明 `Recall_raw` 为 0.5 阈值，**不得**用于冠军筛选。

---

**2026-03-31 Streamlit Cloud：`找不到文件: CHARLS.csv`**
- **现象**：云端无 `CHARLS.csv`、无插补 CSV 时，`preprocess_charls_data` 被调用 → `ERROR:data.charls_complete_preprocessing:找不到文件: CHARLS.csv`。
- **原因**：`load_df_for_analysis` 在缺文件时仍走预处理；路径相对 **cwd**，不如相对 **仓库根** 稳。
- **修正**：`utils/charls_script_data_loader.py` — `_repo_root()` / `_resolve_repo_path()`；插补与原始路径用 **`os.path.isfile`**；**仅当原始文件存在** 才调用 `preprocess_charls_data`；否则加载 **`data/sample_data.csv`**，走与插补分支相同的 `age` 过滤、`prepare_exposures`、`COLS_TO_DROP`、`reapply_cohort_definition`（**演示用**，非完整样本）。`load_supervised_prediction_df` 在 `USE_IMPUTED_DATA=True` 且原始 CHARLS 缺失时 **回退** `load_df_for_analysis`（含演示表），避免 CPM 路径单独报错。
- **XGBoost pickle UserWarning**：旧版序列化模型用新版 XGBoost 反序列化时的提示；若推理异常，需用训练版 **`Booster.save_model`** 导出再在云端加载，或对齐 **`xgboost` 版本**。

**2026-03-31 演示表：`Preprocessor transform failed: columns are missing`**
- **原因**：`data/sample_data.csv` 缺 **adlab_c, fall_down, pension, wspeed, ins, iadl, retire, disability**，冠军 Pipeline 的 ColumnTransformer 要求列名齐全。
- **修正**：`utils/charls_script_data_loader.py` 在读取演示 CSV 后 **`_pad_bundled_demo_columns`** 补列（二分类/计数默认 0，**wspeed** 演示占位 **1.0**）；**需重新 push** 后 Streamlit 拉代码生效。

**Mistake: SHAP / XGBoost `could not convert string to float: '[1.4061464E-1]'`（仅对部分列做清洗）**
**Wrong**:
- `_coerce_float_matrix_for_shap` 在 **`is_numeric_dtype`** 为真时直接 **`pd.to_numeric(ser)`**，不剥离 **`[...]`**；若列被标成 numeric 但单元格仍是带括号的字符串，或 object 列被误判分支遗漏，仍会报错。
**Correct**:
- **每一列**先 **`astype(str)`**，循环剥离外层 **`^\[(.*)\]$`**（最多 4 次防嵌套），再 **`pd.to_numeric(..., errors="coerce")`**，**`np.nan_to_num`** 写入 **`float64` ndarray**，重建 **`DataFrame`**；在 **`_build_explainer`** 入口对 **`X_bg`** 再调用一次 **`_coerce_float_matrix_for_shap`** 作兜底。
- **参考**：`streamlit_shap_three_cohorts.py` 中 **`_coerce_float_matrix_for_shap`**、**`_build_explainer`**。
