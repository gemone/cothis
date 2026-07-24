# 贡献指南 / PR 工作流程

本文件定义 **cothis 开发团队**（Multica squad `cothis 开发团队`）全员提交代码的统一流程。所有成员（人类与 agent）完成工作后，都必须按此链路把改动合入 `main`。

## 角色与职责

| 角色 | Multica agent | 职责 |
| --- | --- | --- |
| 提交者（Author） | 全体成员 | 完成工作后提交 Pull Request |
| **PR 代码审阅员** | `PR 代码审阅员` | 对 PR 做代码 review，关注正确性、缺陷、设计、测试、可读性、性能 |
| **工程架构师** | `工程架构师`（squad leader） | review 通过后，分配 **PR 合并审阅官** 检查合并前置条件 |
| **PR 合并审阅官** | `PR 合并审阅官` | 核对 review 意见清零、CI 通过、无冲突，满足后 squash-merge 并把关联 issue 流转到 done |

> 三个角色 agent 均为 `cothis 开发团队` squad 成员；任一 agent 被 `@` 提及或被指派到 issue 时会自行启动，无需额外触发。

## 工作流程（四步）

```
提交者开 PR  →  ① PR 代码审阅员 review  →  ② 工程架构师分派 PR 合并审阅官  →  ③ 合并审阅官检查并合并
```

1. **提交 PR（Author）**
   - 从 `main` 切出分支，命名遵循 Conventional Commits 前缀：`feat/`、`fix/`、`docs/`、`chore/`、`refactor/`、`test/`。
   - PR 标题用 Conventional Commits 格式（如 `feat(tools): add fs.search glob support`）。
   - PR 描述写清：改了什么、为什么、如何验证；如涉及重命名，附上 *post-rename docs scan* 结论（见 `AGENTS.md` § Renaming）。
   - 关联对应的 Multica issue（在描述里贴 issue 链接）。
   - 本地确认 `uv run ruff check && uv run ty check && uv run pytest -q` 通过（与 `.github/workflows/ci.yml` 一致）后再请求 review。

2. **代码 review（PR 代码审阅员）**
   - 逐文件审阅，对具体行号留精准评论（引用 `path:line`），区分「必须修改」与「建议」。
   - 给出结论：**通过 / 需要修改 / 需要讨论**。不擅自合并。
   - review 通过后，在 PR 或关联 issue 上 `@工程架构师`，提示进入下一步。

3. **分派合并检查（工程架构师）**
   - 收到 review 通过的信号后，`@PR 合并审阅官` 并指明要合并的 PR，由其核查合并前置条件。
   - 工程架构师不重复 review 代码质量，只做调度。

4. **合并（PR 合并审阅官）**
   - 核查：所有 review 线程已解决 / 无 `changes requested`；CI 全绿（`gh pr checks` 取一次快照，不轮询等待）；无合并冲突；已关联 Multica issue。
   - 全部满足 → `gh pr merge <pr> --squash` 合并，随后把关联 Multica issue 流转到 `done`，两步连续完成。
   - 任一不满足 → 不合并，列出具体阻塞项清单，issue 标为 `blocked`，指明下一步。
   - 合并策略默认 **squash**（与本仓库历史一致），除非 issue 明确要求 rebase / merge commit。

## 约束

- 不提交、不推送、不改写 `main` 历史，除非被 PR 合并审阅官在合并流程中执行。
- 不读取、不打印、不提交 `.env` 与凭据文件。
- 只改动任务所需范围，不做无关重构或重命名（遵循 `AGENTS.md` 的工程纪律）。
- draft PR、未关联 issue、或缺少必要测试的 PR 不进入合并流程。
