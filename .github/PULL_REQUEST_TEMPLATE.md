<!-- English first, 中文在下 -->

## What this adds

<!-- Skill name, one line on what it does, and why the official skills (or a combination of them) cannot do it. -->

## Checklist

- [ ] Only files under `contrib/skills/<Name>/` change
- [ ] `python scripts/skill_check.py contrib/skills/<Name>` exits 0 (CI checks all contributions with `python scripts/skill_check.py --all-contrib`)
- [ ] `python -m pytest contrib -q` passes
- [ ] It is a CompositeSpec JSON — or it is Python because: <!-- why a spec cannot express it -->
- [ ] `manifest.json` → `verification`: `unit-tested` / `contributor-hardware` (then `hardware_notes` says where and how)
- [ ] Not a port of a published paper or of someone else's code
- [ ] No defaults that describe one instrument (calibration values, ranges, gains, working points)
- [ ] Every commit is signed off (`git commit -s`, Developer Certificate of Origin 1.1)
- [ ] I agree that this contribution is licensed under MIT and may also be included in closed-source or commercial distributions of MAST

---

## 这个 PR 加了什么

<!-- 技能名，一句话说它做什么，以及为什么官方技能（或它们的组合）做不到。 -->

## 清单

- [ ] 只改 `contrib/skills/<名>/` 下的文件
- [ ] `python scripts/skill_check.py contrib/skills/<名>` 退出码为 0（CI 使用 `python scripts/skill_check.py --all-contrib` 检查全部投稿）
- [ ] `python -m pytest contrib -q` 通过
- [ ] 这是组合 spec（CompositeSpec JSON）—— 或者必须写 Python，因为：<!-- 为什么 spec 表达不了 -->
- [ ] `manifest.json` 的 `verification`：`unit-tested` / `contributor-hardware`（后者要在 `hardware_notes` 写明在哪、怎么验的）
- [ ] 不是已发表论文或他人代码的移植
- [ ] 默认值里没有描述某一台仪器的数（标定值、量程、增益、工作点）
- [ ] 每个提交都签了 DCO（`git commit -s`，Developer Certificate of Origin 1.1）
- [ ] 我同意这份投稿按 MIT 许可，并且可能同时被收进 MAST 的闭源或商业发行版
