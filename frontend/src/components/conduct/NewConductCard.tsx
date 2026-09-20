import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, EmptyNote, Spinner } from "@/components/ui";
import { Button, Field, SelectField, TextField } from "@/components/controls";
import {
  buildCreateParams,
  createOutcomeMessage,
  envelopeHint,
  experimentPickState,
  initialValues,
  normalizeCreateError,
  parseParamInput,
  pickTemplate,
  preSubmitNote,
  serverFieldErrors,
  templateApprovalNote,
  templateListState,
  templateOptionLabel,
  unassignedErrors,
  type CreateOutcomeLike,
  type ParamSpecLike,
  type TemplateRowLike,
} from "@/lib/conductForm";

// ════════════════════════════════════════════════════════════════════════════
// 新建一份 conduct —— **这一页是唯一的入口**。
//
// 在它之前空态上写的是「新建走 POST /api/conducts」,也就是说这个动作只有 curl
// 一条路。那与 approve 是同一个形状(设计 §7 让 approve 只接受 UI 来源 ⇒ 不给按钮
// approve 就不存在),而**做不到的动作和不存在的动作在屏幕上长得一模一样**。
//
// ── 表单不预填任何物理量 ────────────────────────────────────────────────────
//
// 每个格子都是空的。模板的 `ParamSpec.default` 全是 `None` 且是刻意的:填不出来
// 就说明这次实验的工作点还没定,那正是该停下来的时候。后端的 `ParamSpecRow` 契约
// 里**根本没有 default 这个字段** —— 拿不到就不会预填,这比在这里写一行「记得别
// 预填」可靠。
//
// ── 三处「不许把人锁在门外」──────────────────────────────────────────────────
//
// 1. **提交按钮永远能按。** 本地校验只是提前告知(红字 + 一句提示),真源是后端。
//    本地判据万一比 `check_params` 严,用户不会被自己这一侧锁死。
// 2. **实验下拉拉不到时留手填。** `experiment_id` 必填,下拉堵死就等于新建堵死。
// 3. **批不下去的模板照样能建草稿。** 被挡的是下一步的 approve,不是这一步。
//
// ── 逐字段回显 ──────────────────────────────────────────────────────────────
//
// 400 的回包里 `params_echo` 是后端**按字段分好组**的,直接盖到对应格子上。而
// `errors` 里没能落到任何格子上的那些(单活跃不变式、模板不存在……)一条都不许
// 吞 —— 逐字段回显是好东西,但它不能变成一个吞掉其余一切的漏斗。
// ════════════════════════════════════════════════════════════════════════════

export function NewConductCard({
  activeConductId,
  onCreated,
  onClose,
}: {
  /** 当前那个未了结的 conduct(单活跃不变式)。非空 ⇒ 新建会被 409。 */
  activeConductId: string;
  onCreated: (conductId: string) => void;
  onClose: () => void;
}) {
  const [specId, setSpecId] = useState("");
  const [experimentId, setExperimentId] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [outcome, setOutcome] = useState<CreateOutcomeLike | null>(null);
  const [serverErrs, setServerErrs] = useState<Record<string, string>>({});
  const [note, setNote] = useState<{ text: string; tone: "ok" | "err" } | null>(null);
  const [busy, setBusy] = useState(false);

  const templates = useQuery({
    queryKey: ["conduct", "templates"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/conducts/templates");
      if (error) throw error;
      return data;
    },
  });

  const experiments = useQuery({
    queryKey: ["experiments", "list"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experiments");
      if (error) throw error;
      return data;
    },
  });

  const listState = templateListState(templates);
  const rows = (templates.data?.templates ?? []) as TemplateRowLike[];
  const picked = pickTemplate(rows, specId);
  const specs = (picked?.params_schema ?? []) as ParamSpecLike[];
  const approval = templateApprovalNote(picked);

  const expState = experimentPickState(experiments);
  const expRows = experiments.data?.experiments ?? [];

  const built = useMemo(() => buildCreateParams(specs, values), [specs, values]);

  const chooseTemplate = (id: string) => {
    setSpecId(id);
    // 换模板 = 换一整套参数。**初值全空** —— 见文件头。
    setValues(initialValues(pickTemplate(rows, id)?.params_schema ?? []));
    setOutcome(null);
    setServerErrs({});
    setNote(null);
  };

  const setOne = (name: string, v: string) => {
    setValues((prev) => ({ ...prev, [name]: v }));
    // 后端那条错误说的是**上次提交的那个值**。值一改它就过期了 —— 留着不动
    // 就是拿一次旧结论当成对当前输入的判断。
    setServerErrs((prev) => {
      if (!(name in prev)) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  const submit = async () => {
    setBusy(true);
    setNote(null);
    try {
      const { data, error, response } = await api.POST("/api/conducts", {
        body: {
          spec_id: specId,
          experiment_id: experimentId.trim(),
          params: built.params,
          // 手建的不是从某个 APPROVED plan 编译来的。空串就是「没有来源」——
          // 编造一个 plan id 会让审计流里多出一份根本不存在的来路。
          from_plan_id: "",
        },
      });
      // 4xx 的**回包体**才是有用的那部分(哪个格子错了),所以不 throw:
      // throw 掉的话屏幕上只剩一句「加载失败」,而后端刚刚逐字段说清楚了。
      const out: CreateOutcomeLike = response.ok
        ? ((data ?? {}) as CreateOutcomeLike)
        : normalizeCreateError(error ?? data, response.status);
      setOutcome(out);
      setServerErrs(serverFieldErrors(out));
      const msg = createOutcomeMessage(out, Boolean(response.ok && out.ok));
      setNote(msg);
      if (response.ok && out.ok && out.conduct_id) onCreated(String(out.conduct_id));
    } catch (e) {
      // 发不出去(网络/代理)⇒ 一句话,**不冻结**、不转一个不会停的圈。
      setNote({ text: `发不出去：${String((e as Error)?.message ?? e)}`, tone: "err" });
    } finally {
      setBusy(false);
    }
  };

  const topErrors = unassignedErrors(outcome);
  const pre = preSubmitNote(built);

  return (
    <Card className="border-mast-accent">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-[17px] font-semibold text-mast-text">新建一份 conduct</h3>
          <p className="mt-1 text-sm text-mast-muted">
            建出来的是**草稿**：它不会自己跑起来，下一步还要有人批准。
            参数一个都不预填——填不出来就说明这次实验的工作点还没定。
          </p>
        </div>
        <Button variant="ghost" onClick={onClose}>
          收起
        </Button>
      </div>

      {/* ── 模板 ───────────────────────────────────────────────────────── */}
      <div className="mt-4">
        {listState.kind === "loading" && <Spinner label={listState.message} />}
        {listState.kind === "unreadable" && (
          <div className="rounded-mast-ctl border border-mast-danger-border bg-mast-danger-bg px-3 py-2.5 text-sm text-mast-danger">
            {listState.message}
          </div>
        )}
        {listState.kind === "empty" && <EmptyNote label={listState.message} />}
        {listState.kind === "ready" && (
          <div className="max-w-lg">
            <Field
              label="模板（必填）"
              hint="模板决定阶段与步骤；数值一个都不在模板里，全在下面那些格子里。"
            >
              <SelectField
                value={specId}
                onChange={chooseTemplate}
                options={[
                  { value: "", label: "（请选择一个模板）" },
                  ...rows.map((r) => ({
                    value: String(r.spec_id ?? ""),
                    label: templateOptionLabel(r),
                  })),
                ]}
              />
            </Field>
          </div>
        )}
      </div>

      {picked && (
        <>
          {picked.stages?.length ? (
            <p className="mt-2 text-xs text-mast-faint">
              阶段：{picked.stages.join(" → ")}
            </p>
          ) : null}

          {/* 批不下去 ≠ 不能建。被挡的是下一步。 */}
          {approval.blocked && (
            <div className="mt-3 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2.5 text-sm text-mast-warn">
              <p>{approval.text}</p>
              {picked.findings?.length ? (
                <>
                  <p className="mt-2 text-xs text-mast-faint">发现：</p>
                  <ul className="ml-4 list-disc text-xs">
                    {picked.findings.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                </>
              ) : null}
              {picked.checks_skipped?.length ? (
                <>
                  <p className="mt-2 text-xs text-mast-faint">这些检查**没能跑**：</p>
                  <ul className="ml-4 list-disc text-xs">
                    {picked.checks_skipped.map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                </>
              ) : null}
            </div>
          )}

          {/* ── 实验归属 ──────────────────────────────────────────────── */}
          <div className="mt-4 max-w-lg">
            {expState.kind === "list" ? (
              <Field
                label="绑定的实验（必填）"
                hint="没有实验归属的 conduct，产物没有落点：spec 快照与 progress.jsonl 都写进那个实验的文件夹。"
              >
                <SelectField
                  value={experimentId}
                  onChange={setExperimentId}
                  options={[
                    { value: "", label: "（请选择一个实验）" },
                    ...expRows.map((e) => ({
                      value: String(e.id ?? ""),
                      label: `${e.name || e.id}${e.sample_name ? `（${e.sample_name}）` : ""}`,
                    })),
                  ]}
                />
              </Field>
            ) : (
              <Field label="绑定的实验 ID（必填）" hint={expState.message}>
                <TextField value={experimentId} onChange={setExperimentId} mono
                  placeholder="实验 ID" />
              </Field>
            )}
          </div>

          {/* ── 逐字段 ────────────────────────────────────────────────── */}
          {specs.length ? (
            <div className="mt-4">
              <h4 className="mb-2 text-sm font-semibold text-mast-muted">
                这个模板要你填的参数（{specs.length} 个）
              </h4>
              <div className="grid gap-3 sm:grid-cols-2">
                {specs.map((p) => (
                  <ParamField
                    key={p.name}
                    spec={p}
                    raw={values[p.name] ?? ""}
                    serverError={serverErrs[p.name] ?? ""}
                    onChange={(v) => setOne(p.name, v)}
                  />
                ))}
              </div>
            </div>
          ) : (
            <p className="mt-4 text-sm text-mast-faint">
              这个模板没有声明可填参数。
            </p>
          )}

          {/* ── 提交 ──────────────────────────────────────────────────── */}
          {activeConductId && (
            <div className="mt-4 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2.5 text-sm text-mast-warn">
              已经有一份未了结的 conduct（{activeConductId}）。单活跃不变式由数据库执行，
              所以这一次多半会被拒 —— 按钮**不拦你**，那句拒绝由后端说，说得比这里准。
            </div>
          )}

          {topErrors.length > 0 && (
            <div className="mt-4 rounded-mast-ctl border border-mast-danger-border bg-mast-danger-bg px-3 py-2.5 text-sm text-mast-danger">
              <p className="font-semibold">没建成</p>
              <ul className="ml-4 mt-1 list-disc text-xs">
                {topErrors.map((e) => (
                  <li key={e}>{e}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <Button variant="primary" loading={busy} onClick={submit}>
              建立草稿
            </Button>
            {/* 本地看出来的问题写在这里，但**不禁用按钮**:一个本地判错就再也
                提交不上去的表单，是「能停不能解」的又一例。 */}
            {pre && <span className="text-xs text-mast-warn">{pre}</span>}
            {note && (
              <span
                className={
                  note.tone === "ok" ? "text-xs text-mast-auto" : "text-xs text-mast-danger"
                }
              >
                {note.text}
              </span>
            )}
          </div>
        </>
      )}
    </Card>
  );
}

/**
 * 一个格子。
 *
 * 单位、用途、包络**写在格子旁边**,不藏在 tooltip 里:1.5 nA 填成 1.5 安培那次,
 * 缺的正是「这一格的单位是安培、上限是 100 nA」这句话摆在眼前。
 *
 * 回读(`= 1.50 nA（= 1.5e-9 A）`)是这一格最要紧的东西:它显示的是**系统实际
 * 收到的那个数**,量级错了当场看得见 —— 而包络只在越界时才出声,量级错在包络
 * 之内的那些(比如 50 pA 打成 5 pA)只有回读能帮上忙。
 */
function ParamField({
  spec,
  raw,
  serverError,
  onChange,
}: {
  spec: ParamSpecLike;
  raw: string;
  serverError: string;
  onChange: (v: string) => void;
}) {
  const parsed = parseParamInput(raw, spec);
  const hint = envelopeHint(spec);
  // 后端那句优先:它是真源。本地那句在还没提交时先顶上。
  const error = serverError || parsed.error;
  const required = spec.required !== false;

  return (
    <div className="flex flex-col gap-1">
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-mast-muted">
          <span className="font-mono text-mast-text">{spec.name}</span>
          {spec.unit ? <span className="ml-1 text-xs">（{spec.unit}）</span> : null}
          {required ? (
            <span className="ml-1 text-xs text-mast-danger">必填</span>
          ) : (
            <span className="ml-1 text-xs text-mast-faint">选填（留空则用模板的默认值）</span>
          )}
        </span>
        <input
          type="text"
          value={raw}
          onChange={(e) => onChange(e.target.value)}
          placeholder={spec.unit ? `如 1.5${spec.unit === "A" ? "n" : ""}${spec.unit}` : ""}
          className="rounded-mast-ctl border border-mast-border-strong bg-mast-panel px-2.5 py-2 font-mono text-sm tabular-nums text-mast-text outline-none focus:border-mast-accent"
        />
      </label>

      {/* 系统**实际**收到的那个数。 */}
      {parsed.echo && (
        <span className="font-mono text-xs text-mast-accent">= {parsed.echo}</span>
      )}
      {hint && <span className="text-xs text-mast-faint">{hint}</span>}
      {spec.help && <span className="text-xs text-mast-faint">{spec.help}</span>}
      {error && <span className="text-xs text-mast-danger">{error}</span>}
    </div>
  );
}
