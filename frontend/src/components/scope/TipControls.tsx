// 右栏的针尖卡片 —— 「现在装的是哪根针」+ 登记换针。
//
// 为什么和实验/样品放在一起、但**不是**它们的下级：针尖是仪器级的。换实验、
// 换样品都未必换针尖，所以它没有 experiment_id，也没有「切换针尖」——
// 针尖服役期是线性的，装下一根就自动退役上一根。
//
// 登记弹窗里那句关于「会清掉学习标定」的提示不是客套：dI/dV 接触标定和 qPlus
// 自由振幅基线绑的是上一根针，不清就会被继续当真值用（撞针判据的分母尤其致命）。
// 用户按下去之前必须知道这件事，所以它就地写在按钮上方，不做 toast。

import { useState } from "react";
import clsx from "clsx";
import { Button, Field, SelectField, TextField } from "../controls";
import { ConfirmDialog } from "./ConfirmDialog";
import {
  serviceDays,
  tipSummary,
  useCurrentTip,
  useRegisterTip,
  useRemoveCurrentTip,
  useTipHistory,
  useTipVocabulary,
  type Tip,
} from "@/api/tips";

function errText(e: unknown): string {
  if (!e) return "";
  if (e instanceof Error) return e.message;
  return typeof e === "string" ? e : JSON.stringify(e);
}

/** 数字输入：空串 = 未知（线径经常真的不知道），非法输入返回 null。 */
function numOrNull(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const n = Number.parseFloat(t);
  return Number.isFinite(n) && n > 0 ? n : null;
}

export function TipControls() {
  const cur = useCurrentTip();
  const [register, setRegister] = useState(false);
  const [history, setHistory] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const remove = useRemoveCurrentTip();

  const tip = cur.data?.tip ?? null;
  const days = serviceDays(tip);

  const flash = (m: string) => {
    setMsg(m);
    setTimeout(() => setMsg(null), 4000);
  };

  return (
    <div className="mt-2 space-y-2">
      <button
        className="flex w-full items-center justify-between gap-2 rounded-mast-ctl border border-mast-border-strong bg-mast-panel px-2.5 py-1.5 text-left text-xs hover:bg-mast-panel-2"
        onClick={() => setRegister(true)}
        data-testid="tip-chip"
      >
        <span className="text-mast-muted">针尖</span>
        <span
          className={
            "min-w-0 flex-1 truncate text-right " +
            (tip ? "text-mast-text" : "text-mast-faint")
          }
        >
          {tip ? tipSummary(tip) : "未登记"}
          {tip?.form === "qplus" && " ⚡"}
        </span>
        <span className="text-mast-faint">＋</span>
      </button>

      {tip && (
        <div className="px-0.5 text-[10.5px] leading-snug text-mast-faint">
          「{tip.name}」
          {days !== null && <>，已服役 {days} 天</>}
          {tip.wire_diameter_mm ? <>，线径 {tip.wire_diameter_mm} mm</> : null}
        </div>
      )}

      <div className="flex gap-2">
        <Button variant="ghost" onClick={() => setHistory(true)}>
          换针史
        </Button>
        {tip && (
          <Button
            variant="ghost"
            onClick={async () => {
              try {
                const r = await remove.mutateAsync();
                flash(r.ok ? "已记录针尖取出" : r.error || "未生效");
              } catch (e) {
                flash(errText(e) || "未送达，请重试");
              }
            }}
          >
            记录取出
          </Button>
        )}
      </div>

      <div className="text-[10.5px] leading-snug text-mast-faint">
        针尖是<b>仪器级</b>的：换实验、换样品都不必重新登记。装了新针请登记一次——
        修针方案会按针尖的材料/制备/形态取参数，样品会话也会自动带上针尖信息。
      </div>
      {msg && <div className="text-[11px] text-mast-muted">{msg}</div>}

      <RegisterTipDialog open={register} onClose={() => setRegister(false)} />
      <TipHistoryDialog open={history} onClose={() => setHistory(false)} />
    </div>
  );
}

export function RegisterTipDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const vocab = useTipVocabulary(open);
  const register = useRegisterTip();
  const cur = useCurrentTip();

  const [material, setMaterial] = useState("W");
  const [fabrication, setFabrication] = useState("etched");
  const [form, setForm] = useState("stm_wire");
  const [name, setName] = useState("");
  const [diameter, setDiameter] = useState("");
  const [installedAt, setInstalledAt] = useState("");
  const [sensorModel, setSensorModel] = useState("");
  const [f0, setF0] = useState("");
  const [q, setQ] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const prev = cur.data?.tip ?? null;

  function reset() {
    setName("");
    setDiameter("");
    setInstalledAt("");
    setSensorModel("");
    setF0("");
    setQ("");
    setNote("");
    setErr(null);
    onClose();
  }

  async function doRegister() {
    setErr(null);
    try {
      const r = await register.mutateAsync({
        material,
        fabrication,
        form,
        name: name.trim(),
        wire_diameter_mm: numOrNull(diameter),
        installed_at: installedAt.trim(),
        qplus_sensor_model: form === "qplus" ? sensorModel.trim() : "",
        qplus_f0_hz: form === "qplus" ? numOrNull(f0) : null,
        qplus_q: form === "qplus" ? numOrNull(q) : null,
        note: note.trim(),
      });
      // 失败时弹窗保持打开 + 就地红字。静默失败是这里最贵的失败模式。
      if (!r.ok) {
        setErr(r.error || "登记未生效");
        return;
      }
      reset();
    } catch (e) {
      setErr(errText(e) || "登记未送达——网络或后端不可达，请重试");
    }
  }

  const materials = vocab.data?.materials ?? [{ value: "W", label: "钨 (W)" }];
  const fabrications = vocab.data?.fabrications ?? [
    { value: "etched", label: "电化学腐蚀" },
  ];
  const forms = vocab.data?.forms ?? [
    { value: "stm_wire", label: "普通 STM 金属丝针尖" },
  ];

  return (
    <ConfirmDialog
      open={open}
      title="登记装入针尖"
      confirmLabel="登记这根针尖"
      busy={register.isPending}
      busyLabel="登记中…"
      error={err}
      onConfirm={doRegister}
      onClose={reset}
    >
      <div className="space-y-3">
        <p className="text-xs text-mast-muted">
          记录「装入了一根新针尖」。针尖是<b>仪器级</b>的，与当前实验/样品无关。
        </p>

        <Field label="材料">
          <SelectField value={material} onChange={setMaterial} options={materials} />
        </Field>
        <Field label="制备方式">
          <SelectField
            value={fabrication}
            onChange={setFabrication}
            options={fabrications}
          />
        </Field>
        <Field label="形态" hint="qPlus 型针尖会对「戳表面」类处理自动加一道门。">
          <SelectField value={form} onChange={setForm} options={forms} />
        </Field>
        <Field label="名称（可留空）" hint="留空会按材料和序号自动生成，如 W-etched #3。">
          <TextField value={name} onChange={setName} placeholder="如 W-etched #3" />
        </Field>
        <Field label="线材直径 mm（如果知道）">
          <TextField value={diameter} onChange={setDiameter} placeholder="如 0.25" />
        </Field>
        <Field
          label="装入日期（可留空 = 现在）"
          hint="可以回填——过一两天才想起来记是常事。格式 2026-07-28。"
        >
          <TextField
            value={installedAt}
            onChange={setInstalledAt}
            placeholder="2026-07-28"
          />
        </Field>

        {form === "qplus" && (
          <div className="space-y-3 rounded border border-mast-border bg-mast-panel-2 px-3 py-2">
            <p className="text-[11px] text-mast-muted">
              qPlus 传感器的<b>标称</b>参数（铭牌/型录值）。实测的共振频率和 Q 由
              PLL 频率扫描自动写入，与这里分开存。
            </p>
            <Field label="音叉型号">
              <TextField
                value={sensorModel}
                onChange={setSensorModel}
                placeholder="如 qPlus TF-32k"
              />
            </Field>
            <Field label="标称共振频率 f₀ (Hz)">
              <TextField value={f0} onChange={setF0} placeholder="如 32768" />
            </Field>
            <Field label="标称 Q">
              <TextField value={q} onChange={setQ} placeholder="如 30000" />
            </Field>
          </div>
        )}

        <Field label="备注">
          <TextField value={note} onChange={setNote} placeholder="可留空" />
        </Field>

        {prev && (
          <div className="rounded border border-mast-warn/40 bg-mast-warn/10 px-3 py-2 text-xs text-mast-text">
            当前登记的是「{prev.name}」。登记新针尖会把它标记为已取出，并
            <b>清除绑它的学习标定</b>（到样品 dI/dV 标定值、qPlus 自由振幅基线）——
            那些量是上一根针学出来的，换针后继续用就是错的。
            旧值会归档进那一行，日后仍可查。
          </div>
        )}
      </div>
    </ConfirmDialog>
  );
}

/**
 * 换针史 —— 全仓**唯一**的针尖卡片渲染。
 *
 * 2026-08-06 导出：实验记录页要「链到针尖卡片」，而针尖没有独立路由，
 * 卡片就是这里这一份。导出它而不是在记录页照着画一遍 —— 抄一份的话，两处会各自
 * 演化，而针尖卡片上写的是材料/制法/服役期这类**会被拿去做判断**的事实。
 *
 * `highlightId` 把某一行圈出来。这是「链接」在没有 per-tip 路由时能做到的最好形式：
 * 打开的是真正的那张卡片，而不是一份看起来像它的副本。
 */
export function TipHistoryDialog({
  open,
  onClose,
  highlightId,
}: {
  open: boolean;
  onClose: () => void;
  highlightId?: string | null;
}) {
  const history = useTipHistory(open);
  const tips = history.data?.tips ?? [];

  return (
    <ConfirmDialog
      open={open}
      title="换针史"
      confirmLabel="关闭"
      cancelLabel="返回"
      wide
      onConfirm={onClose}
      onClose={onClose}
    >
      {history.isLoading ? (
        <p className="text-sm text-mast-muted">加载中…</p>
      ) : tips.length === 0 ? (
        <p className="text-sm text-mast-muted">还没有登记过针尖。</p>
      ) : (
        <div className="max-h-96 space-y-2 overflow-y-auto">
          {tips.map((t: Tip) => (
            <div
              key={t.id}
              className={clsx(
                "rounded border px-3 py-2 text-xs",
                t.id === highlightId
                  ? "border-mast-accent bg-mast-accent-soft"
                  : "border-mast-border bg-mast-panel",
              )}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-semibold text-mast-text">
                  {t.tip_index ? `T${String(t.tip_index).padStart(2, "0")} ` : ""}
                  {t.name || "未命名"}
                </span>
                <span
                  // `mast-ok` is not a token (the success colour is `mast-auto`),
                  // so this used to compile to nothing and「在用」inherited the
                  // body colour — the same silent-no-op shape as ①/#61.
                  className={t.removed_at ? "text-mast-faint" : "text-mast-auto"}
                >
                  {t.removed_at ? "已取出" : "在用"}
                </span>
              </div>
              <div className="mt-0.5 text-mast-muted">{tipSummary(t)}</div>
              <div className="mt-0.5 text-mast-faint">
                {t.installed_at ? `${t.installed_at.slice(0, 10)} 装入` : "装入日期未记录"}
                {t.removed_at ? ` → ${t.removed_at.slice(0, 10)} 取出` : ""}
                {t.wire_diameter_mm ? ` · 线径 ${t.wire_diameter_mm} mm` : ""}
              </div>
              {t.note && <div className="mt-0.5 text-mast-muted">{t.note}</div>}
              {Object.keys(t.retire_snapshot || {}).length > 0 && (
                <div className="mt-1 text-[10.5px] text-mast-faint">
                  退役时归档的标定：{Object.keys(t.retire_snapshot).join("、")}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </ConfirmDialog>
  );
}
