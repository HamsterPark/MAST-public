// 右栏的日常切换器 —— 「回去做那个」= 一次点击。
//
// 取代旧的 ExperimentControls（自由文本输入 + New/End 按钮）。两处关键改动：
//
//  * **删掉 End 按钮**。实验和样品都没有「结束」这个动作 ——「没必要做归档。
//    有的实验可能过了十年重启。」剩下的唯一真实用例是「样品被物理取下、还没
//    装新的」，那是 clear-sample（取消选中），它不写样品行任何字段。
//  * **删掉自由文本 + New**。那正是 create-always 的源头：库里 5 条同名实验
//    相隔 2 分钟就是这么来的。新建改走带重名检测的确认弹窗。
//
// 不新增顶级 tab：一个必须随处可达的东西不该是第 15 个 tab。

import { useState } from "react";
import { Button } from "../controls";
import { useClearSample, useCurrentScope, useSwitchExperiment, useSwitchSample } from "@/api/scope";
import { NewExperimentDialog, NewSampleDialog } from "./NewScopeDialogs";
import { ExperimentPicker, SamplePicker } from "./ScopePickers";
import { TipControls } from "./TipControls";

export function ScopeControls() {
  const scope = useCurrentScope();
  const switchExp = useSwitchExperiment();
  const switchSmp = useSwitchSample();
  const clearSmp = useClearSample();

  const [expPicker, setExpPicker] = useState(false);
  const [smpPicker, setSmpPicker] = useState(false);
  const [newExp, setNewExp] = useState(false);
  const [newSmp, setNewSmp] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const exp = scope.data?.experiment ?? null;
  const smp = scope.data?.sample ?? null;

  const flash = (m: string) => {
    setMsg(m);
    setTimeout(() => setMsg(null), 3000);
  };

  async function pickExperiment(id: string, name: string) {
    setExpPicker(false);
    try {
      const r = await switchExp.mutateAsync({ experimentId: id });
      // 失败必须说出来。一个点了没反应的切换按钮读起来就是「坏了」。
      if (!r.ok) flash(r.block_reason || "切换未生效");
      else if (r.changed) flash(`已切换到「${name}」`);
    } catch {
      flash("切换未送达——网络或后端不可达，请重试");
    }
  }

  async function pickSample(id: string, name: string) {
    setSmpPicker(false);
    if (!exp) return;
    try {
      const r = await switchSmp.mutateAsync({ experimentId: exp.id, sampleId: id });
      if (!r.ok) flash(r.block_reason || "切换未生效");
      else if (r.changed) flash(`已切换到样品「${name}」`);
    } catch {
      flash("切换未送达——请重试");
    }
  }

  return (
    <div className="mt-2 space-y-2">
      <button
        className="flex w-full items-center justify-between gap-2 rounded-mast-ctl border border-mast-border-strong bg-mast-panel px-2.5 py-1.5 text-left text-xs hover:bg-mast-panel-2"
        onClick={() => setExpPicker(true)}
      >
        <span className="text-mast-muted">实验</span>
        <span className="min-w-0 flex-1 truncate text-right text-mast-text">
          {exp?.name || "未选择"}
        </span>
        <span className="text-mast-faint">▾</span>
      </button>

      <button
        className="flex w-full items-center justify-between gap-2 rounded-mast-ctl border border-mast-border-strong bg-mast-panel px-2.5 py-1.5 text-left text-xs hover:bg-mast-panel-2 disabled:opacity-40"
        onClick={() => setSmpPicker(true)}
        disabled={!exp}
      >
        <span className="text-mast-muted">└ 样品</span>
        <span
          className={
            "min-w-0 flex-1 truncate text-right " +
            (smp ? "text-mast-text" : "text-mast-danger")
          }
        >
          {smp?.name || "未选择"}
        </span>
        <span className="text-mast-faint">▾</span>
      </button>

      {smp && (
        <Button variant="ghost" onClick={() => clearSmp.mutate()}>
          取消选中样品（物理出样时用）
        </Button>
      )}

      <div className="text-[10.5px] leading-snug text-mast-faint">
        实验和样品都可以来回切换，切换不会结束任何东西。新开的对话/群聊会记在当前样品下
        —— 在 实验记录 → 群聊记录 按 实验 → 样品 → 群聊 查看。
      </div>
      {msg && <div className="text-[11px] text-mast-muted">{msg}</div>}

      {/* 针尖是仪器级的 —— 与实验/样品并列而不是它们的下级。放在这里是因为
          用户的动线一致（都是「现在在做什么」），但它跨实验存活。 */}
      <div className="border-t border-mast-border pt-2">
        <TipControls />
      </div>

      <ExperimentPicker
        open={expPicker}
        onClose={() => setExpPicker(false)}
        onPick={pickExperiment}
        onCreate={() => {
          setExpPicker(false);
          setNewExp(true);
        }}
        currentId={exp?.id}
      />
      <SamplePicker
        open={smpPicker}
        onClose={() => setSmpPicker(false)}
        onPick={pickSample}
        onCreate={() => {
          setSmpPicker(false);
          setNewSmp(true);
        }}
        experimentId={exp?.id}
        experimentName={exp?.name}
        currentId={smp?.id}
      />
      <NewExperimentDialog
        open={newExp}
        onClose={() => setNewExp(false)}
        currentName={exp?.name}
      />
      <NewSampleDialog
        open={newSmp}
        onClose={() => setNewSmp(false)}
        experimentId={exp?.id}
        experimentName={exp?.name}
        currentSampleName={smp?.name}
      />
    </div>
  );
}

/** 无样品时的常驻横幅。只告知，不困住 —— 对话和查询完全不受影响。 */
export function NoSampleBanner() {
  const scope = useCurrentScope();
  const [dismissed, setDismissed] = useState(false);
  const d = scope.data;
  if (!d || d.degraded || dismissed) return null;
  if (d.has_sample) return null;
  if (!d.hint) return null;

  return (
    <div
      role="status"
      data-testid="no-sample-banner"
      className="flex items-center gap-3 border-b border-mast-warn/40 bg-mast-warn/10 px-4 py-1.5 text-xs text-mast-text"
    >
      <span className="flex-1">{d.hint}</span>
      <button
        className="text-mast-faint hover:text-mast-text"
        onClick={() => setDismissed(true)}
        aria-label="关闭提示"
      >
        ✕
      </button>
    </div>
  );
}
