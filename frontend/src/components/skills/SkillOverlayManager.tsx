import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import { DataTable } from "@/components/DataTable";
import {
  Section,
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "@/components/ui";
import { Field, TextField, Button, Modal, useToast } from "@/components/controls";
import { SkillEjectPanel } from "./SkillEjectPanel";
import { SkillPacksPanel } from "./SkillPacksPanel";

// ── 技能覆盖层 —— 改 skill 不用发新版本 ─────────────────────────────────────
//
// 这个面板的第一职责**不是**列条目，是回答一个问题：**现在生效了吗？**
//
// 这个功能有三条独立的失败路径，全都不报错：
//   · 重载排队了（任务在跑）—— 注册表都没动；
//   · 注册表换了，但 agent 图还没重建 —— 模型手上仍是旧工具；
//   · 重建被调度了但失败了。
// 所以后端每个响应都带 fingerprint_matches / agent_path_pending，这里按它们显示，
// **不按 ok**。三态而不是两态：null = 判断不了（进程刚起、还没建过工具表），
// 它必须显示成灰色的「未知」，不是绿色的「已生效」。

type EntryRow = {
  path: string;
  enabled: boolean;
  exists: boolean;
  overlay_of: string;
  applied: boolean;
  skills: string[];
  sha256: string;
  error: string;
  valid_path: boolean;
  path_error: string;
};

type SkillInfo = {
  name: string;
  origin: string;
  module: string;
  source_path: string;
  short_sha: string;
  displaced_module: string;
  signature: string;
  described: string;
};

/** 每个条目基于的内置版之后变过没有。
 *
 * 这是「以为生效」的另一个变体，而且更隐蔽：覆盖**确实生效了**，但它基于三个
 * 版本前的代码，上游后来修的 bug 被它原样盖了回去。所以并进主列表显示，
 * 不做成一个要人记得去点的按钮 —— 靠「记得检查」的守卫等于没有守卫。
 */
function useDrift() {
  return useQuery({
    queryKey: ["skill-overlay", "drift"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-overlay/drift");
      if (error) throw error;
      return data;
    },
  });
}

function useOverlay() {
  return useQuery({
    queryKey: ["skill-overlay"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-overlay");
      if (error) throw error;
      return data;
    },
  });
}

/** 生效状态横幅 —— 三态，而且「未知」不能长得像「已生效」。 */
function EffectBanner({
  matches,
  registryOverlaid,
  agentOverlaid,
  pendingRebuild,
}: {
  matches: boolean | null | undefined;
  registryOverlaid: string[];
  agentOverlaid: string[];
  pendingRebuild: boolean;
}) {
  if (pendingRebuild) {
    return (
      <Card>
        <Badge tone="WARN">已排队</Badge>{" "}
        当前有任务在运行，覆盖层重载<strong>已排队</strong>——任务结束后自动生效。
        <div style={{ opacity: 0.75, marginTop: 4, fontSize: "0.9em" }}>
          正在跑的流程从头到尾用它启动时那一版：技能是逐步现查的，中途换会让同一次
          实验的前后半段行为不一致。
        </div>
      </Card>
    );
  }
  if (matches === true) {
    return (
      <Card>
        <Badge tone="AUTO">已生效</Badge>{" "}
        agent 手上的工具表与注册表一致
        {agentOverlaid.length > 0 && <>（{agentOverlaid.length} 个覆盖版技能）</>}。
      </Card>
    );
  }
  if (matches === false) {
    const missing = registryOverlaid.filter((n) => !agentOverlaid.includes(n));
    return (
      <Card>
        <Badge tone="DANGEROUS">尚未生效</Badge>{" "}
        注册表已经换了，但 <strong>agent 手上那张工具表还是旧的</strong> —— 模型现在调的仍是
        覆盖之前的实现。点上面的「重新加载技能」。
        {missing.length > 0 && (
          <div style={{ opacity: 0.75, marginTop: 4, fontSize: "0.9em" }}>
            agent 还看不到：{missing.join("、")}
          </div>
        )}
      </Card>
    );
  }
  return (
    <Card>
      <Badge tone="INFO">未知</Badge>{" "}
      还没有建过 agent 工具表（服务刚启动，或者从未跑过任务），<strong>判断不了</strong>是否生效。
      <div style={{ opacity: 0.75, marginTop: 4, fontSize: "0.9em" }}>
        这不等于「没生效」，也不等于「已生效」—— 跑一次任务或点一次重载之后这里
        才有答案。
      </div>
    </Card>
  );
}

export function SkillOverlayManager() {
  const q = useOverlay();
  const qc = useQueryClient();
  const { toast, node: toastNode } = useToast();
  const [pin, setPin] = useState("");
  const [detail, setDetail] = useState<EntryRow | null>(null);
  const [lastReload, setLastReload] = useState<string>("");

  const reload = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/skill-overlay/reload", {
        body: { pin, reason: "ui" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d: any) => {
      setLastReload(d?.summary || "");
      qc.invalidateQueries({ queryKey: ["skill-overlay"] });
      qc.invalidateQueries({ queryKey: ["skills", "catalog"] });
      if (d?.degraded) {
        toast(d.reason || "覆盖层不可用", "err");
      } else if (d?.status === "queued") {
        // ⚠️ 排队**不是**成功。toast 只有 ok/err 两档，装不下三态；这里用 err：
        // 宁可看起来重一点，也不能让「已安排」被读成「已完成」—— 用户会基于
        // 「已生效」去做下一件事。权威的三态显示在下面的 EffectBanner。
        toast("尚未生效：任务运行中，已排队 —— 当前任务结束后自动生效", "err");
      } else if (d?.failed?.length) {
        toast(`${d.failed.length} 个覆盖模块被拒绝，见下方原因`, "err");
      } else if (d?.agent_path_pending) {
        toast("尚未生效：注册表已更新，但 agent 工具表还没跟上", "err");
      } else {
        toast(`已生效（${d?.applied?.length ?? 0} 个覆盖模块）`, "ok");
      }
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  const setEntry = useMutation({
    mutationFn: async (v: { path: string; enabled: boolean }) => {
      const { data, error } = await api.POST("/api/skill-overlay/entry", {
        body: { path: v.path, enabled: v.enabled, pin, allow_removals: [], note: "" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d: any) => {
      qc.invalidateQueries({ queryKey: ["skill-overlay"] });
      if (!d?.ok) toast(d?.reason || "写入失败", "err");
      // 刻意不自动重载：写清单是一次编辑，让它生效是一次**决定**。
      // 所以文案必须把「还差一次重载」说出来，否则勾完复选框就以为改完了。
      else toast("清单已更新，尚未生效 —— 点「重新加载技能」", "err");
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  const restore = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/skill-overlay/restore-all", {
        body: { pin, reason: "ui" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d: any) => {
      qc.invalidateQueries({ queryKey: ["skill-overlay"] });
      toast(d?.ok ? d.summary || "已恢复到启动基线" : d?.reason || "恢复失败",
            d?.ok ? "ok" : "err");
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  const driftQ = useDrift();
  const drifted = ((driftQ.data ?? []) as any[]).filter((d) => d.drifted === true);
  const driftUnknown = ((driftQ.data ?? []) as any[]).filter(
    (d) => d.known && d.drifted === null);

  const entries = (q.data?.entries ?? []) as EntryRow[];
  const untracked = (q.data?.untracked ?? []) as string[];
  const skills = (q.data?.overlaid_skills ?? []) as SkillInfo[];

  const cols = useMemo<ColumnDef<EntryRow>[]>(
    () => [
      { accessorKey: "path", header: "文件" },
      {
        id: "覆盖",
        header: "覆盖",
        cell: ({ row }) =>
          row.original.overlay_of ? (
            <code style={{ fontSize: "0.85em" }}>{row.original.overlay_of}</code>
          ) : (
            <span style={{ opacity: 0.6 }}>新增</span>
          ),
      },
      {
        id: "状态",
        header: "状态",
        cell: ({ row }) => {
          const r = row.original;
          if (!r.exists) return <Badge tone="DANGEROUS">文件不在</Badge>;
          if (!r.valid_path) return <Badge tone="DANGEROUS">路径非法</Badge>;
          if (!r.enabled) return <Badge tone="INFO">未启用</Badge>;
          if (r.applied) return <Badge tone="AUTO">已生效</Badge>;
          // 启用了、文件也在、却没生效 —— 要么被拒，要么还没重载过。
          return <Badge tone="WARN">未加载</Badge>;
        },
      },
      {
        id: "技能",
        header: "技能",
        cell: ({ row }) =>
          row.original.skills.length ? (
            <span>{row.original.skills.join("、")}</span>
          ) : (
            <span style={{ opacity: 0.6 }}>—</span>
          ),
      },
      {
        id: "sha",
        header: "sha",
        cell: ({ row }) => (
          <code style={{ fontSize: "0.85em", opacity: 0.75 }}>
            {row.original.sha256 ? row.original.sha256.slice(0, 8) : "—"}
          </code>
        ),
      },
      {
        id: "操作",
        header: "",
        cell: ({ row }) => (
          <div style={{ display: "flex", gap: 6 }}>
            <Button
              onClick={() =>
                setEntry.mutate({ path: row.original.path, enabled: !row.original.enabled })
              }
              disabled={setEntry.isPending}
            >
              {row.original.enabled ? "停用" : "启用"}
            </Button>
            <Button onClick={() => setDetail(row.original)}>详情</Button>
          </div>
        ),
      },
    ],
    [setEntry],
  );

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded)
    return <DegradedNote what={`技能覆盖层（${q.data.reason || "不可用"}）`} />;

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <Card>
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end", flexWrap: "wrap" }}>
          <Field label="管理员 PIN（未设置则留空）">
            <TextField value={pin} onChange={setPin} placeholder="" />
          </Field>
          <Button onClick={() => reload.mutate()} disabled={reload.isPending}>
            {reload.isPending ? "重新加载中…" : "重新加载技能"}
          </Button>
          <Button onClick={() => restore.mutate()} disabled={restore.isPending}>
            全部恢复内置
          </Button>
        </div>
        <div style={{ opacity: 0.75, marginTop: 8, fontSize: "0.9em" }}>
          覆盖层目录：<code>{q.data?.overlay_dir}</code>
          <br />
          把改好的 <code>.py</code> 放进去（路径就是它覆盖的模块，例如{" "}
          <code>builtins/bias.py</code> 覆盖 <code>mast.skills.builtins.bias</code>），
          启用，然后重新加载。覆盖<strong>只能收紧</strong>安全画像 —— 放宽包络请走「技能详情编辑」
          里的管理员覆写。
        </div>
      </Card>

      {drifted.length > 0 && (
        <Card>
          <Badge tone="WARN">内置版已变</Badge>{" "}
          {drifted.length} 份覆盖基于的是更早版本的内置代码：
          {drifted.map((d) => d.rel).join("、")}。
          <div style={{ opacity: 0.75, marginTop: 4, fontSize: "0.9em" }}>
            覆盖是<strong>生效的</strong> —— 问题在于它可能把上游后来的修复一起盖了
            回去。重新导出一份，再把你的改动挪过去。
          </div>
        </Card>
      )}
      {driftUnknown.length > 0 && (
        <Card>
          <Badge tone="INFO">判断不了</Badge>{" "}
          {driftUnknown.length} 份覆盖对不出基于哪一版内置代码（
          {driftUnknown.map((d) => d.rel).join("、")}）。
          <div style={{ opacity: 0.75, marginTop: 4, fontSize: "0.9em" }}>
            这不等于「没变」 —— 手写的覆盖没有 sidecar，无从比对。
          </div>
        </Card>
      )}

      <EffectBanner
        matches={q.data?.fingerprint_matches as boolean | null | undefined}
        registryOverlaid={(q.data?.registry_overlaid ?? []) as string[]}
        agentOverlaid={(q.data?.agent_overlaid ?? []) as string[]}
        pendingRebuild={!!q.data?.pending_rebuild}
      />

      {(q.data?.baseline_drift ?? []).length > 0 && (
        <Card>
          <Badge tone="DANGEROUS">基线漂移</Badge>{" "}
          下列技能既不是内置版、也不由覆盖层解释 —— 说明某次回滚没回干净：
          {(q.data!.baseline_drift as string[]).join("、")}。
          可以用「全部恢复内置」硬复位。
        </Card>
      )}

      {lastReload && (
        <Card>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>上一次重载</div>
          <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontSize: "0.9em" }}>
            {lastReload}
          </pre>
        </Card>
      )}

      <Section title={`覆盖层条目（${entries.length}）`}>
        {entries.length === 0 ? (
          <EmptyNote label="覆盖层里还没有条目" />
        ) : (
          <DataTable columns={cols} data={entries} />
        )}
      </Section>

      {untracked.length > 0 && (
        <Card>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>
            目录里有、但清单里没登记（{untracked.length}）
          </div>
          <div style={{ opacity: 0.75, fontSize: "0.9em", marginBottom: 8 }}>
            拷进目录<strong>不会</strong>自动生效 —— 启用是一次显式动作。不单独列出来的话，
            这些文件会被当成「怎么改了没反应」。
          </div>
          <div style={{ display: "grid", gap: 6 }}>
            {untracked.map((p) => (
              <div key={p} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <code style={{ flex: 1 }}>{p}</code>
                <Button onClick={() => setEntry.mutate({ path: p, enabled: true })}>
                  登记并启用
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      {skills.length > 0 && (
        <Section title={`已被覆盖的技能（${skills.length}）`}>
          <div style={{ display: "grid", gap: 6 }}>
            {skills.map((s) => (
              <Card key={s.name}>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <strong>{s.name}</strong>
                  <Badge tone={s.signature.startsWith("verified") ? "AUTO" : "INFO"}>
                    {s.signature.startsWith("verified") ? "已签名" : "本机未签名"}
                  </Badge>
                </div>
                <div style={{ opacity: 0.8, fontSize: "0.9em", marginTop: 4 }}>
                  {s.described}
                </div>
              </Card>
            ))}
          </div>
        </Section>
      )}

      <Modal
        open={!!detail}
        onClose={() => setDetail(null)}
        title={detail ? `覆盖层条目：${detail.path}` : ""}
        wide
      >
        {detail && (
          <div style={{ display: "grid", gap: 8 }}>
            <div>
              覆盖：
              <code>{detail.overlay_of || "（纯新增，不覆盖任何内置模块）"}</code>
            </div>
            <div>文件存在：{detail.exists ? "是" : "否"}</div>
            <div>已启用：{detail.enabled ? "是" : "否"}</div>
            <div>本轮生效：{detail.applied ? "是" : "否"}</div>
            <div>提供技能：{detail.skills.join("、") || "—"}</div>
            <div>
              sha256：<code>{detail.sha256 || "—"}</code>
            </div>
            {detail.path_error && (
              <Card>
                <Badge tone="DANGEROUS">路径非法</Badge> {detail.path_error}
              </Card>
            )}
            {detail.error && (
              <Card>
                <Badge tone="DANGEROUS">被拒绝</Badge>
                <pre style={{ margin: "6px 0 0", whiteSpace: "pre-wrap" }}>
                  {detail.error}
                </pre>
              </Card>
            )}
          </div>
        )}
      </Modal>

      <SkillPacksPanel pin={pin} />
      <SkillEjectPanel pin={pin} />
      {toastNode}
    </div>
  );
}
