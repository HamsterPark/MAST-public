import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Section, Card, Badge, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { Button, TextField } from "@/components/controls";

// ── 签名技能包 ──────────────────────────────────────────────────────────────
//
// 这个面板要回答两个问题，而它们是**不同**的问题：
//   · 装了哪些包？
//   · 它们**现在**还验得过吗？
// 后者不是装的时候验一次就完了 —— 那次和现在之间隔着一段任何人都能写的时间，
// 而覆盖层会把 _packs/ 下的东西标成「已签名」。所以每次刷新都真的重算。

type Pack = {
  pack_id: string;
  version: string;
  author: string;
  description: string;
  n_files: number;
  valid: boolean;
  reasons: string[];
  signature: string;
};

export function SkillPacksPanel({ pin }: { pin: string }) {
  const qc = useQueryClient();
  const [packId, setPackId] = useState("");
  const [fetchNote, setFetchNote] = useState("");

  const q = useQuery({
    queryKey: ["skill-overlay", "packs"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-overlay/packs");
      if (error) throw error;
      return data;
    },
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["skill-overlay"] });
  };

  const remove = useMutation({
    mutationFn: async (pack_id: string) => {
      const { data, error } = await api.POST("/api/skill-overlay/packs/remove", {
        body: { pack_id, pin },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: invalidate,
  });

  const setAuto = useMutation({
    mutationFn: async (enabled: boolean) => {
      const { data, error } = await api.POST("/api/skill-overlay/packs/auto-enable", {
        body: { enabled, pin },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: invalidate,
  });

  // 从推送服务器拉一个包装上。这条后端 2026-08-20 就写好了，但一直**没有入口** ——
  // 界面上只看得到已装的包，装新包得有人到机器前面跑脚本。
  //
  // 默认**不重载**：装是一次编辑，让它生效是一次决定（同 /skill-overlay/entry）。
  // 所以按钮有两个，而不是一个带勾选框的。
  const fetchPack = useMutation({
    mutationFn: async (v: { pack_id: string; reload_now: boolean }) => {
      const { data, error } = await api.POST("/api/skill-overlay/packs/fetch", {
        body: { pack_id: v.pack_id, pin, reload_now: v.reload_now },
      });
      if (error) throw error;
      return data as unknown as {
        ok: boolean; reason: string; summary: string; needs_reload: boolean;
        reloaded: boolean; shadowed: string[];
        agent_path_pending: boolean | null; fingerprint_matches: boolean | null;
      };
    },
    onSuccess: (d) => {
      invalidate();
      // 后端那句 summary 逐字显示 —— 它把「装了几个 / 有没有被遮盖 / 生效了没有」
      // 说全了，前端改写会把它们压成一句。
      setFetchNote(d.ok
        ? d.summary + (d.agent_path_pending ? "（agent 侧尚未跟上）" : "")
        : d.reason || "拉取失败");
    },
    onError: (e: any) => setFetchNote(String(e?.message || e)),
  });

  const packs = (q.data?.packs ?? []) as Pack[];
  const auto = q.data?.auto_enable ?? true;
  const bad = packs.filter((p) => !p.valid);

  return (
    <Section title="签名技能包">
      {q.isLoading && <Spinner />}
      {q.error && <ErrorNote error={q.error} />}
      {q.data?.degraded && (
        <DegradedNote what={`签名技能包（${q.data.reason || "不可用"}）`} />
      )}

      {q.data && !q.data.degraded && (
        <>
          <Card>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <Badge tone={auto ? "AUTO" : "INFO"}>
                {auto ? "自动启用：开" : "自动启用：关"}
              </Badge>
              <div style={{ flex: 1, opacity: 0.8, fontSize: "0.9em", lineHeight: 1.7 }}>
                {auto
                  ? "签名包装进来就登记进清单并启用 —— 但仍要一次重载才生效。"
                  : "签名包照装，但不自动登记 —— 什么时候换由你决定（适合正在跑长实验的机器）。"}
              </div>
              <Button
                onClick={() => setAuto.mutate(!auto)}
                loading={setAuto.isPending}
              >
                {auto ? "关掉" : "打开"}
              </Button>
            </div>
            {q.data.reason && (
              <div style={{ marginTop: 8 }}>
                <Badge tone="DANGEROUS">清单读不出来</Badge>{" "}
                {q.data.reason} —— 这种状态下装包不会自动启用（当成空清单写回去会
                抹掉你已经启用的条目）。
              </div>
            )}
            <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center",
                          flexWrap: "wrap" }}>
              <div style={{ width: 200 }}>
                <TextField value={packId} onChange={setPackId}
                           placeholder="pack_id（从推送服务器索引取）" mono />
              </div>
              <Button
                disabled={!packId.trim() || fetchPack.isPending}
                loading={fetchPack.isPending}
                onClick={() => fetchPack.mutate({ pack_id: packId.trim(), reload_now: false })}
              >
                拉取并安装
              </Button>
              <Button
                variant="primary"
                disabled={!packId.trim() || fetchPack.isPending}
                onClick={() => fetchPack.mutate({ pack_id: packId.trim(), reload_now: true })}
              >
                拉取并立即生效
              </Button>
              <span style={{ fontSize: "0.85em", opacity: 0.75 }}>
                验签 · 逐文件哈希 · 装不上就整包拒绝
              </span>
            </div>
            {fetchNote && (
              <div style={{ marginTop: 6, fontSize: "0.9em" }}>{fetchNote}</div>
            )}
          </Card>

          {bad.length > 0 && (
            <Card>
              <Badge tone="DANGEROUS">复核不过</Badge>{" "}
              {bad.length} 个包现在验不过了 —— 落盘之后被改动过，或者不是通过签名
              通道装进来的。
              <div style={{ opacity: 0.75, marginTop: 4, fontSize: "0.9em" }}>
                它们的条目在下一次重载时会被<strong>整包拒绝</strong>，对应的技能
                退回内置版。这是有意的：「已签名」那个标记必须有凭据。
              </div>
            </Card>
          )}

          {packs.length === 0 ? (
            <EmptyNote label="这台机器还没有装过签名技能包" />
          ) : (
            <div style={{ display: "grid", gap: 8 }}>
              {packs.map((p) => (
                <Card key={p.pack_id}>
                  <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                    <Badge tone={p.valid ? "AUTO" : "DANGEROUS"}>
                      {p.valid ? "已验证" : "验不过"}
                    </Badge>
                    <code style={{ fontWeight: 600 }}>{p.pack_id}</code>
                    <span style={{ opacity: 0.8 }}>v{p.version}</span>
                    <span style={{ opacity: 0.6, fontSize: "0.9em" }}>
                      {p.n_files} 个文件
                    </span>
                    <span style={{ flex: 1 }} />
                    <code style={{ opacity: 0.7, fontSize: "0.85em" }}>
                      {p.signature}
                    </code>
                    <Button
                      variant="danger"
                      onClick={() => remove.mutate(p.pack_id)}
                      loading={remove.isPending}
                    >
                      移除
                    </Button>
                  </div>
                  {(p.author || p.description) && (
                    <div style={{ opacity: 0.75, marginTop: 6, fontSize: "0.9em" }}>
                      {p.author && <>作者 {p.author}　</>}
                      {p.description}
                    </div>
                  )}
                  {p.reasons.length > 0 && (
                    <div style={{ marginTop: 6 }}>
                      {p.reasons.map((r, i) => (
                        <div key={i} style={{ fontSize: "0.9em" }}>
                          · {r}
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          )}
        </>
      )}
    </Section>
  );
}
