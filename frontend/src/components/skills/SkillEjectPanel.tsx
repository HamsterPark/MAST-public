import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Section, Card, Badge, Spinner, ErrorNote, DegradedNote } from "@/components/ui";
import { Field, TextField, Button, Modal } from "@/components/controls";

// ── 导出到覆盖层 ────────────────────────────────────────────────────────────
//
// 打包版里源码全在 MAST.exe 内嵌的 PYZ 归档里，磁盘上一个 .py 都没有 —— 用户
// 想改一个技能，手上根本没有那份文件。这个面板把它取出来。
//
// 但它真正的职责是**导出之后那张告警**：覆盖层换的是注册表里注册的技能类，而
// `from mast.skills.builtins.bias import SetBias` 拿的是模块上的旧引用，热重载
// 换不动。不把这些入口当场列出来，就会出现「我改了它，为什么没反应」——
// 而界面上一切正常。所以导出成功之后**强制**弹一次结果，不是一个 toast 就过去。

type Mod = {
  dotted: string;
  rel: string;
  n_bytes: number;
  already: boolean;
  skill_classes: string[];
  undecidable: string[];
};

type Site = {
  file: string;
  line: number;
  kind: string;
  names: string[];
  text: string;
  binds_skill: boolean;
};

type EjectResult = {
  ok: boolean;
  reason: string;
  dotted: string;
  rel: string;
  path: string;
  sha256: string;
  n_bytes: number;
  skill_classes: string[];
  undecidable: string[];
  n_skill_binds: number;
  warning: string;
  importers: Site[];
};

/** 导出结果 —— 三种 warning 对应三种**决定**，所以徽章也得分开。 */
function WarnBadge({ r }: { r: EjectResult }) {
  if (r.undecidable.length > 0) return <Badge tone="INFO">判断不了</Badge>;
  if (r.skill_classes.length === 0) return <Badge tone="DANGEROUS">不会生效</Badge>;
  if (r.n_skill_binds > 0) return <Badge tone="WARN">部分生效</Badge>;
  return <Badge tone="AUTO">全线生效</Badge>;
}

export function SkillEjectPanel({ pin }: { pin: string }) {
  const qc = useQueryClient();
  const [q, setQ] = useState("");
  const [result, setResult] = useState<EjectResult | null>(null);
  const [busy, setBusy] = useState("");

  const list = useQuery({
    queryKey: ["skill-overlay", "ejectable", q],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-overlay/ejectable", {
        params: { query: { q } },
      });
      if (error) throw error;
      return data;
    },
  });

  const eject = useMutation({
    mutationFn: async (v: { dotted: string; overwrite: boolean }) => {
      const { data, error } = await api.POST("/api/skill-overlay/eject", {
        body: { dotted: v.dotted, overwrite: v.overwrite, pin },
      });
      if (error) throw error;
      return data as EjectResult;
    },
    onSettled: () => setBusy(""),
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ["skill-overlay"] });
      // 成功和失败都弹 —— 失败的那两句（「里面已经有改动」）比成功更需要被读到。
      setResult(d);
    },
  });

  const mods = (list.data?.modules ?? []) as Mod[];
  const searching = q.trim().length > 0;

  return (
    <Section title="导出内置技能到覆盖层">
      <Card>
        <div style={{ opacity: 0.8, fontSize: "0.92em", lineHeight: 1.7 }}>
          打包版里源码全在 <code>MAST.exe</code> 里，磁盘上没有 <code>.py</code>。
          导出会把那份字节原样写进覆盖层目录，<strong>不启用</strong> ——
          改完之后回上面勾选启用，再点「重新加载技能」。
          <br />
          搜索既认模块名，也认技能类名（搜 <code>SetBias</code> 会找到{" "}
          <code>builtins/bias.py</code>）。
        </div>
        <div style={{ marginTop: 10, maxWidth: 420 }}>
          <Field label="搜索" hint="留空则列出全部可导出模块">
            <TextField value={q} onChange={setQ} placeholder="SetBias 或 builtins.bias" mono />
          </Field>
        </div>
      </Card>

      {list.isLoading && <Spinner />}
      {list.error && <ErrorNote error={list.error} />}
      {list.data?.degraded && (
        <DegradedNote what={`导出源码（${list.data.reason || "不可用"}）`} />
      )}

      {list.data && !list.data.degraded && (
        <Card>
          <div style={{ opacity: 0.75, fontSize: "0.9em", marginBottom: 8 }}>
            {mods.length} 个模块
            {searching ? "（已按搜索过滤）" : "（未搜索时不做类分析，所以这里不列技能名）"}
            ，源码来自 <code>{list.data.source_root}</code>
          </div>
          <div style={{ display: "grid", gap: 6, maxHeight: 460, overflowY: "auto" }}>
            {mods.map((m) => (
              <div
                key={m.dotted}
                style={{
                  display: "flex", gap: 10, alignItems: "center",
                  padding: "4px 0", borderBottom: "1px solid rgba(128,128,128,.15)",
                }}
              >
                <code style={{ flex: 1 }}>{m.dotted}</code>
                {m.skill_classes.length > 0 && (
                  <span style={{ opacity: 0.7, fontSize: "0.85em" }}>
                    {m.skill_classes.slice(0, 3).join("、")}
                    {m.skill_classes.length > 3 && ` +${m.skill_classes.length - 3}`}
                  </span>
                )}
                {m.undecidable.length > 0 && <Badge tone="INFO">判断不了</Badge>}
                {m.already && <Badge tone="WARN">已在覆盖层</Badge>}
                <span style={{ opacity: 0.55, fontSize: "0.85em" }}>
                  {(m.n_bytes / 1024).toFixed(1)} KB
                </span>
                <Button
                  onClick={() => {
                    setBusy(m.dotted);
                    eject.mutate({ dotted: m.dotted, overwrite: false });
                  }}
                  loading={busy === m.dotted}
                  disabled={busy !== ""}
                >
                  导出
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Modal
        open={result !== null}
        onClose={() => setResult(null)}
        title={result?.ok ? "已导出到覆盖层" : "没有导出"}
        wide
      >
        {result && (
          <div style={{ display: "grid", gap: 10, fontSize: "0.94em" }}>
            {!result.ok ? (
              <Card>
                <Badge tone="DANGEROUS">未导出</Badge>
                <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>{result.reason}</div>
                {result.reason.includes("overwrite") && (
                  <div style={{ marginTop: 10 }}>
                    <Button
                      variant="danger"
                      onClick={() => {
                        const d = result.dotted;
                        setResult(null);
                        setBusy(d);
                        eject.mutate({ dotted: d, overwrite: true });
                      }}
                    >
                      仍然覆盖（会丢掉覆盖层里现有那份）
                    </Button>
                  </div>
                )}
              </Card>
            ) : (
              <>
                <div>
                  <code>{result.dotted}</code> → <code>{result.rel}</code>（
                  {(result.n_bytes / 1024).toFixed(1)} KB，sha{" "}
                  <code>{result.sha256.slice(0, 12)}</code>）
                </div>
                <Card>
                  <WarnBadge r={result} />
                  <div style={{ marginTop: 6, lineHeight: 1.7 }}>{result.warning}</div>
                </Card>
                <div style={{ opacity: 0.8 }}>
                  这个模块里的技能类：
                  {result.skill_classes.join("、") || "（一个都没有）"}
                </div>
                {result.importers.length > 0 && (
                  <Card>
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>
                      直接 import 这个模块的地方（{result.importers.length}）
                    </div>
                    <div style={{ opacity: 0.75, fontSize: "0.9em", marginBottom: 8 }}>
                      标「绑技能类」的那些拿的是模块上的旧引用，
                      <strong>热重载换不动</strong>：它们会继续跑内置版。
                      要一起改，仍然得发版本。其余的是基类/常量/辅助函数。
                    </div>
                    <div style={{ display: "grid", gap: 4, maxHeight: 280, overflowY: "auto" }}>
                      {result.importers.map((s, i) => (
                        <div key={i} style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
                          {s.binds_skill ? (
                            <Badge tone="WARN">绑技能类</Badge>
                          ) : (
                            <span style={{ width: 68, flexShrink: 0 }} />
                          )}
                          <code style={{ flex: 1 }}>
                            {s.file}:{s.line}
                          </code>
                          <code style={{ opacity: 0.7 }}>{s.text}</code>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
                <Card>
                  下一步：编辑 <code>{result.path}</code>，回到上面把它
                  <strong>启用</strong>，再点「重新加载技能」。
                  导出本身不会改变任何正在运行的东西。
                </Card>
              </>
            )}
          </div>
        )}
      </Modal>
    </Section>
  );
}
