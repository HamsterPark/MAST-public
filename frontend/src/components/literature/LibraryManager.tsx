import { Fragment, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Card, Badge, Spinner, ErrorNote, DegradedNote, EmptyNote } from "../ui";
import type { components } from "../../api/schema";
import { useCurrentScope } from "../../api/scope";
import { libraryLabel } from "../../lib/libraryLabel";

type LibrarySummary = components["schemas"]["LibrarySummary"];

const SCOPE_TONE: Record<string, string> = {
  global: "DANGEROUS",
  experiment: "WARN",
  custom: "INFO",
};

/** 库管理 — list / create / rename / delete pointer-set libraries.
 *  库 = 大库的指针集合；所有论文都在大库，库只持有 work_id 引用。
 *  ``selectedId`` lets the parent (search panel) scope search to one library.
 *
 *  2026-07-29：**一实验一专属库**。摄取和 lib_add 的默认落点不再是那个全机唯一的
 *  「活动库」，而是**有效库**（后端算出的 `effective_library_id`）—— 有活跃实验时
 *  就是该实验自己的库。所以这里三处要说清楚，否则界面会撒谎：
 *
 *    · 哪个库是「现在真的会被写入的那个」（不是「活动」徽章那个）
 *    · 「设为活动」在实验活跃时**不改落点**（它设的是无实验时的手动指针）
 *    · 库不共享 —— 要用别人的书目就复制一份到当前实验
 */
export function LibraryManager({
  selectedId,
  onSelect,
}: {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const qc = useQueryClient();
  const [newName, setNewName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const librariesQ = useQuery({
    queryKey: ["literature", "libraries"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/libraries");
      if (error) throw error;
      return data;
    },
  });
  // 给还没懒创建出来的实验库取名用。共用 TopBar 的缓存，不额外发请求。
  const scope = useCurrentScope();

  const invalidate = () =>
    qc.invalidateQueries({ queryKey: ["literature", "libraries"] });

  const createM = useMutation({
    mutationFn: async () => {
      // scope 固定 custom：实验库的 id 由实验推导（exp_<id8>）、成员存在实验文件夹里，
      // 只能由后端在首次用到时懒创建。这里让人选 "experiment" 的话，后端会把它降级成
      // custom —— 一个静默变成别的东西的下拉框比没有这个选项更糟。
      const { data, error } = await api.POST("/api/literature/libraries", {
        body: { name: newName.trim(), scope: "custom" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      setNotice(res.message || (res.ok ? "已创建" : "创建失败"));
      if (res.ok) setNewName("");
      invalidate();
    },
    onError: (e) => setNotice(`创建失败：${String((e as Error)?.message ?? e)}`),
  });

  const renameM = useMutation({
    mutationFn: async (vars: { id: string; name: string }) => {
      const { data, error } = await api.PATCH(
        "/api/literature/libraries/{library_id}",
        {
          params: { path: { library_id: vars.id } },
          body: { new_name: vars.name.trim() },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      setNotice(res.message || (res.ok ? "已重命名" : "重命名失败"));
      setRenamingId(null);
      invalidate();
    },
    onError: (e) => setNotice(`重命名失败：${String((e as Error)?.message ?? e)}`),
  });

  const deleteM = useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.DELETE(
        "/api/literature/libraries/{library_id}",
        { params: { path: { library_id: id } } },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (res, id) => {
      setNotice(res.message || (res.ok ? "已删除" : "未删除"));
      if (res.ok && selectedId === id) onSelect(null);
      if (res.ok && expandedId === id) setExpandedId(null);
      invalidate();
    },
    onError: (e) => setNotice(`删除失败：${String((e as Error)?.message ?? e)}`),
  });

  const activateM = useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.POST(
        "/api/literature/libraries/{library_id}/activate",
        { params: { path: { library_id: id } } },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      if (res.degraded) setNotice("文献库后端不可用，未能设置手动指针。");
      else if (!res.ok) setNotice(res.message || "未能设置手动指针");
      else if (effectiveSource === "experiment")
        // 别让它看起来生效了：当前落点仍是实验专属库。
        setNotice(
          `已记下手动指针，但**当前不生效**：有活跃实验时落点仍是它的专属库 ` +
            `${effectiveId}。手动指针只在没有活跃实验时使用。`,
        );
      else setNotice(res.message || "已设为默认库");
      invalidate();
    },
    onError: (e) =>
      setNotice(`设置手动指针失败：${String((e as Error)?.message ?? e)}`),
  });

  const copyM = useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.POST(
        "/api/literature/libraries/{library_id}/copy",
        {
          params: { path: { library_id: id } },
          body: { to_experiment_id: "" }, // 空 = 当前实验
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      if (res.degraded) setNotice("文献库后端不可用，未能复制。");
      else setNotice(res.message || (res.ok ? "已复制" : "未能复制"));
      invalidate();
      qc.invalidateQueries({ queryKey: ["literature", "library-detail"] });
    },
    onError: (e) => setNotice(`复制失败：${String((e as Error)?.message ?? e)}`),
  });

  const data = librariesQ.data;
  const libs: LibrarySummary[] = data?.libraries ?? [];
  const effectiveId = data?.effective_library_id ?? null;
  const effectiveSource = data?.effective_source ?? "";
  const experimentActive = effectiveSource === "experiment";
  const effectiveLabel = libraryLabel(
    effectiveId ?? "",
    effectiveSource,
    libs,
    scope.data?.experiment?.name,
  );

  return (
    <div className="space-y-4">
      {/* create row */}
      <Card>
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col text-xs text-mast-muted">
            新建库名称
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="例如：Ag(111) 自旋态"
              className="mt-1 w-56 rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
            />
          </label>
          <button
            onClick={() => createM.mutate()}
            disabled={!newName.trim() || createM.isPending}
            className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-40"
          >
            {createM.isPending ? "创建中…" : "创建库"}
          </button>
        </div>
        <p className="mt-2 text-xs text-mast-muted">
          库 = 大库的指针集合。所有论文都在大库；库只持有 work_id 引用。
          <br />
          每个实验自带一个专属库（首次收录文献时自动出现，不用在这里建）。这里建的是
          跨实验的自定义库，比如攒着的长期阅读清单。
        </p>
        {/* 「现在真的会被写入哪个库」——「活动」徽章回答不了这个问题了。 */}
        {data && !data.degraded && effectiveId && (
          <p className="mt-2 text-xs">
            {/* 名字在前、编号在后。实验库是懒创建的，第一次收录之前它根本不在
                libs 里，于是这行从前只剩一个裸 exp_1a2b3c4d。 */}
            <span className="text-mast-muted">当前生效的库：</span>
            <b className="text-mast-text">{effectiveLabel.text}</b>{" "}
            <code className="text-mast-accent">{effectiveId}</code>
            <span className="text-mast-muted">
              {experimentActive
                ? " —— 当前实验的专属库；摄取和收录默认都落在这里。"
                : effectiveSource === "manual"
                  ? " —— 手动指定（当前没有活跃实验）。"
                  : " —— 默认兜底库。"}
              {effectiveLabel.pending && "（还没建出来，首次收录时自动建立）"}
            </span>
          </p>
        )}
        {notice && <p className="mt-2 text-xs text-mast-accent">{notice}</p>}
      </Card>

      {/* list */}
      {librariesQ.isPending && <Spinner />}
      {librariesQ.isError && <ErrorNote error={librariesQ.error} />}
      {data?.degraded && <DegradedNote what="文献库" />}
      {data && !data.degraded && libs.length === 0 && (
        <EmptyNote label="暂无文献库。" />
      )}

      {data && !data.degraded && libs.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-mast-border">
          <table className="w-full text-sm">
            <thead className="bg-mast-panel text-mast-muted">
              <tr>
                <th className="px-3 py-2 text-left">名称</th>
                <th className="px-3 py-2 text-left">范围</th>
                <th className="px-3 py-2 text-right">成员</th>
                <th className="px-3 py-2 text-left">library_id</th>
                <th className="px-3 py-2 text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {libs.map((l) => {
                const isGlobal = l.scope === "global";
                const isSelected = selectedId === l.library_id;
                const isExpanded = expandedId === l.library_id;
                const isEffective = l.library_id === effectiveId;
                return (
                  <Fragment key={l.library_id}>
                  <tr
                    className={
                      "border-t border-mast-border " +
                      (isSelected ? "bg-mast-accent/10" : "hover:bg-mast-bg/40")
                    }
                  >
                    <td className="px-3 py-2 align-top font-medium">
                      {renamingId === l.library_id ? (
                        <div className="flex items-center gap-1">
                          <input
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            className="w-40 rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm"
                          />
                          <button
                            onClick={() =>
                              renameM.mutate({ id: l.library_id, name: renameValue })
                            }
                            disabled={!renameValue.trim() || renameM.isPending}
                            className="rounded bg-mast-accent/20 px-2 py-1 text-xs text-mast-accent disabled:opacity-40"
                          >
                            保存
                          </button>
                          <button
                            onClick={() => setRenamingId(null)}
                            className="rounded px-2 py-1 text-xs text-mast-muted hover:text-mast-text"
                          >
                            取消
                          </button>
                        </div>
                      ) : (
                        <span className="flex flex-wrap items-center gap-2">
                          {isEffective && (
                            <Badge tone="AUTO">
                              {experimentActive ? "本实验" : "生效中"}
                            </Badge>
                          )}
                          {l.is_active && !isEffective && (
                            <span title="无活跃实验时才会用到的手动指针">
                              <Badge tone="INFO">手动指针</Badge>
                            </span>
                          )}
                          {l.name || "(未命名)"}
                          {isGlobal && <span title="全局库不可删除">🔒</span>}
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 align-top">
                      <Badge tone={SCOPE_TONE[l.scope] ?? "INFO"}>{l.scope}</Badge>
                      {/* 一个裸 exp_1a2b3c4d 在界面上说明不了任何事 */}
                      {l.experiment_name && (
                        <div
                          className="mt-1 text-[11px] text-mast-muted"
                          title={l.experiment_id ?? ""}
                        >
                          《{l.experiment_name}》
                        </div>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right align-top tabular-nums">
                      {l.member_count}
                    </td>
                    <td className="px-3 py-2 align-top">
                      <code className="text-xs text-mast-muted">{l.library_id}</code>
                    </td>
                    <td className="px-3 py-2 text-right align-top">
                      <div className="flex justify-end gap-1">
                        <button
                          onClick={() => onSelect(isSelected ? null : l.library_id)}
                          className="rounded px-2 py-1 text-xs text-mast-accent hover:bg-mast-accent/20"
                        >
                          {isSelected ? "取消检索范围" : "检索此库"}
                        </button>
                        <button
                          onClick={() =>
                            setExpandedId(
                              expandedId === l.library_id ? null : l.library_id,
                            )
                          }
                          className="rounded px-2 py-1 text-xs text-mast-muted hover:text-mast-text"
                        >
                          {expandedId === l.library_id ? "收起成员" : "成员"}
                        </button>
                        {/* 库不共享 —— 引入别人的书目靠复制（定案） */}
                        {experimentActive && !isEffective && (
                          <button
                            onClick={() => copyM.mutate(l.library_id)}
                            disabled={copyM.isPending || l.member_count === 0}
                            title={
                              l.member_count === 0
                                ? "这个库还没有成员，没什么可复制"
                                : "把这个库的文献复制进当前实验的专属库（原库不动）"
                            }
                            className="rounded px-2 py-1 text-xs text-mast-accent hover:bg-mast-accent/20 disabled:opacity-30"
                          >
                            复制到本实验
                          </button>
                        )}
                        <button
                          onClick={() => activateM.mutate(l.library_id)}
                          disabled={l.is_active || activateM.isPending}
                          title={
                            l.is_active
                              ? "已是手动指针"
                              : experimentActive
                                ? "设为手动指针 —— 有活跃实验时不改变落点，只在没有活跃实验时使用"
                                : "设为默认库"
                          }
                          className="rounded px-2 py-1 text-xs text-mast-accent hover:bg-mast-accent/20 disabled:opacity-40"
                        >
                          {l.is_active
                            ? "已指定"
                            : experimentActive
                              ? "设为手动指针"
                              : "设为默认"}
                        </button>
                        <button
                          onClick={() => {
                            setRenamingId(l.library_id);
                            setRenameValue(l.name);
                          }}
                          className="rounded px-2 py-1 text-xs text-mast-muted hover:text-mast-text"
                        >
                          重命名
                        </button>
                        <button
                          onClick={() => deleteM.mutate(l.library_id)}
                          disabled={isGlobal || deleteM.isPending}
                          title={isGlobal ? "全局库不可删除" : "删除库"}
                          className="rounded px-2 py-1 text-xs text-mast-danger hover:bg-mast-danger-bg disabled:opacity-30"
                        >
                          删除
                        </button>
                      </div>
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr className="border-t border-mast-border bg-mast-bg/30">
                      <td colSpan={5} className="px-3 py-2">
                        <MemberList libraryId={l.library_id} />
                      </td>
                    </tr>
                  )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Per-library member pointer list — GET /api/literature/libraries/{id} (detail
 *  returns members), with per-row remove via POST .../members/remove. */
function MemberList({ libraryId }: { libraryId: string }) {
  const qc = useQueryClient();
  const [notice, setNotice] = useState<string | null>(null);

  const detailQ = useQuery({
    queryKey: ["literature", "library-detail", libraryId],
    queryFn: async () => {
      const { data, error } = await api.GET(
        "/api/literature/libraries/{library_id}",
        { params: { path: { library_id: libraryId } } },
      );
      if (error) throw error;
      return data;
    },
  });

  const removeM = useMutation({
    mutationFn: async (workId: string) => {
      const { data, error } = await api.POST(
        "/api/literature/libraries/{library_id}/members/remove",
        {
          params: { path: { library_id: libraryId } },
          body: { work_ids: [workId] },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      if (res.degraded) setNotice("文献库后端不可用，未能移除指针。");
      else setNotice(res.message || (res.ok ? "已移除" : "未移除"));
      qc.invalidateQueries({
        queryKey: ["literature", "library-detail", libraryId],
      });
      qc.invalidateQueries({ queryKey: ["literature", "libraries"] });
    },
    onError: (e) => setNotice(`移除失败：${String((e as Error)?.message ?? e)}`),
  });

  const detail = detailQ.data;
  const members = detail?.library?.members ?? [];

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-mast-muted">成员指针</div>
      {detailQ.isPending && <Spinner />}
      {detailQ.isError && <ErrorNote error={detailQ.error} />}
      {detail?.degraded && <DegradedNote what="成员列表" />}
      {detail && !detail.degraded && !detail.found && (
        <EmptyNote label="未找到该库。" />
      )}
      {detail && !detail.degraded && detail.found && members.length === 0 && (
        <EmptyNote label="此库暂无成员指针。" />
      )}
      {detail && !detail.degraded && members.length > 0 && (
        <div className="divide-y divide-mast-border overflow-hidden rounded border border-mast-border">
          {members.map((mem) => (
            <div
              key={mem.work_id}
              className="flex flex-wrap items-center gap-2 px-2 py-1.5 text-xs"
            >
              <code className="text-mast-accent">{mem.work_id}</code>
              {mem.doi && <span className="text-mast-muted">DOI {mem.doi}</span>}
              {mem.source && <Badge tone="INFO">{mem.source}</Badge>}
              {/* 全文在不在本机 —— 不显示的话「要不要请用户上传」无从判断 */}
              {mem.fulltext_status === "ingested" ? (
                <span title={mem.fulltext_ref ?? ""}>
                  <Badge tone="AUTO">全文在本机</Badge>
                </span>
              ) : mem.fulltext_status === "requested" ? (
                <Badge tone="WARN">已请求全文</Badge>
              ) : null}
              {mem.copied_from && (
                <span className="text-mast-muted" title="来自另一个库的复制">
                  复制自 {mem.copied_from}
                </span>
              )}
              {mem.reason && (
                <span className="text-mast-muted">理由：{mem.reason}</span>
              )}
              <button
                onClick={() => removeM.mutate(mem.work_id)}
                disabled={removeM.isPending}
                title="从此库移除该指针（论文仍留在大库）"
                className="ml-auto rounded px-2 py-0.5 text-xs text-mast-danger hover:bg-mast-danger-bg disabled:opacity-30"
              >
                移除指针
              </button>
            </div>
          ))}
        </div>
      )}
      {notice && <p className="text-xs text-mast-accent">{notice}</p>}
    </div>
  );
}
