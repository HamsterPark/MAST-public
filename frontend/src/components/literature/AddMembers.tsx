import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, Badge, DegradedNote } from "@/components/ui";
import { Field, TextField, Button } from "@/components/controls";

// 库成员加入/移除 — POST /api/literature/libraries/{id}/members (add) and
// POST /api/literature/libraries/{id}/members/remove (remove). Paste work_ids /
// DOIs (comma or newline separated); each is a POINTER into the chosen library
// (the paper itself lives in the big library). Mirrors literature_panel
// add_members_h / remove_members_h. The per-library member list is rendered in
// LibraryManager (expand「成员」).

function parseIds(text: string): string[] {
  return text
    .replace(/,/g, "\n")
    .split("\n")
    .map((t) => t.trim())
    .filter(Boolean);
}

export function AddMembers({ libraryId }: { libraryId: string | null }) {
  const qc = useQueryClient();
  const [idsText, setIdsText] = useState("");
  const [reason, setReason] = useState("");

  const m = useMutation({
    mutationFn: async () => {
      if (!libraryId) throw new Error("先在上方选择一个库（检索此库）");
      const ids = parseIds(idsText);
      if (ids.length === 0) throw new Error("粘贴至少一个 work_id / DOI");
      const { data, error } = await api.POST(
        "/api/literature/libraries/{library_id}/members",
        {
          params: { path: { library_id: libraryId } },
          body: { work_ids: ids, reason: reason.trim(), added_by: "user" },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d?.ok) setIdsText("");
      qc.invalidateQueries({ queryKey: ["literature", "libraries"] });
    },
  });

  const removeM = useMutation({
    mutationFn: async () => {
      if (!libraryId) throw new Error("先在上方选择一个库（检索此库）");
      const ids = parseIds(idsText);
      if (ids.length === 0) throw new Error("粘贴至少一个 work_id / DOI");
      const { data, error } = await api.POST(
        "/api/literature/libraries/{library_id}/members/remove",
        {
          params: { path: { library_id: libraryId } },
          body: { work_ids: ids },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d?.ok) setIdsText("");
      qc.invalidateQueries({ queryKey: ["literature", "libraries"] });
      if (libraryId)
        qc.invalidateQueries({
          queryKey: ["literature", "library-detail", libraryId],
        });
    },
  });

  const res = m.data;
  const rmRes = removeM.data;
  return (
    <Card>
      <h3 className="mb-2 text-sm font-semibold">加入库成员（指针）</h3>
      <p className="mb-3 text-xs text-mast-muted">
        {libraryId ? (
          <>
            目标库 <code>{libraryId}</code>。粘贴 work_id / DOI（逗号或换行分隔），每个加入为指针。
          </>
        ) : (
          "先在上方库列表中点「检索此库」选定目标库，再加入成员。"
        )}
      </p>
      <div className="space-y-3">
        <Field label="work_ids / DOIs">
          <textarea
            value={idsText}
            onChange={(e) => setIdsText(e.target.value)}
            rows={3}
            placeholder="W2912345678, 10.1103/PhysRevLett.…"
            className="rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 font-mono text-sm text-mast-text outline-none focus:border-mast-accent"
          />
        </Field>
        <Field label="理由（可选）">
          <TextField value={reason} onChange={setReason} placeholder="为何加入此库" />
        </Field>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="primary"
            disabled={!libraryId || !idsText.trim() || m.isPending}
            onClick={() => m.mutate()}
          >
            {m.isPending ? "加入中…" : "加入指针"}
          </Button>
          <Button
            variant="ghost"
            disabled={!libraryId || !idsText.trim() || removeM.isPending}
            onClick={() => removeM.mutate()}
          >
            {removeM.isPending ? "移除中…" : "移除指针"}
          </Button>
        </div>
        <p className="text-xs text-mast-muted">
          「移除指针」仅从此库删除上面列出的 work_id 指针；论文本身仍留在大库。完整成员列表见上方库列表的「成员」展开。
        </p>
      </div>
      {m.isError && (
        <p className="mt-3 text-sm text-mast-danger">
          加入失败：{String((m.error as Error)?.message ?? m.error)}
        </p>
      )}
      {removeM.isError && (
        <p className="mt-3 text-sm text-mast-danger">
          移除失败：{String((removeM.error as Error)?.message ?? removeM.error)}
        </p>
      )}
      {res && (
        <div className="mt-3 space-y-1 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={res.ok ? "AUTO" : "DANGEROUS"}>{res.message}</Badge>
            {res.added && res.added.length > 0 && (
              <Badge tone="INFO">加入 {res.added.length}</Badge>
            )}
            {res.skipped && res.skipped.length > 0 && (
              <Badge tone="WARN">跳过 {res.skipped.length}</Badge>
            )}
            {res.rejected && res.rejected.length > 0 && (
              <Badge tone="DANGEROUS">拒绝 {res.rejected.length}</Badge>
            )}
            {res.at_cap && <Badge tone="WARN">已达上限</Badge>}
          </div>
          {res.degraded && <DegradedNote what="文献库后端" />}
          <p className="text-mast-muted tabular-nums">
            成员数：{res.member_count}
          </p>
        </div>
      )}
      {rmRes && (
        <div className="mt-3 space-y-1 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={rmRes.ok ? "AUTO" : "DANGEROUS"}>
              {rmRes.message || (rmRes.ok ? "已移除" : "未移除")}
            </Badge>
            {rmRes.n_removed > 0 && (
              <Badge tone="INFO">移除 {rmRes.n_removed}</Badge>
            )}
          </div>
          {rmRes.degraded && <DegradedNote what="文献库后端" />}
          <p className="text-mast-muted tabular-nums">
            成员数：{rmRes.member_count}
          </p>
        </div>
      )}
    </Card>
  );
}
