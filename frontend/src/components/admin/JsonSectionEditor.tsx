import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, ErrorNote, Spinner } from "@/components/ui";
import { Accordion, Button, useToast } from "@/components/controls";
import { StructuredView } from "@/components/admin/StructuredView";

/** Generic effective+override section editor over the section-config seam
 *  (GET/POST → {data, override, has_override, degraded}). Used by knowledge,
 *  encyclopedia, and (as a fallback) guidance. The shapes are arbitrary nested
 *  JSON the old Gradio admin edited field-by-field.
 *
 *  READ: the effective value is rendered with <StructuredView> — a smart
 *  recursive renderer that shapes dicts into titled sections, lists of objects
 *  into reference tables, scalar lists into chips, etc. (NOT a raw JSON blob).
 *  WRITE: the override layer stays editable as validated JSON under a
 *  collapsible "高级" section (NO Dataframe, never freezes, round-trips the
 *  exact override the core persists). Empty/blank override ⇒ reset to default. */

type SectionResponse = {
  key: string;
  data?: unknown;
  override?: unknown;
  has_override?: boolean;
  degraded?: boolean;
};

export function JsonSectionEditor({
  title,
  description,
  getPath,
  postPath,
  pathParam,
  queryKey,
}: {
  title: string;
  description?: string;
  // openapi path templates — caller passes the exact path + the single path param
  getPath: any;
  postPath: any;
  pathParam: Record<string, string>;
  queryKey: (string | undefined)[];
}) {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();

  const q = useQuery({
    queryKey,
    queryFn: async () => {
      const { data, error } = await api.GET(getPath, { params: { path: pathParam } });
      if (error) throw error;
      return data as SectionResponse;
    },
  });

  // Editable draft of the OVERRIDE layer (what the core persists). Seeded from
  // the fetched override (or {} when none). Effective `data` is shown read-only.
  const [draft, setDraft] = useState<string | null>(null);
  const [jsonErr, setJsonErr] = useState<string | null>(null);

  const overrideText = useMemo(() => {
    const ovr = q.data?.override;
    return ovr == null ? "{}" : JSON.stringify(ovr, null, 2);
  }, [q.data]);

  useEffect(() => {
    setDraft(null);
    setJsonErr(null);
  }, [q.data?.key]);

  const effective = q.data?.data;

  const save = useMutation({
    mutationFn: async (payload: unknown) => {
      const { data, error } = await api.POST(postPath, {
        params: { path: pathParam },
        body: { data: payload } as never,
      });
      if (error) throw error;
      return data as { ok?: boolean; degraded?: boolean; reloaded?: boolean };
    },
    onSuccess: (res) => {
      setDraft(null);
      queryClient.invalidateQueries({ queryKey });
      if (res?.degraded) toast("写入未生效（内核未接入）。", "err");
      else if (res?.ok) toast(res.reloaded ? "已保存并热重载。" : "已保存。", "ok");
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const onSave = (resetToDefault: boolean) => {
    if (resetToDefault) {
      save.mutate({});
      return;
    }
    const raw = (draft ?? overrideText).trim();
    let parsed: unknown;
    try {
      parsed = raw === "" ? {} : JSON.parse(raw);
    } catch (e) {
      setJsonErr(`JSON 解析失败：${String((e as Error)?.message ?? e)}`);
      return;
    }
    setJsonErr(null);
    save.mutate(parsed);
  };

  return (
    <Card>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-medium text-mast-text">{title}</span>
        {q.data?.has_override ? <Badge tone="INFO">已覆盖</Badge> : <Badge tone="AUTO">使用默认</Badge>}
      </div>
      {description && <p className="mb-3 text-xs text-mast-muted">{description}</p>}

      {q.isPending && <Spinner />}
      {q.error && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what={`${title}（读取）`} />}

      {q.data && !q.data.degraded && (
        <div className="space-y-4">
          <div>
            <h4 className="mb-2 text-xs font-medium text-mast-muted">生效值（默认 + 覆盖，只读）</h4>
            {effective == null ? (
              <p className="text-sm text-mast-muted">（无默认值）</p>
            ) : (
              <StructuredView value={effective} />
            )}
          </div>

          <Accordion title="高级：编辑覆盖 JSON" defaultOpen={false}>
            <p className="mb-2 text-xs text-mast-muted">
              覆盖层（可编辑 JSON，留空 = 恢复默认）。保存后与上方默认值合并为生效值。
            </p>
            <textarea
              value={draft ?? overrideText}
              spellCheck={false}
              onChange={(e) => {
                setDraft(e.target.value);
                setJsonErr(null);
              }}
              className="h-80 w-full resize-y rounded-md border border-mast-border bg-mast-bg p-3 font-mono text-xs text-mast-text outline-none focus:border-mast-accent"
            />
            {jsonErr && <p className="mt-1 text-xs text-mast-danger">{jsonErr}</p>}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button variant="primary" disabled={save.isPending} onClick={() => onSave(false)}>
                {save.isPending ? "保存中…" : "保存覆盖"}
              </Button>
              {draft != null && (
                <Button variant="ghost" onClick={() => { setDraft(null); setJsonErr(null); }}>
                  撤销编辑
                </Button>
              )}
              <Button variant="danger" disabled={save.isPending} onClick={() => onSave(true)}>
                恢复默认
              </Button>
            </div>
          </Accordion>
        </div>
      )}
      {node}
    </Card>
  );
}
