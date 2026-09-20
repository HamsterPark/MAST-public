import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { useCurrentScope } from "../../api/scope";
import { libraryLabel } from "../../lib/libraryLabel";

/** 「这篇会进哪个库」的目标横幅。
 *
 *  为什么需要它（2026-07-29）：摄取和上传面板把 ``library_id`` 空着提交时，落点
 *  以前是那个全机唯一的「活动库」，现在是**有效库** —— 有活跃实验时就是该实验自己
 *  的库。落点变了而界面没说，用户上传一篇 PDF 就不知道它进了哪儿；而「进错实验的
 *  库」这种错，事后没有任何东西会提示他。
 *
 *  只读一个已经存在的查询（``/api/literature/libraries`` 带 ``effective_library_id``
 *  与 ``effective_source``），与 LibraryManager 共用同一份缓存，不额外请求。
 *  后端降级或还没加载时**什么都不显示** —— 猜一个落点写上去比不写更糟。
 */
export function LibraryTargetNote({
  libraryId,
  verb = "加入",
}: {
  /** 面板当前显式选中的库；``null`` = 交给后端算有效库。 */
  libraryId: string | null;
  verb?: string;
}) {
  const q = useQuery({
    queryKey: ["literature", "libraries"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/libraries");
      if (error) throw error;
      return data;
    },
  });
  // 已经被 TopBar 轮询着，共用同一份缓存，不额外发请求。
  const scope = useCurrentScope();

  const data = q.data;
  if (!data || data.degraded) return null;

  const libs = data.libraries ?? [];
  const targetId = libraryId || data.effective_library_id || "";
  if (!targetId) return null;
  const explicit = Boolean(libraryId);
  const source = explicit ? "picked" : (data.effective_source ?? "");

  // 实验库优先用实验名 —— 一个裸 exp_1a2b3c4d 说明不了任何事。库还没被懒创建
  // 出来时（第一次收录之前）名字也拿得到：它就是当前实验的名字。
  const { text: label, pending } = libraryLabel(
    targetId,
    source,
    libs,
    scope.data?.experiment?.name,
  );

  const why =
    source === "picked"
      ? "（你在左侧指定的库）"
      : source === "experiment"
        ? "（当前实验的专属库）"
        : source === "manual"
          ? "（手动指定；当前没有活跃实验）"
          : "（默认兜底库）";

  return (
    <p className="mt-2 rounded border border-mast-border bg-mast-bg/40 px-2 py-1.5 text-xs text-mast-muted">
      将{verb} {label} <span className="opacity-70">{why}</span>
      {" · "}
      <code className="text-mast-accent">{targetId}</code>
      {source === "experiment" && (
        <>
          <br />
          书目记在这个实验的文件夹里（<code>library/members.jsonl</code>）；论文本身
          和全文都在本机大库，库只持有指针。
          {pending && "这个库还没建出来 —— 这一次收录就会建它。"}
        </>
      )}
    </p>
  );
}

/** 「它进了哪个库」的行内引用 —— 名字在前，编号在后。
 *
 *  #49 修的是三个「将会落到哪」的横幅，漏了两个「已经落到哪」的回显
 *  （摄取结果的「· 库 exp_example」、上传结果的「· 已入库 exp_example
 *  （当前实验的专属库）」）。同一个抱怨的同一个形状：操作刚做完、正要确认它去了
 *  对的地方的那一刻，界面给的恰恰是最没有信息量的那串编号。
 *
 *  与 :func:`LibraryTargetNote` 共用同一个 queryKey 与同一个 `libraryLabel`，
 *  所以不会有第二套命名规则，也不额外发请求。名字查不到时退回裸 id —— 少一个
 *  名字是不便，少一个 id 是丢信息。
 */
export function LibraryRef({
  libraryId,
  source = "",
}: {
  libraryId: string;
  /** 后端回的 `pointer_library_source`；没有就留空。 */
  source?: string | null;
}) {
  const q = useQuery({
    queryKey: ["literature", "libraries"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/libraries");
      if (error) throw error;
      return data;
    },
  });
  const scope = useCurrentScope();

  const id = (libraryId || "").trim();
  if (!id) return null;
  const libs = q.data && !q.data.degraded ? (q.data.libraries ?? []) : [];
  const { text } = libraryLabel(
    id,
    source || "",
    libs,
    scope.data?.experiment?.name,
  );
  return (
    <>
      {text && text !== id && <span className="text-mast-text">{text} </span>}
      <code className="text-mast-accent" title={id}>
        {id}
      </code>
    </>
  );
}
