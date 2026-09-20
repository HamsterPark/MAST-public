import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

// 共享的 markdown 渲染。报告/计划/论文草稿都是 markdown，此前 report tab 用
// <pre> 直接吐源码 —— 表格是一堆竖线、公式是一堆美元符号、图片是一行链接文本。
//
// 两条约束（项目铁律）：
//   1. 宽内容（表格、代码块）必须在自己的容器里横向滚动，页面本身绝不横向滚动。
//   2. 图片链接是相对路径（../_assets/x.png），浏览器解析不到 —— 所以图片一律
//      降级成一行可见的占位说明，而不是留一个碎图标让人以为报告坏了。
//      要看带图的成品请用「导出 HTML」（图片 base64 内嵌，单文件自包含）。

export function MarkdownView({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  return (
    <div className={`mast-md text-sm leading-relaxed text-mast-text ${className}`}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (p) => <h1 className="mb-2 mt-4 text-lg font-semibold" {...p} />,
          h2: (p) => <h2 className="mb-2 mt-4 text-base font-semibold" {...p} />,
          h3: (p) => <h3 className="mb-1.5 mt-3 text-sm font-semibold" {...p} />,
          p: (p) => <p className="my-2" {...p} />,
          ul: (p) => <ul className="my-2 list-disc space-y-1 pl-5" {...p} />,
          ol: (p) => <ol className="my-2 list-decimal space-y-1 pl-5" {...p} />,
          blockquote: (p) => (
            <blockquote
              className="my-2 border-l-2 border-mast-border pl-3 text-mast-muted"
              {...p}
            />
          ),
          code: ({ className: cls, children, ...rest }) => {
            const inline = !String(cls ?? "").includes("language-");
            if (inline) {
              return (
                <code
                  className="rounded bg-mast-bg/60 px-1 py-0.5 font-mono text-[11px]"
                  {...rest}
                >
                  {children}
                </code>
              );
            }
            return (
              <code className={`font-mono text-[11px] ${cls ?? ""}`} {...rest}>
                {children}
              </code>
            );
          },
          pre: (p) => (
            <pre
              className="my-2 overflow-x-auto rounded-lg bg-mast-bg/60 p-3 text-[11px] leading-relaxed"
              {...p}
            />
          ),
          table: (p) => (
            <div className="my-2 overflow-x-auto">
              <table className="w-full border-collapse text-xs" {...p} />
            </div>
          ),
          th: (p) => (
            <th
              className="border border-mast-border bg-mast-bg/40 px-2 py-1 text-left font-medium"
              {...p}
            />
          ),
          td: (p) => <td className="border border-mast-border px-2 py-1" {...p} />,
          a: (p) => (
            <a
              className="text-mast-accent underline decoration-dotted"
              target="_blank"
              rel="noreferrer"
              {...p}
            />
          ),
          hr: () => <hr className="my-3 border-mast-border" />,
          // 相对路径的图在浏览器里必然加载不到 —— 说清楚它是什么、去哪看，
          // 而不是留一个碎图标。
          img: ({ src, alt }) => (
            <span className="my-2 flex items-center gap-2 rounded border border-dashed border-mast-border px-2 py-1.5 text-[11px] text-mast-muted">
              <span aria-hidden>🖼</span>
              <span>
                图：{alt || "(无说明)"}
                <span className="ml-1 font-mono opacity-70">{src}</span>
                <span className="ml-1">— 导出 HTML 可看到内嵌图片</span>
              </span>
            </span>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}
