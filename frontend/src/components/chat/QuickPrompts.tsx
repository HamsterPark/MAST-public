import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

// Quick-prompt button row — curated + operator-override prompts from
// GET /api/chat/quick-prompts. Clicking a button fills the chat input. Mirrors
// the old Gradio 快速提示按钮行.

export function QuickPrompts({ onPick }: { onPick: (prompt: string) => void }) {
  const q = useQuery({
    queryKey: ["chat", "quick-prompts"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/chat/quick-prompts");
      if (error) throw error;
      return data;
    },
    staleTime: 60_000,
  });

  const prompts = q.data?.prompts ?? [];
  if (q.isPending || prompts.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1.5">
      {q.data?.degraded && (
        <span className="text-xs text-mast-warn">（覆盖项降级，仅显示内置）</span>
      )}
      {prompts.map((p, i) => (
        <button
          key={`${i}-${p.label}`}
          type="button"
          onClick={() => onPick(p.prompt)}
          title={p.prompt}
          className="rounded-full border border-mast-border bg-mast-panel/60 px-3 py-1 text-xs text-mast-text hover:border-mast-accent hover:text-mast-accent"
        >
          {p.label || p.prompt.slice(0, 20)}
        </button>
      ))}
    </div>
  );
}
