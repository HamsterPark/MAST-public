import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { SelectField, TextField } from "@/components/controls";
import {
  channelListNote,
  channelOptions,
  type SignalChannelOption,
} from "@/lib/signalChannels";

// 信号通道选择器：并列显示索引与名称，避免把不同解调分量混为一谈。
//
// ── 退化时给输入框，绝不给一个空下拉 ───────────────────────────────────
// 名单读不到（没连仪器 / 端点降级）时回落成自由文本输入。硬规则：任何点击最
// 坏只能「无反应/提示」。一个空的下拉框会让已经填好的值**看起来是空的**，
// 而用户的下一步就是重填一遍——用一个他此刻查不到的数字。
//
// 判断逻辑（选项拼装 / 三种退化理由）在 lib/signalChannels.ts，那里能单测：
// node --test 剥不掉 .tsx 里的 JSX，所以能出错的那部分不放在这个文件里。

export function useSignalChannels() {
  return useQuery({
    // Shares the key with SignalCapturePanel on purpose: the two must not each
    // poll the instrument for a table that changes only when someone rewires
    // the rack. `staleTime: Infinity` matches api/tips.ts — the other
    // backend-served vocabulary in this app.
    queryKey: ["experimental", "signals"],
    staleTime: Infinity,
    gcTime: 30 * 60_000,
    retry: 1,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experimental/signals");
      if (error) throw error;
      return data;
    },
  });
}

export function SignalIndexField({
  value,
  onCommit,
  /** Offer 「自动（按通道名查找）」 mapped to this value. qPlus uses -1. */
  autoValue,
  placeholder,
}: {
  /** Stored value as text; "" = unset. */
  value: string;
  onCommit: (next: string) => void;
  autoValue?: number;
  placeholder?: string;
}) {
  const q = useSignalChannels();
  const channels = (q.data?.channels ?? []) as SignalChannelOption[];
  const note = channelListNote({
    pending: q.isPending,
    error: q.isError,
    degraded: q.data?.degraded,
    truncated: q.data?.truncated,
    declared_n: q.data?.declared_n,
    n_channels: q.data?.n_channels,
    count: channels.length,
  });

  if (note) {
    return (
      <div className="space-y-1">
        <TextField value={value} onChange={onCommit} placeholder={placeholder} mono />
        <p className="text-xs text-mast-warn">{note}</p>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <SelectField
        value={value}
        onChange={onCommit}
        options={channelOptions(channels, value, autoValue)}
      />
      <p className="text-xs text-mast-faint">
        共 {channels.length} 路。存下来的是索引，名字只用来认。
      </p>
    </div>
  );
}
