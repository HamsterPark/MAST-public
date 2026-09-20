import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Badge, Card, ErrorNote, Spinner } from "../ui";

// 设置 → 远程访问 — cross-network remote control over Tailscale (READ-ONLY status).
//
// The bind itself (0.0.0.0 + basic-auth + self-signed TLS) is enabled in the
// desktop 启动器 (勾「启用局域网 / 远程访问」+ 重启服务) because it must happen at
// server start. Here we SHOW the live readiness the backend reports via
// GET /api/remote-access: whether the service is bound for remote access, the
// Tailscale status, and the exact address(es) to open on the OTHER computer.
//
// Foolproof flow (also spelled out in the panel):
//   ① 两台电脑都装 Tailscale 并登录同一账号
//   ② 本机启动器勾「启用局域网 / 远程访问」→ 重启服务
//   ③ 另一台电脑浏览器打开下方「Tailscale」地址（首次证书告警点「继续」）

type ToastFn = (msg: string, kind?: "ok" | "err") => void;

async function copyText(text: string, toast?: ToastFn) {
  try {
    await navigator.clipboard.writeText(text);
    toast?.("已复制到剪贴板", "ok");
  } catch {
    toast?.("复制失败，请手动选择复制", "err");
  }
}

export function RemoteAccessSection({ toast }: { toast?: ToastFn }) {
  const q = useQuery({
    queryKey: ["remote-access"],
    // Slow poll so the badges flip to 就绪 shortly after Tailscale comes up /
    // logs in, without hammering the CLI. Degrade-safe: never raises server-side.
    refetchInterval: 10000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/remote-access");
      if (error) throw error;
      return data;
    },
  });

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;

  const d = q.data;
  const ts = d.tailscale;
  const urls = d.urls ?? [];

  const tsBadge = !ts.installed ? (
    <Badge tone="DANGEROUS">未安装</Badge>
  ) : ts.ready ? (
    <Badge tone="AUTO">就绪</Badge>
  ) : (
    <Badge tone="WARN">{ts.backend_state || "未就绪"}</Badge>
  );

  return (
    <Card>
      <div className="mb-3 flex flex-wrap items-center gap-2" data-testid="remote-access-section">
        <span className="text-sm font-medium text-mast-text">跨网远程访问（Tailscale）</span>
        {d.lan_enabled ? (
          <Badge tone="AUTO">已开放远程</Badge>
        ) : (
          <Badge tone="WARN">仅本机</Badge>
        )}
        <span className="text-mast-muted">·</span>
        <span className="text-xs text-mast-muted">Tailscale</span>
        {tsBadge}
        {ts.device_name && (
          <span className="text-xs text-mast-muted">
            本机 <span className="font-mono text-mast-text">{ts.device_name}</span>
            {ts.tailnet ? ` · ${ts.tailnet}` : ""}
          </span>
        )}
        <button
          onClick={() => q.refetch()}
          className="ml-auto rounded-md border border-mast-border px-2 py-1 text-xs text-mast-muted hover:text-mast-text"
        >
          刷新
        </button>
      </div>

      {/* 一句话状态/引导 — always shown, from the backend's `note`. */}
      {d.note && (
        <p className="mb-3 rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 text-sm text-mast-text">
          {d.note}
        </p>
      )}

      {/* 远端地址 — the whole point: what to open on the OTHER computer. */}
      {urls.length > 0 ? (
        <div className="space-y-2">
          <span className="text-xs text-mast-muted">在另一台电脑（已登录同一 Tailscale 账号）打开：</span>
          {urls.map((u, i) => (
            <div key={u.url} className="flex items-center gap-2 text-sm">
              <span className="w-32 shrink-0 text-xs text-mast-muted">
                {u.label}
                {i === 0 ? " ★" : ""}
              </span>
              <code className="flex-1 truncate rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 font-mono text-mast-text">
                {u.url}
              </code>
              <button
                onClick={() => copyText(u.url, toast)}
                className="shrink-0 rounded-md border border-mast-border px-2 py-1.5 text-xs text-mast-muted hover:text-mast-text"
              >
                复制
              </button>
              <button
                onClick={() => window.open(u.url, "_blank", "noopener")}
                className="shrink-0 rounded-md border border-mast-border px-2 py-1.5 text-xs text-mast-muted hover:text-mast-text"
              >
                打开
              </button>
            </div>
          ))}
        </div>
      ) : null}

      {/* LAN IP — secondary; same-office fallback that doesn't need Tailscale. */}
      {d.lan_ip && (
        <p className="mt-3 text-xs text-mast-muted">
          局域网地址（同一网络内，无需 Tailscale）：
          <span className="ml-1 font-mono text-mast-text">
            {d.scheme}://{d.lan_ip}:{d.port}
          </span>
        </p>
      )}

      {/* 傻瓜式三步 — always visible so the operator knows the whole recipe. */}
      <div className="mt-3 rounded-md border border-mast-border bg-mast-bg/40 px-3 py-2 text-xs text-mast-muted">
        <div className="mb-1 font-medium text-mast-text">怎么用（三步）</div>
        <ol className="list-decimal space-y-0.5 pl-4">
          <li>两台电脑都安装 Tailscale 并登录<strong>同一账号</strong>（tailscale.com/download）。</li>
          <li>
            <strong>本机</strong>在桌面「MAST 启动器」勾选「启用局域网 / 远程访问」，设置账号密码，然后<strong>重启服务</strong>。
          </li>
          <li>
            <strong>另一台电脑</strong>浏览器打开上方带 <strong>★（推荐）</strong> 的 <strong>Tailscale IP</strong> 地址，输入账号密码即可控制；首次提示证书不受信任，点<strong>「高级 → 继续」</strong>。
            <br />
            <span className="text-mast-muted">
              ⚠️ 别用 <code>.ts.net</code> 名字打开——自签证书下浏览器会因 HSTS 直接拒绝（除非用 <code>tailscale cert</code> 配了真证书）。
            </span>
          </li>
        </ol>
      </div>
    </Card>
  );
}
