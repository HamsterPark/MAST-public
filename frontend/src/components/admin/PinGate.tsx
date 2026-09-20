import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import { useUiStore } from "@/store";
import { Card } from "@/components/ui";

/** Admin PIN unlock gate. Posts the raw PIN to the core (which compares
 *  SHA256-hex against the launcher-written admin_pin.txt); on success it flips
 *  the shared zustand flag so the rest of the page reveals. Mirrors the live
 *  Gradio admin PIN gate (gui/route_auth.py / app.py:_read_admin_pin_hash). */

const REASON_ZH: Record<string, string> = {
  no_pin_set: "尚未设置管理 PIN（config/admin_pin.txt 缺失或为空）。",
  empty: "请输入 PIN。",
  wrong: "PIN 不正确。",
  degraded: "PIN 文件不可读，管理门控暂不可用。",
};

export function PinGate() {
  const setPinUnlocked = useUiStore((s) => s.setPinUnlocked);
  const [pin, setPin] = useState("");
  const [reason, setReason] = useState<string | null>(null);

  const unlock = useMutation({
    mutationFn: async (raw: string) => {
      const { data, error } = await api.POST("/api/admin/unlock-pin", {
        body: { pin: raw },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data.ok) {
        setReason(null);
        setPinUnlocked(true);
      } else {
        setReason(data.reason ?? "wrong");
      }
    },
    onError: () => setReason("degraded"),
  });

  return (
    <div className="mx-auto max-w-md py-12">
      <Card>
        <h2 className="mb-2 text-base font-semibold text-mast-text">高级管理 · 需要解锁</h2>
        <p className="mb-3 text-sm text-mast-muted">
          此页面包含安全限值、覆盖配置与硬件控制。请输入管理 PIN 解锁。
        </p>
        <p className="mb-4 text-xs italic text-mast-muted/80">
          密码（PIN）在<strong className="not-italic text-mast-text">桌面启动器</strong>的「高级管理 PIN」中设置；此处输入解锁后才会构建编辑器。
        </p>
        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            unlock.mutate(pin);
          }}
        >
          <label className="block text-sm text-mast-muted">管理密码 PIN</label>
          <input
            type="password"
            value={pin}
            onChange={(e) => setPin(e.target.value)}
            placeholder="在启动器中设置的 PIN"
            autoComplete="off"
            className="w-full rounded-md border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent"
          />
          <button
            type="submit"
            disabled={unlock.isPending}
            className="w-full rounded-md bg-mast-accent/20 px-3 py-2 text-sm font-medium text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
          >
            {unlock.isPending ? "校验中…" : "解锁"}
          </button>
        </form>
        {reason && (
          <p className="mt-3 text-sm text-mast-danger">{REASON_ZH[reason] ?? `解锁失败：${reason}`}</p>
        )}
      </Card>
    </div>
  );
}
