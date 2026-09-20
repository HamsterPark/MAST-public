import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  DEFAULT_LAYERS,
  MAP_PRESETS,
  withLayerDefaults,
  type MapLayerId,
  type MapLayers,
  type MapViewMode,
} from "@/lib/scanMapView";

type Theme = "dark" | "light";

interface UiState {
  theme: Theme;
  activeConversationId: string | null;
  pinUnlocked: boolean;
  /** Which scan-map overlays are drawn. Shared by every place the map is
   *  embedded, so switching a layer off on one page does not leave it on in
   *  another view of the same surface. */
  mapLayers: MapLayers;
  mapViewMode: MapViewMode;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
  setActiveConversation: (id: string | null) => void;
  setPinUnlocked: (v: boolean) => void;
  setMapLayer: (id: MapLayerId, on: boolean) => void;
  applyMapPreset: (presetId: string) => void;
  setMapViewMode: (mode: MapViewMode) => void;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      theme: "dark",
      activeConversationId: null,
      pinUnlocked: false,
      mapLayers: { ...DEFAULT_LAYERS },
      mapViewMode: "fit",
      setTheme: (theme) => set({ theme }),
      toggleTheme: () => set((s) => ({ theme: s.theme === "dark" ? "light" : "dark" })),
      setActiveConversation: (activeConversationId) => set({ activeConversationId }),
      setPinUnlocked: (pinUnlocked) => set({ pinUnlocked }),
      setMapLayer: (id, on) =>
        set((s) => ({ mapLayers: { ...s.mapLayers, [id]: on } })),
      applyMapPreset: (presetId) =>
        set(() => {
          const p = MAP_PRESETS.find((x) => x.id === presetId);
          return p ? { mapLayers: { ...p.layers } } : {};
        }),
      setMapViewMode: (mapViewMode) => set({ mapViewMode }),
    }),
    {
      name: "mast-ui",
      partialize: (s) => ({
        theme: s.theme,
        mapLayers: s.mapLayers,
        mapViewMode: s.mapViewMode,
      }),
      // A layer added after this browser last wrote its state would arrive
      // undefined and read as OFF — invisible, with nothing to click.
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<UiState>;
        return {
          ...current,
          ...p,
          mapLayers: withLayerDefaults(p.mapLayers),
        };
      },
    },
  ),
);

/** Apply the theme class to <html> (replaces the old mastApplyTheme JS). */
export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
}
