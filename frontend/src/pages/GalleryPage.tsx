import { GalleryApp } from "@/components/gallery/GalleryApp";

// 实验记录 → 数据图库（/records/gallery）。
//
// 薄壳：视图切换在 GalleryApp 里用 URL query 做，不用 SubTabs——所以这一页不接
// useStickyTab（stickyTab.test.ts 只查用了 <SubTabs 的页面）。「回到上次停的视图」
// 由 GalleryApp 自己记（localStorage["mast.gallery.view"]，读时校验）。

export default function GalleryPage() {
  return <GalleryApp />;
}
