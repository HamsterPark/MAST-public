import { useEffect, useRef } from "react";

import {
  isAtBottom,
  nearestScrollable,
  type ScrollMetrics,
} from "@/lib/stickToBottom";

/**
 * 新内容来了就跟到底部 —— **但只在用户本来就在底部的时候**。
 *
 * 判断与找容器的逻辑都在 `@/lib/stickToBottom`(纯函数,有单测);这里只负责
 * 把它接到 React 上。为什么要这样拆:那两件事各自都能悄悄坏掉,而它们坏掉的
 * 症状是滚动条会不受控地被拉回去 —— 一个截图看不出、typecheck 抓不到的毛病。
 *
 * ## 与 `scrollIntoView()` 的区别(这是修的那个 bug)
 *
 * `scrollIntoView()` 会把**每一个可滚动祖先**都滚到目标可见 —— 聊天面板一个、
 * 整页一个,于是用户看到「两个滚动条都被拉回去」。这里只对最近的那一个
 * 容器设 `scrollTop`,上面的祖先一律不碰。
 *
 * ## 用户翻上去之后
 *
 * `stickRef` 记的是**上一次滚动事件时**在不在底部。用户一往上翻它就变 false,
 * 跟随随即停止;翻回底部又变 true,跟随自动恢复 —— 不需要任何按钮。
 */
export function useStickToBottom(
  anchor: React.RefObject<HTMLElement | null>,
  deps: unknown[],
  enabled = true,
) {
  // 初值 true:第一次渲染时用户还没滚过,理应跟着最新的走。
  const stick = useRef(true);
  const box = useRef<Element | null>(null);

  // 监听容器的滚动,只更新「在不在底部」这一个布尔。
  useEffect(() => {
    if (!enabled) return;
    const el = nearestScrollable(anchor.current);
    box.current = el;
    if (!el) return;
    const onScroll = () => {
      stick.current = isAtBottom(el as unknown as ScrollMetrics);
    };
    onScroll();
    // passive:这个回调一行赋值,绝不该有能力拖慢滚动本身。
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [anchor, enabled]);

  useEffect(() => {
    if (!enabled || !stick.current) return;
    const el = box.current ?? nearestScrollable(anchor.current);
    if (!el) return;
    // 直接设 scrollTop 而不是 smooth 动画:内容在连续追加时,一个还没跑完的
    // 平滑滚动会和下一条消息打架,表现为画面来回抖。
    el.scrollTop = el.scrollHeight;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled]);
}
