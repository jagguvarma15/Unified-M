import { createSignal, onMount, onCleanup, For, type JSX } from "solid-js";

interface VirtualizedListProps<T> {
  rows: T[];
  height: number;
  rowHeight: number;
  renderRow: (row: T, index: number, style: JSX.CSSProperties) => JSX.Element;
}

export function VirtualizedList<T>(props: VirtualizedListProps<T>) {
  let containerEl!: HTMLDivElement;
  const [scrollTop, setScrollTop] = createSignal(0);

  const overscan = 8;

  const visibleStart = () => Math.max(0, Math.floor(scrollTop() / props.rowHeight) - overscan);
  const visibleEnd = () =>
    Math.min(
      props.rows.length,
      Math.ceil((scrollTop() + props.height) / props.rowHeight) + overscan
    );

  const totalHeight = () => props.rows.length * props.rowHeight;

  const onScroll = () => {
    setScrollTop(containerEl.scrollTop);
  };

  onMount(() => {
    containerEl.addEventListener("scroll", onScroll, { passive: true });
    onCleanup(() => containerEl.removeEventListener("scroll", onScroll));
  });

  const visibleRows = () => {
    const start = visibleStart();
    const end = visibleEnd();
    return props.rows.slice(start, end).map((row, i) => ({ row, index: start + i }));
  };

  return (
    <div
      ref={containerEl}
      style={{ height: `${props.height}px`, overflow: "auto", position: "relative" }}
    >
      <div style={{ height: `${totalHeight()}px`, position: "relative" }}>
        <For each={visibleRows()}>
          {({ row, index }) =>
            props.renderRow(row, index, {
              position: "absolute",
              top: `${index * props.rowHeight}px`,
              left: 0,
              right: 0,
              height: `${props.rowHeight}px`,
            })
          }
        </For>
      </div>
    </div>
  );
}
