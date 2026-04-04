/**
 * React-island wrapper for rendering recharts (React) component trees
 * inside a SolidJS application.
 *
 * recharts components are React class / forwardRef objects that cannot be
 * called as plain functions by SolidJS's JSX transform. This component
 * creates a real React root and renders the chart tree through React's
 * reconciler, bridging the two frameworks.
 *
 * Usage:
 *   <ReactChart>
 *     {() => h(ResponsiveContainer, { width: "100%", height: 300 },
 *       h(BarChart, { data: myData() },
 *         h(XAxis, { dataKey: "date" }),
 *         h(Bar, { dataKey: "value" })
 *       )
 *     )}
 *   </ReactChart>
 */
import React from "react";
import { createRoot, type Root } from "react-dom/client";
import { createEffect, onCleanup } from "solid-js";

/** Alias for React.createElement — keeps chart code concise. */
export const h = React.createElement;

export default function ReactChart(props: {
  children: () => React.ReactNode;
  class?: string;
}) {
  let container!: HTMLDivElement;
  let root: Root | undefined;

  createEffect(() => {
    const content = props.children();
    if (!root) {
      root = createRoot(container);
    }
    root.render(content as any);
  });

  onCleanup(() => {
    const r = root;
    if (r) {
      // Defer unmount so React doesn't warn about sync unmount during render
      queueMicrotask(() => r.unmount());
    }
  });

  return (<div ref={container!} class={props.class} />) as any;
}
