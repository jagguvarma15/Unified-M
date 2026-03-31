import { createSignal, type JSX } from "solid-js";

interface TooltipProps {
  content: JSX.Element;
  children: JSX.Element;
  side?: "top" | "bottom";
}

export default function Tooltip(props: TooltipProps) {
  const [visible, setVisible] = createSignal(false);

  return (
    <div
      class="relative inline-flex"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {props.children}
      <div
        class={`absolute left-1/2 z-50 -translate-x-1/2 max-w-[240px] rounded-lg bg-slate-800 px-3 py-2 text-xs text-slate-200 shadow-lg ring-1 ring-slate-700/50 whitespace-normal transition-opacity pointer-events-none ${
          props.side === "bottom" ? "top-full mt-2" : "bottom-full mb-2"
        } ${visible() ? "opacity-100" : "opacity-0"}`}
        role="tooltip"
      >
        {props.content}
      </div>
    </div>
  );
}
