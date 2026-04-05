import { createSignal, type JSX } from "solid-js";

interface TooltipProps {
  content: JSX.Element;
  children: JSX.Element;
  side?: "top" | "bottom" | "right";
}

export default function Tooltip(props: TooltipProps) {
  const [visible, setVisible] = createSignal(false);

  const positionClass = () => {
    if (props.side === "bottom")
      return "top-full mt-2 left-1/2 -translate-x-1/2";
    if (props.side === "right")
      return "left-full ml-2 top-1/2 -translate-y-1/2";
    return "bottom-full mb-2 left-1/2 -translate-x-1/2";
  };

  return (
    <div
      class="relative inline-flex"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocusIn={() => setVisible(true)}
      onFocusOut={() => setVisible(false)}
    >
      {props.children}
      <div
        class={`absolute z-50 max-w-[240px] rounded-lg bg-slate-800 px-3 py-2 text-xs text-slate-200 shadow-lg ring-1 ring-slate-700/50 whitespace-normal transition-opacity pointer-events-none ${positionClass()} ${visible() ? "opacity-100" : "opacity-0"}`}
        role="tooltip"
      >
        {props.content}
      </div>
    </div>
  );
}
