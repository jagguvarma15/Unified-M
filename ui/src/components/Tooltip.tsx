import { useState, type ReactNode } from "react";

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom";
}

export default function Tooltip({ content, children, side = "top" }: TooltipProps) {
  const [visible, setVisible] = useState(false);

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div
          className={`absolute left-1/2 z-50 -translate-x-1/2 max-w-[240px] rounded-lg bg-slate-800 px-3 py-2 text-xs text-slate-200 shadow-lg ring-1 ring-slate-700/50 whitespace-normal ${
            side === "top" ? "bottom-full mb-2" : "top-full mt-2"
          }`}
          role="tooltip"
        >
          {content}
        </div>
      )}
    </div>
  );
}
