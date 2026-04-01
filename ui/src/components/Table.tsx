import type { JSX } from "solid-js";

interface TableProps {
  children: JSX.Element;
  class?: string;
}

interface TableHeadProps {
  children: JSX.Element;
  class?: string;
}

interface TableBodyProps {
  children: JSX.Element;
  class?: string;
}

interface TableRowProps {
  children: JSX.Element;
  class?: string;
  onClick?: () => void;
}

interface TableHeaderCellProps {
  children: JSX.Element;
  align?: "left" | "right" | "center";
  class?: string;
}

interface TableCellProps {
  children: JSX.Element;
  align?: "left" | "right" | "center";
  class?: string;
}

const alignClass = { left: "text-left", right: "text-right", center: "text-center" };

/**
 * Semantic table wrapper with consistent styling (shadcn / Tremor style).
 * Use for data tables with header, striped or hover rows.
 */
export function Table(props: TableProps) {
  return (
    <div class="overflow-x-auto rounded-lg border border-slate-200">
      <table class={`w-full text-sm ${props.class ?? ""}`}>{props.children}</table>
    </div>
  );
}

export function TableHead(props: TableHeadProps) {
  return <thead class={`bg-slate-50 border-b border-slate-200 ${props.class ?? ""}`}>{props.children}</thead>;
}

export function TableBody(props: TableBodyProps) {
  return <tbody class={props.class ?? ""}>{props.children}</tbody>;
}

export function TableRow(props: TableRowProps) {
  return (
    <tr
      class={`border-b border-slate-100 transition-colors ${props.onClick ? "cursor-pointer hover:bg-slate-50" : ""} ${props.class ?? ""}`}
      onClick={props.onClick}
      role={props.onClick ? "button" : undefined}
    >
      {props.children}
    </tr>
  );
}

export function TableHeaderCell(props: TableHeaderCellProps) {
  return (
    <th
      class={`py-3 px-4 font-semibold text-slate-600 ${alignClass[props.align ?? "left"]} ${props.class ?? ""}`}
    >
      {props.children}
    </th>
  );
}

export function TableCell(props: TableCellProps) {
  return (
    <td class={`py-3 px-4 ${alignClass[props.align ?? "left"]} ${props.class ?? ""}`}>{props.children}</td>
  );
}
