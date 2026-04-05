import type { JSX } from "solid-js";
import { createSignal, Show, For } from "solid-js";
import { density, TH_PAD, TD_PAD } from "../lib/density";
import { ArrowUp, ArrowDown, ArrowUpDown } from "../lib/icons";

interface TableProps {
  children: JSX.Element;
  class?: string;
}

interface TableHeadProps {
  children: JSX.Element;
  class?: string;
  sticky?: boolean;
}

interface TableBodyProps {
  children: JSX.Element;
  class?: string;
}

interface TableRowProps {
  children: JSX.Element;
  class?: string;
  onClick?: () => void;
  selected?: boolean;
}

interface TableHeaderCellProps {
  children: JSX.Element;
  align?: "left" | "right" | "center";
  class?: string;
  sortable?: boolean;
  sorted?: "asc" | "desc" | null;
  onSort?: () => void;
}

interface TableCellProps {
  children: JSX.Element;
  align?: "left" | "right" | "center";
  class?: string;
}

interface SelectableTableRowProps {
  children: JSX.Element;
  class?: string;
  selected?: boolean;
  onSelect?: (selected: boolean) => void;
  onClick?: () => void;
}

const alignClass = { left: "text-left", right: "text-right", center: "text-center" };

export function Table(props: TableProps) {
  return (
    <div class="overflow-x-auto rounded-lg border border-slate-200">
      <table class={`w-full text-sm ${props.class ?? ""}`}>{props.children}</table>
    </div>
  );
}

export function TableHead(props: TableHeadProps) {
  return (
    <thead
      class={`bg-slate-50 border-b border-slate-200 ${props.sticky ? "sticky top-0 z-10" : ""} ${props.class ?? ""}`}
    >
      {props.children}
    </thead>
  );
}

export function TableBody(props: TableBodyProps) {
  return <tbody class={props.class ?? ""}>{props.children}</tbody>;
}

export function TableRow(props: TableRowProps) {
  return (
    <tr
      class={`border-b border-slate-100 transition-colors ${
        props.selected ? "bg-indigo-50" : ""
      } ${props.onClick ? "cursor-pointer hover:bg-slate-50" : ""} ${props.class ?? ""}`}
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
      class={`font-semibold text-slate-600 ${TH_PAD[density()]} ${alignClass[props.align ?? "left"]} ${
        props.sortable ? "cursor-pointer select-none hover:text-slate-900 transition-colors" : ""
      } ${props.class ?? ""}`}
      onClick={props.sortable ? props.onSort : undefined}
    >
      <span class="inline-flex items-center gap-1">
        {props.children}
        <Show when={props.sortable}>
          <span class="inline-flex shrink-0">
            {props.sorted === "asc"
              ? ArrowUp({ size: 12, class: "text-indigo-600" })
              : props.sorted === "desc"
              ? ArrowDown({ size: 12, class: "text-indigo-600" })
              : ArrowUpDown({ size: 12, class: "text-slate-300" })}
          </span>
        </Show>
      </span>
    </th>
  );
}

export function TableCell(props: TableCellProps) {
  return (
    <td class={`${TD_PAD[density()]} ${alignClass[props.align ?? "left"]} ${props.class ?? ""}`}>{props.children}</td>
  );
}

export function SelectableTableRow(props: SelectableTableRowProps) {
  return (
    <tr
      class={`border-b border-slate-100 transition-colors ${
        props.selected ? "bg-indigo-50" : "hover:bg-slate-50"
      } ${props.onClick ? "cursor-pointer" : ""} ${props.class ?? ""}`}
      onClick={props.onClick}
    >
      <td class={`${TD_PAD[density()]} w-10`}>
        <input
          type="checkbox"
          checked={props.selected ?? false}
          onChange={(e) => props.onSelect?.(e.currentTarget.checked)}
          onClick={(e) => e.stopPropagation()}
          class="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
        />
      </td>
      {props.children}
    </tr>
  );
}

/** Helper hook for managing sort state on a table */
export function createSortState<T extends string>(defaultKey?: T, defaultDir: "asc" | "desc" = "asc") {
  const [sortKey, setSortKey] = createSignal<T | null>(defaultKey ?? null);
  const [sortDir, setSortDir] = createSignal<"asc" | "desc">(defaultDir);

  const toggle = (key: T) => {
    if (sortKey() === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(() => key);
      setSortDir("asc");
    }
  };

  const sorted = (key: T) => (sortKey() === key ? sortDir() : null);

  return { sortKey, sortDir, toggle, sorted };
}
