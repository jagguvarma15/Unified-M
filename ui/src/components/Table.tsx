import type { ReactNode } from "react";

interface TableProps {
  children: ReactNode;
  className?: string;
}

interface TableHeadProps {
  children: ReactNode;
  className?: string;
}

interface TableBodyProps {
  children: ReactNode;
  className?: string;
}

interface TableRowProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}

interface TableHeaderCellProps {
  children: ReactNode;
  align?: "left" | "right" | "center";
  className?: string;
}

interface TableCellProps {
  children: ReactNode;
  align?: "left" | "right" | "center";
  className?: string;
}

const alignClass = { left: "text-left", right: "text-right", center: "text-center" };

/**
 * Semantic table wrapper with consistent styling (shadcn / Tremor style).
 * Use for data tables with header, striped or hover rows.
 */
export function Table({ children, className = "" }: TableProps) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className={`w-full text-sm ${className}`}>{children}</table>
    </div>
  );
}

export function TableHead({ children, className = "" }: TableHeadProps) {
  return <thead className={`bg-slate-50 border-b border-slate-200 ${className}`}>{children}</thead>;
}

export function TableBody({ children, className = "" }: TableBodyProps) {
  return <tbody className={className}>{children}</tbody>;
}

export function TableRow({
  children,
  className = "",
  onClick,
}: TableRowProps) {
  return (
    <tr
      className={`border-b border-slate-100 transition-colors ${onClick ? "cursor-pointer hover:bg-slate-50" : ""} ${className}`}
      onClick={onClick}
      role={onClick ? "button" : undefined}
    >
      {children}
    </tr>
  );
}

export function TableHeaderCell({
  children,
  align = "left",
  className = "",
}: TableHeaderCellProps) {
  return (
    <th
      className={`py-3 px-4 font-semibold text-slate-600 ${alignClass[align]} ${className}`}
    >
      {children}
    </th>
  );
}

export function TableCell({
  children,
  align = "left",
  className = "",
}: TableCellProps) {
  return (
    <td className={`py-3 px-4 ${alignClass[align]} ${className}`}>{children}</td>
  );
}
