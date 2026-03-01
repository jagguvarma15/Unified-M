import { memo, type CSSProperties, type ReactNode } from "react";
import { FixedSizeList, type ListChildComponentProps } from "react-window";

interface VirtualizedListProps<T> {
  rows: T[];
  height: number;
  rowHeight: number;
  renderRow: (row: T, index: number, style: CSSProperties) => ReactNode;
}

function RowRenderer<T>({
  data,
  index,
  style,
}: ListChildComponentProps<{ rows: T[]; renderRow: VirtualizedListProps<T>["renderRow"] }>) {
  return data.renderRow(data.rows[index], index, style);
}

const MemoRowRenderer = memo(RowRenderer) as typeof RowRenderer;

export function VirtualizedList<T>({
  rows,
  height,
  rowHeight,
  renderRow,
}: VirtualizedListProps<T>) {
  return (
    <FixedSizeList
      height={height}
      itemCount={rows.length}
      itemSize={rowHeight}
      width="100%"
      itemData={{ rows, renderRow }}
      overscanCount={8}
    >
      {MemoRowRenderer}
    </FixedSizeList>
  );
}
