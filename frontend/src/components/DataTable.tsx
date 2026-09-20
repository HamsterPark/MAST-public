import { useState } from "react";
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table";

/** Generic sortable/filterable table — the typed replacement for Gradio's
 *  freeze-prone gr.Dataframe. Virtualization can be layered on for huge sets. */
export function DataTable<T>({
  data,
  columns,
  globalFilter,
  empty = "暂无数据",
}: {
  data: T[];
  columns: ColumnDef<T, any>[];
  globalFilter?: string;
  empty?: string;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  if (!data.length)
    return (
      <div className="flex items-center gap-2 rounded-mast-ctl border border-dashed border-mast-border-strong px-3 py-2.5 text-xs text-mast-muted">
        <svg
          width="15"
          height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M22 12h-6l-2 3h-4l-2-3H2" />
          <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
        </svg>
        {empty}
      </div>
    );

  return (
    <div className="overflow-hidden rounded-mast-card border border-mast-border shadow-mast">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-mast-panel-2">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => {
                  const sorted = h.column.getIsSorted() as string;
                  return (
                    <th
                      key={h.id}
                      onClick={h.column.getToggleSortingHandler()}
                      className={
                        "cursor-pointer select-none whitespace-nowrap px-3.5 py-2 text-left text-xs font-medium tracking-wide " +
                        (sorted ? "text-mast-accent" : "text-mast-faint")
                      }
                    >
                      <span className="inline-flex items-center gap-1">
                        {flexRender(h.column.columnDef.header, h.getContext())}
                        <span className="text-[10px] opacity-70">
                          {sorted === "asc" ? "▲" : sorted === "desc" ? "▼" : ""}
                        </span>
                      </span>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className="border-t border-mast-border transition-colors hover:bg-mast-panel-2/40"
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-3.5 py-2.5 align-top">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
