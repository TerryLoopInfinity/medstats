"use client";

// TODO: implement shadcn/ui DataTable with sort + export
// npx shadcn@latest add table

import type { TableResult } from "@/lib/types";

export function DataTable({ table }: { table: TableResult }) {
  return (
    <div className="overflow-x-auto">
      {table.title && <h3 className="font-semibold mb-2">{table.title}</h3>}
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b-2 border-foreground">
            {table.headers.map((h) => (
              <th key={h} className="px-3 py-2 text-left font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-b border-border">
              {row.map((cell, j) => (
                <td key={j} className="px-3 py-2">
                  {cell ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
