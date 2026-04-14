"use client";

// TODO: implement variable selector (column picker for uploaded dataset)

export interface VariableSelectorProps {
  columnNames: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  label?: string;
}

export function VariableSelector({ label }: VariableSelectorProps) {
  return (
    <div>
      <p className="text-sm text-muted-foreground">{label ?? "选择变量"} — 待实现</p>
    </div>
  );
}
