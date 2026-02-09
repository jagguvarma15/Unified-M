import { Database } from "lucide-react";

interface Props {
  title?: string;
  message?: string;
}

export default function EmptyState({
  title = "No data available",
  message = "Run the pipeline first to generate results.",
}: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="p-4 bg-slate-100 rounded-full mb-4">
        <Database size={32} className="text-slate-400" />
      </div>
      <h3 className="text-lg font-semibold text-slate-700">{title}</h3>
      <p className="text-sm text-slate-500 mt-1 max-w-md">{message}</p>
      <div className="mt-6 bg-slate-900 rounded-lg p-5 text-left shadow-lg">
        <p className="text-xs font-mono text-slate-300 leading-relaxed">
          <span className="text-slate-500"># generate demo data + train</span>
          <br />
          <span className="text-emerald-400">$</span> PYTHONPATH=src python -m
          cli demo
          <br />
          <br />
          <span className="text-slate-500"># start the API server</span>
          <br />
          <span className="text-emerald-400">$</span> PYTHONPATH=src python -m
          cli serve
          <br />
          <br />
          <span className="text-slate-500"># start the UI (separate terminal)</span>
          <br />
          <span className="text-emerald-400">$</span> cd ui && bun dev
        </p>
      </div>
    </div>
  );
}
