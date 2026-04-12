import {
  FlaskConical,
  Plus,
  Calendar,
  CheckCircle2,
  Circle,
} from "../lib/icons";

interface RoadmapItem {
  title: string;
  description: string;
  status: "planned" | "in-progress" | "completed";
  date?: string;
}

const PLACEHOLDER_ITEMS: RoadmapItem[] = [
  {
    title: "Baseline Calibration",
    description: "Establish baseline priors using historical lift tests",
    status: "completed",
    date: "Completed",
  },
  {
    title: "Geo-Lift Test — Paid Social",
    description: "Run incrementality test on Facebook/Instagram spend",
    status: "in-progress",
    date: "In progress",
  },
  {
    title: "Holdout Test — Brand Search",
    description: "Measure true incremental value of branded keywords",
    status: "planned",
    date: "Planned",
  },
  {
    title: "Cross-Channel Interaction Study",
    description: "Test interaction effects between display and paid social",
    status: "planned",
    date: "Planned",
  },
];

function statusIcon(status: RoadmapItem["status"]) {
  switch (status) {
    case "completed":
      return CheckCircle2({ size: 18, class: "text-emerald-500" });
    case "in-progress":
      return Calendar({ size: 18, class: "text-amber-500" });
    case "planned":
      return Circle({ size: 18, class: "text-slate-300" });
  }
}

export default function ExperimentRoadmap() {
  return (
    <div class="space-y-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-slate-900">Experiment Roadmap</h1>
          <p class="mt-1 text-sm text-slate-500">
            Plan and track calibration experiments, lift tests, and validation
            studies.
          </p>
        </div>
        <button class="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 transition-colors">
          {Plus({ size: 16 })}
          New Experiment
        </button>
      </div>

      <div class="rounded-xl border border-slate-200 bg-white shadow-sm">
        <div class="border-b border-slate-100 px-5 py-4">
          <div class="flex items-center gap-2 text-sm font-semibold text-slate-700">
            {FlaskConical({ size: 16, class: "text-indigo-500" })}
            Experiment Timeline
          </div>
        </div>
        <ul class="divide-y divide-slate-100">
          {PLACEHOLDER_ITEMS.map((item) => (
            <li class="flex items-start gap-4 px-5 py-4">
              <div class="mt-0.5 shrink-0">{statusIcon(item.status)}</div>
              <div class="flex-1 min-w-0">
                <p class="text-sm font-medium text-slate-900">{item.title}</p>
                <p class="mt-0.5 text-sm text-slate-500">{item.description}</p>
              </div>
              <span
                class={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  item.status === "completed"
                    ? "bg-emerald-50 text-emerald-700"
                    : item.status === "in-progress"
                      ? "bg-amber-50 text-amber-700"
                      : "bg-slate-50 text-slate-500"
                }`}
              >
                {item.date}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div class="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
        <p class="text-sm text-slate-500">
          Connect your experiment platform or add experiments manually to build
          your roadmap.
        </p>
      </div>
    </div>
  );
}
