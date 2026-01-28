import { ReactNode, useState } from "react";

interface StepTabsProps {
  tabs: { id: string; label: string; icon?: ReactNode; content: ReactNode }[];
}

export function StepTabs({ tabs }: StepTabsProps) {
  const [active, setActive] = useState(tabs[0]?.id);

  return (
    <div className="space-y-6">
      <div className="flex gap-3 overflow-x-auto pb-2">
        {tabs.map((tab) => {
          const isActive = tab.id === active;
          return (
            <button
              key={tab.id}
              onClick={() => setActive(tab.id)}
              className={`flex items-center gap-3 rounded-none px-6 py-3 text-sm whitespace-nowrap border-2 transition-all font-bold uppercase tracking-wider ${
                isActive
                  ? "bg-green-600 text-black border-green-400 shadow-neon-md"
                  : "bg-black text-green-500 border-green-700 hover:border-green-500 hover:shadow-neon-sm"
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      <div className="glass-panel p-6 md:p-8">
        {tabs.find((t) => t.id === active)?.content}
      </div>
    </div>
  );
}

