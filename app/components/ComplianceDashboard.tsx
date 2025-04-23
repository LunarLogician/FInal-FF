type ComplianceScore = {
  name: string;
  results: { status: string }[];
};

const getStatusIcon = (compliant: number, total: number) => {
  const ratio = compliant / total;
  if (ratio === 1) return "✅";
  if (ratio >= 0.5) return "⚠️";
  return "❌";
};

export default function ComplianceDashboard({ frameworks }: { frameworks: ComplianceScore[] }) {
  return (
    <div className="p-4 border border-zinc-200 bg-[#f8f9fe] rounded-xl mb-4">
      <h3 className="text-sm font-medium mb-3">📊 Compliance Dashboard</h3>
      <div className="space-y-2">
        {frameworks.map(({ name, results }) => {
          const total = results.length;
          const compliant = results.filter(r => r.status.includes("✅")).length;
          const statusIcon = getStatusIcon(compliant, total);

          return (
            <div key={name} className="flex justify-between items-center bg-white p-3 rounded-lg border">
              <div className="flex items-center gap-2">
                <span className="text-lg">{statusIcon}</span>
                <span className="text-sm font-medium">{name}</span>
              </div>
              <span className="text-sm text-zinc-500">{compliant}/{total} Compliant</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
