'use client';

interface SimpleLineChartProps {
  data: number[];
  labels?: string[];
  title?: string;
  color?: string;
}

export function SimpleLineChart({ data, labels, title, color = '#10b981' }: SimpleLineChartProps) {
  if (!data || data.length === 0) {
    return <div className="text-muted-foreground text-sm">No data available</div>;
  }

  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100;
    const y = 100 - ((value - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="space-y-2">
      {title && <h4 className="text-sm font-semibold">{title}</h4>}
      <div className="relative w-full h-48 border rounded-lg bg-muted/10 p-4">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="100"
              y2={y}
              stroke="currentColor"
              strokeOpacity="0.1"
              strokeWidth="0.5"
            />
          ))}
          
          {/* Line */}
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
          
          {/* Points */}
          {data.map((value, index) => {
            const x = (index / (data.length - 1)) * 100;
            const y = 100 - ((value - min) / range) * 100;
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="1"
                fill={color}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
        
        {/* Labels */}
        <div className="absolute bottom-1 left-4 right-4 flex justify-between text-xs text-muted-foreground">
          <span>Start</span>
          <span>${min.toFixed(2)}</span>
          <span>${max.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}

